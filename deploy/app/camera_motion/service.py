"""One camera owns one library; consumers receive immutable versioned results.

Migration candidates never replace the active library before commit. Feedback
must already be expressed in the candidate reference coordinates.
"""
from dataclasses import replace
from uuid import uuid4
import cv2 as cv
import numpy as np
from .contracts import MotionResult, Feedback, MotionStatus
from .params import MotionParams
from .references import ReferenceLibrary
from .state import CameraState
from .store import Store
from .vision import Vision


class OffsetModule:
    def __init__(self, camera_id, params=None, store=None, vision=None):
        self.camera_id = str(camera_id)
        self.params = params or MotionParams()
        self.store = store or Store(self.params, camera_id)
        self.vision = vision or Vision(self.params)
        self.references = ReferenceLibrary(self.params, self.store, self.vision)
        self.state = CameraState(self.params)
        self.latest = None
        self.pending = None
        self._candidate = None
        self._feedback = {}
        self._background = None
        self._iterations = 0
        self._acks = {}
        self._version = 0
        self.commits = []

    def inspect(self, frame, stamp, frame_id, reference_light=None):
        if self.pending is not None:
            return self.pending
        if self.latest and self.latest.frame_id == str(frame_id):
            return self.latest
        quality = dict(self.vision.quality(frame))
        if reference_light is not None:quality['reference_light']=reference_light
        if self.references.anchor is None:
            if quality['ok']:
                self.references.start_view(frame, stamp, quality)
            action, alignment = 'unavailable', None
        else:
            alignment = self.references.align(frame) if quality['ok'] else None
            action = self.state.check(alignment)
            if action == 'aligned' and isinstance(self.vision,Vision):self.vision.quality(frame,learn=True)
            if action == 'aligned' and self.references.refresh_due(quality, stamp):
                self.references.add_appearance(frame, stamp, quality, alignment.matrix)
        self._version += 1
        valid = alignment is not None and alignment.valid and quality['ok']
        self.latest = MotionResult(self.camera_id, '', str(frame_id), stamp,
            self.references.view, self.references.view, self._version, action,
            alignment.matrix.tolist() if valid else None, tuple(frame.shape[:2]),
            float(alignment.details.get('inlier_ratio', 0)) if valid else 0.,
            dict(quality=quality, alignment=alignment.details if alignment else {},
                 overlap=self._overlap(alignment.matrix,frame.shape[:2]) if valid else None))
        return self.latest

    def begin_migration(self, frame, stamp, frame_id, timeout_s=1500.):
        if self.pending is not None:
            raise RuntimeError('Migration already pending')
        if self.references.anchor is None or timeout_s <= 0:
            raise ValueError('An active old reference and positive timeout are required')
        quality = self.vision.quality(frame)
        if not quality['ok']:
            raise ValueError('Candidate reference quality rejected')
        alignment = self.references.align(frame)
        self._version += 1
        candidate = ReferenceLibrary(self.params, self.store, self.vision)
        candidate.start_view(frame, stamp, quality)
        self._candidate = candidate
        self._feedback = {}
        self._iterations = 0
        self._background = self.vision.last_matches
        self.pending = MotionResult(self.camera_id, uuid4().hex, str(frame_id), stamp,
            self.references.view, candidate.view, self._version, MotionStatus.PENDING.value,
            alignment.matrix.tolist() if alignment and alignment.valid else None,
            tuple(frame.shape[:2]), 0., dict(deadline=stamp+timeout_s, accepted=False))
        return self.pending

    def submit(self, feedback: Feedback):
        r = self.pending
        if (r is None or feedback.camera_id != self.camera_id or feedback.task_id != r.task_id
                or feedback.old_reference != r.old_reference or feedback.new_reference != r.new_reference
                or feedback.version != r.version or feedback.frame_id != r.frame_id):
            raise ValueError('Stale or mismatched feedback; normalize to candidate reference first')
        feedback.points()
        self._feedback[feedback.consumer] = feedback

    def refine(self, stamp, *, max_iterations=3, error_px=3., min_points=8,
               min_span=.1, holdout_tolerance_px=1.):
        r = self.pending
        if r is None:
            raise RuntimeError('No pending migration')
        if stamp >= r.details['deadline'] or self._iterations >= max_iterations:
            self.pending = replace(r, status='failed', details={**r.details, 'reason':'timeout_or_iteration_limit'})
            return self.pending
        sources, targets, held_a, held_b = [], [], [], []
        groups = []
        if self._background is not None and r.matrix is not None:
            _, _, a, b = self._background
            # last_matches are in an appearance image; convert to old anchor.
            # Use the actual appearance selected by the library's alignment.
            reference_image = self._background[0]
            for entry in self.references.appearances:
                image = self.store.read_image(entry['image'])
                if image is not None and np.array_equal(image, reference_image):
                    a = cv.perspectiveTransform(np.asarray(a,np.float32)[None],
                        np.linalg.inv(np.asarray(entry['matrix'],float)))[0]
                    groups.append((a, b))
                    break
        for f in self._feedback.values():
            if f.confidence >= .8:
                groups.append(f.points())
        # Balance groups by capping their contribution; reserve every fourth pair.
        for a, b in groups:
            ids = np.linspace(0, len(a)-1, min(len(a), 48), dtype=int)
            a, b = np.asarray(a)[ids], np.asarray(b)[ids]
            held = np.arange(len(a)) % 4 == 0
            sources.extend(a[~held]); targets.extend(b[~held])
            held_a.extend(a[held]); held_b.extend(b[held])
        self._iterations += 1
        if len(sources) < min_points or len(held_a) < 2:
            self.pending = replace(r, details={**r.details, 'accepted':False, 'reason':'insufficient_independent_support'})
            return self.pending
        a, b = np.asarray(sources,float), np.asarray(targets,float)
        h, w = r.shape
        if min(np.ptp(a,axis=0)/[w,h]) < min_span or min(np.ptp(b,axis=0)/[w,h]) < min_span:
            self.pending = replace(r, details={**r.details, 'accepted':False, 'reason':'poor_spatial_coverage'})
            return self.pending
        affine, mask = cv.estimateAffinePartial2D(a,b,method=cv.RANSAC,ransacReprojThreshold=error_px)
        if affine is None:
            self.pending=replace(r,details={**r.details,'accepted':False,'reason':'ransac_failed'})
            return self.pending
        matrix = np.eye(3); matrix[:2] = affine
        ha, hb = np.asarray(held_a,float), np.asarray(held_b,float)
        errors = np.linalg.norm(cv.perspectiveTransform(ha[None],matrix)[0]-hb,axis=1)
        previous = (np.linalg.norm(cv.perspectiveTransform(ha[None],np.asarray(r.matrix))[0]-hb,axis=1)
                    if r.matrix is not None else np.full(len(ha),np.inf))
        scale = np.hypot(affine[0,0],affine[1,0])
        accepted = (mask.mean() >= .8 and np.max(errors) <= error_px
            and .5 <= scale <= 2 and np.max(errors-previous) <= holdout_tolerance_px)
        self.pending = replace(r, version=r.version+1, matrix=matrix.tolist() if accepted else r.matrix,
            confidence=float(mask.mean()), details={**r.details, 'accepted':bool(accepted),
            'iteration':self._iterations, 'holdout_max_px':float(errors.max()),
            'reason':'accepted' if accepted else 'validation_failed'})
        self._feedback.clear()
        return self.pending

    def commit(self):
        r = self.pending
        if r is None or not r.details.get('accepted') or r.status == 'failed':
            raise RuntimeError('Migration has not passed validation')
        # Durable candidate metadata is recorded before switching the active library.
        self.store.record('migrations', r.checked_at, r.to_dict())
        # Preserve the shared library object held by existing consumers.
        for name in ('view','anchor','appearances','last_refresh','_light','_light_count'):
            setattr(self.references,name,getattr(self._candidate,name))
        self.state = CameraState(self.params)
        self._version = max(self._version, r.version)
        self.latest = replace(r, status='committed')
        self.commits.append(self.latest)
        self.commits = self.commits[-64:]
        self.references.migration_sources={v.old_reference for v in self.commits}
        self.pending = self._candidate = None
        return self.latest

    def acknowledge(self, consumer, task_id, version):
        r = next((v for v in reversed(self.commits) if (v.task_id,v.version)==(task_id,version)),None)
        if not r or r.status != 'committed' or (task_id,version) != (r.task_id,r.version):
            raise ValueError('Unknown committed migration')
        key = (consumer, task_id)
        fresh = key not in self._acks
        self._acks[key] = version
        return fresh

    def abandon(self, reason='association_unavailable'):
        if self.pending is None:
            return None
        self.latest = replace(self.pending,status='failed',details={**self.pending.details,
            'accepted':False,'reason':reason})
        self.pending = self._candidate = None
        return self.latest

    def dump(self):
        return dict(schema=1,camera_id=self.camera_id,version=self._version,
            reference=dict(view=self.references.view,appearances=self.references.appearances,
                           last_refresh=self.references.last_refresh),
            latest=self.latest.to_dict() if self.latest else None,
            acknowledgements=[list(key)+[value] for key,value in self._acks.items()],
            interrupted_task=self.pending.task_id if self.pending else None,
            commits=[r.to_dict() for r in self.commits])

    def restore(self,data,stamp):
        if data.get('schema')!=1 or data.get('camera_id')!=self.camera_id:
            raise ValueError('Incompatible offset checkpoint')
        ref=data['reference']
        # Validate all referenced files before publishing restored state.
        appearances=[]
        for entry in ref['appearances']:
            image=self.store.read_image(entry['image'])
            if image is None:raise ValueError('Missing active reference image')
            matrix=np.asarray(entry['matrix'],float)
            if matrix.shape!=(3,3) or not np.isfinite(matrix).all():
                raise ValueError('Invalid reference matrix')
            appearances.append(entry)
        if not appearances:return False
        self.references.view=ref['view']
        self.references.appearances=appearances
        self.references.last_refresh=ref['last_refresh']
        self.references.anchor=self.store.read_image(appearances[0]['image'])
        self.commits=[MotionResult(**v) for v in data.get('commits',[])]
        self.references.migration_sources={v.old_reference for v in self.commits}
        self._version=int(data['version'])
        self.latest=MotionResult(**data['latest']) if data.get('latest') else None
        if self.latest is not None:
            self.latest=replace(self.latest,status='unavailable',matrix=None,frame_id='restored:'+self.latest.frame_id)
        self._acks={(c,t):v for c,t,v in data.get('acknowledgements',[])}
        # A transaction interrupted before commit must be re-observed, never replayed as accepted.
        self.pending=self._candidate=None
        self.state=CameraState(self.params)
        return True

    @staticmethod
    def _overlap(matrix,shape):
        h,w=shape
        corners=np.float32([[0,0],[w,0],[w,h],[0,h]])
        projected=cv.perspectiveTransform(corners[None],np.asarray(matrix,float))[0]
        area,polygon=cv.intersectConvexConvex(projected,corners)
        return dict(area_ratio=float(area/(w*h)),polygon=polygon.reshape(-1,2).tolist() if polygon is not None else [])

    def commit_unmapped(self, reason):
        """Adopt a stable new view without inventing historical correspondence."""
        if self.pending is None or self._candidate is None:
            raise RuntimeError('No candidate reference')
        self.pending=replace(self.pending,matrix=None,status='migration_pending',
            details={**self.pending.details,'accepted':True,'reason':reason})
        return self.commit()
