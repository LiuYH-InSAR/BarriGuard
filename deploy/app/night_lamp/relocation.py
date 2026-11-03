"""Independent new-view discovery and conservative persistent-ID correspondence."""
from collections import deque
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
from camera_motion.contracts import Feedback
from camera_motion.geometry import warp_boxes
from .adaptive import LampPipelineV9
from .landmarks import inside,matrix,midpoint,compare


def constellation(old,new,shape,error_px=3.):
    """Bounded similarity hypotheses; repeated/collinear arrangements abstain."""
    a=np.asarray(old,float);b=np.asarray(new,float)
    if len(a)<4 or len(b)<4 or max(len(a),len(b))>64:return None
    h,w=shape[:2]
    if min(np.ptp(a,axis=0)/[w,h])<.05 or min(np.ptp(b,axis=0)/[w,h])<.05:return None
    def pairs(points,limit):
        values=[(i,j) for i in range(len(points)) for j in range(i+1,len(points))
                if np.linalg.norm(points[i]-points[j])>max(w,h)*.08]
        if len(values)>limit:values=[values[i] for i in np.linspace(0,len(values)-1,limit,dtype=int)]
        return values
    hypotheses=[]
    for i,j in pairs(a,64):
        z=complex(*(a[j]-a[i]))
        for k,l in pairs(b,128):
            for k,l in [(k,l),(l,k)]:
                ratio=complex(*(b[l]-b[k]))/z
                if not .5<=abs(ratio)<=2. or abs(np.angle(ratio))>np.pi/3:continue
                affine=np.array([[ratio.real,-ratio.imag],[ratio.imag,ratio.real]])
                translation=b[k]-affine@a[i]
                hypotheses.append((affine,translation))
    if len(hypotheses)>4096:hypotheses=[hypotheses[i] for i in np.linspace(0,len(hypotheses)-1,4096,dtype=int)]
    solutions=[];minimum=max(4,int(np.ceil(min(len(a),len(b))*.6)))
    for affine,translation in hypotheses:
        predicted=a@affine.T+translation
        distances=np.linalg.norm(predicted[:,None,:]-b[None,:,:],axis=2)
        nearest=distances.argmin(axis=1);back=distances.argmin(axis=0)
        ids=np.array([i for i,j in enumerate(nearest) if back[j]==i and distances[i,j]<=error_px])
        if len(ids)<minimum:continue
        fitted,inliers=cv.estimateAffinePartial2D(a[ids],b[nearest[ids]],method=cv.RANSAC,ransacReprojThreshold=error_px)
        if fitted is None or inliers.mean()<.9:continue
        fitted_prediction=a@fitted[:,:2].T+fitted[:,2]
        if any(np.max(np.linalg.norm(fitted_prediction-s[2],axis=1))<error_px for s in solutions):continue
        residual=np.linalg.norm(fitted_prediction[ids]-b[nearest[ids]],axis=1)
        solutions.append((len(ids),float(np.median(residual)),fitted_prediction,fitted))
    if not solutions:return None
    solutions.sort(key=lambda s:(-s[0],s[1]));best=solutions[0]
    if len(solutions)>1 and solutions[1][0]>=best[0]-1:return None
    transform=np.eye(3);transform[:2]=best[3]
    return transform


def correspondences(lamps,candidates,transform,shape,error_px=3.,frame=None,view_to_frame=None,light='day'):
    if transform is None or not lamps or not candidates:return {}
    transform=matrix(transform);keys=list(lamps);values=list(candidates)
    old=[lamps[k].spatial['anchor_body_box'] for k in keys]
    predicted=warp_boxes(old,transform);cost=np.full((len(keys),len(values)),1e6)
    centers=np.array([midpoint(b) for b in predicted])
    for i,box in enumerate(predicted):
        if not inside(box,shape):continue
        size=np.maximum(np.asarray(box[2:4])-box[:2],1.)
        spacing=min((np.linalg.norm(centers[i]-c) for j,c in enumerate(centers) if i!=j),default=np.inf)
        radius=min(max(error_px*2.,.55*min(size)),spacing*.35)
        for j,key in enumerate(values):
            candidate=candidates[key];other=candidate['body_box']
            ratio=(np.asarray(other[2:4])-other[:2])/size
            distance=np.linalg.norm(midpoint(other)-centers[i])
            if distance>radius or np.any(ratio<.5) or np.any(ratio>2.):continue
            cost[i,j]=distance
    if not np.any(cost<1e6):return {}
    rows,columns=linear_sum_assignment(cost);matched={}
    for i,j in zip(rows,columns):
        distance=cost[i,j]
        if distance>=1e6:continue
        alternatives=np.r_[np.delete(cost[i],j),np.delete(cost[:,j],i)]
        if alternatives.size and np.min(alternatives)-distance<max(2.,error_px*.75):continue
        lamp=lamps[keys[i]];profile=lamp.spatial.get('profiles',{}).get(light)
        if profile is not None and frame is not None:
            evidence=compare(frame,profile,matrix(view_to_frame)@transform)
            if evidence.get('context_correlation',0)<.75 or evidence.get('context_error',999)>18:continue
        matched[keys[i]]=values[j]
    return matched


class RelocationSession:
    def __init__(self,controller,result,reference_frame,night):
        self.controller=controller;self.task=result.task_id;self.reference_frame=reference_frame.copy()
        self.night=bool(night);self.pipeline=LampPipelineV9(controller.detector,controller.vision_params)
        self.pipeline.step(self.reference_frame,result.details.get('source_ts',result.checked_at),'migration-reference:'+result.frame_id,night=self.night)
        self.samples={};self.candidates={};self.last_frame=None;self.last_stamp=None
        self.frame_to_candidate=np.eye(3);self.invalid=False
        self._hypothesis=None;self._hypothesis_at=None;self._hypothesis_keys=None

    def old_lamps(self,result):
        return {k:l for k,l in self.controller.monitor.lamps.items() if l.reference==result.old_reference
            and l.spatial.get('reference')==result.old_reference and 'anchor_body_box' in l.spatial}

    def observe(self,result,frame,stamp,source_ts,frame_id):
        if result.task_id!=self.task:raise ValueError('Migration candidate task changed')
        out=self.pipeline.step(frame,source_ts,frame_id,night=self.night)
        if out.get('duplicate_ignored'):return
        if any(e['event']=='geometry_reset' for e in out.get('events',[])):
            self.invalid=True;self.candidates.clear();return
        q=out['quality']
        if (self.invalid or not all(q.get(k) for k in ['fresh','continuous','clear'])
                or out.get('motion')!='aligned'):
            self.candidates.clear();return
        inverse=np.linalg.inv(matrix(self.pipeline.detector.applied_transform))
        self.frame_to_candidate=inverse;self.last_frame=frame;self.last_stamp=stamp
        current={};cfg=self.controller.monitor.params
        for r in out['tracks']:
            if not(r.get('evidence_admitted') and r.get('confirmed') and r.get('position_observable')
                   and r.get('visible_recent') and r.get('hits',0)>=self.controller.vision_params.promotion_hits):continue
            if self.night and r.get('raw_pulses',r.get('pulses',0))<cfg.normal_pulses:continue
            token=(r['id'],r.get('observation_epoch',0))
            history=self.samples.setdefault(token,deque(maxlen=80))
            # Position samples may survive short gaps after re-registration.
            # Optical cycles still use their own strict continuity counter.
            if history and stamp-history[-1]['at']>10.:history.clear()
            while history and stamp-history[0]['at']>30.:history.popleft()
            body,box=warp_boxes([r.get('body_box',r['box']),r['box']],inverse)
            history.append(dict(at=stamp,body=body,box=box,hits=r['hits']))
            if len(history)<cfg.migration_min_samples or stamp-history[0]['at']<cfg.migration_min_span_s:continue
            bodies=np.array([v['body'] for v in history]);centers=(bodies[:,:2]+bodies[:,2:])/2
            if np.max(np.linalg.norm(centers-np.median(centers,axis=0),axis=1))>cfg.migration_error_px:continue
            if max(v['hits'] for v in history)-min(v['hits'] for v in history)<2:continue
            key=f'{self.task}:T{r["id"]}:E{token[1]}'
            current[key]=dict(body_box=np.median(bodies,axis=0).tolist(),
                box=np.median([v['box'] for v in history],axis=0).tolist(),
                local_id=r['id'],samples=len(history),span_s=stamp-history[0]['at'],
                independent_hits=r['hits'],cycles=r.get('raw_pulses',r.get('pulses',0)),
                observed_at=stamp,admitted=True,night=self.night)
        self.candidates=current

    def ready(self,result):
        lamps=self.old_lamps(result)
        if not lamps or self.invalid:return True
        transform=self.transform(result)
        if transform is None:return False
        visible=[b for b in warp_boxes([l.spatial['anchor_body_box'] for l in lamps.values()],transform) if inside(b,result.shape)]
        return not visible or len(self.matches(result))>=min(4,len(visible))

    def matches(self,result):
        if self.invalid or self.last_frame is None:return {}
        return correspondences(self.old_lamps(result),self.candidates,self.transform(result),result.shape,
            self.controller.monitor.params.migration_error_px,self.last_frame,
            np.linalg.inv(self.frame_to_candidate),'night' if self.night else 'day')

    def transform(self,result):
        if result.matrix is not None:return matrix(result.matrix)
        token=tuple(sorted(self.candidates))
        if (self._hypothesis_keys!=token or self._hypothesis_at is None
            or (self.last_stamp is not None and self.last_stamp-self._hypothesis_at>=5.)):
            lamps=self.old_lamps(result)
            self._hypothesis=constellation([midpoint(l.spatial['anchor_body_box']) for l in lamps.values()],
                [midpoint(c['body_box']) for c in self.candidates.values()],result.shape,
                self.controller.monitor.params.migration_error_px)
            self._hypothesis_keys=token;self._hypothesis_at=self.last_stamp
        return self._hypothesis

    def feedback(self,result):
        matches=self.matches(result)
        if not matches:return None
        lamps=self.old_lamps(result);old=[];new=[];proof=[]
        for key,candidate_key in matches.items():
            c=self.candidates[candidate_key];old.append(midpoint(lamps[key].spatial['anchor_body_box']).tolist())
            new.append(midpoint(c['body_box']).tolist())
            proof.append(dict(lamp_id=key,candidate_id=candidate_key,**c))
        return Feedback(result.camera_id,result.task_id,'night_lamp',result.old_reference,result.new_reference,
            result.version,result.frame_id,tuple(map(tuple,old)),tuple(map(tuple,new)),
            evidence=dict(source='independent_candidate_pipeline',matches=proof))


def committed_matches(result,lamps,params):
    """Revalidate durable independent evidence, including after owner restart."""
    source=result.details.get('feedback',{}).get('night_lamp',{})
    evidence=source.get('evidence',{})
    if not source or evidence.get('source')!='independent_candidate_pipeline':return {}
    if (source.get('camera_id')!=result.camera_id or source.get('task_id')!=result.task_id
        or source.get('old_reference')!=result.old_reference or source.get('new_reference')!=result.new_reference
        or source.get('frame_id')!=result.frame_id or source.get('version')!=result.version-1
        or source.get('independent') is not True or source.get('consumer')!='night_lamp'
        or source.get('confidence',0)<.8 or source.get('coordinate_space')!='new_reference_pixels'):return {}
    found={};used=set();proof=evidence.get('matches',[])
    if (len({p.get('lamp_id') for p in proof})!=len(proof)
        or len({p.get('candidate_id') for p in proof})!=len(proof)):return {}
    old_points=np.asarray(source.get('old_points',[]),float);new_points=np.asarray(source.get('observed_points',[]),float)
    if old_points.shape!=(len(proof),2) or new_points.shape!=old_points.shape:return {}
    if (not np.isfinite(old_points).all() or not np.isfinite(new_points).all()
        or len(np.unique(old_points,axis=0))!=len(proof) or len(np.unique(new_points,axis=0))!=len(proof)):return {}
    for index,p in enumerate(proof):
        key=p.get('lamp_id');candidate=p.get('candidate_id')
        if (key not in lamps or key in found or candidate in used or not p.get('admitted')
            or p.get('samples',0)<params.migration_min_samples or p.get('span_s',0)<params.migration_min_span_s
            or p.get('independent_hits',0)<3 or (p.get('night') and p.get('cycles',0)<params.normal_pulses)
            or not inside(p.get('body_box',[]),result.shape) or not inside(p.get('box',[]),result.shape)
            or not result.checked_at<=p.get('observed_at',0)<=result.details.get('deadline',0)):continue
        anchor=lamps[key].spatial.get('anchor_body_box')
        if anchor is None or not np.allclose(midpoint(anchor),old_points[index],atol=.01):continue
        if not np.allclose(midpoint(p['body_box']),new_points[index],atol=.01):continue
        projected=cv.perspectiveTransform(old_points[index:index+1][None],matrix(result.matrix))[0,0]
        if np.linalg.norm(projected-new_points[index])>params.migration_error_px:continue
        found[key]=p;used.add(candidate)
    return found
