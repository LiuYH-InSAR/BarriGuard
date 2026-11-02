"""Long-window fixes: verified fixture ownership and repeated-content pause."""
import numpy as np
from .adaptive_v7 import AdaptiveV7
from .adaptive_v6 import V6Params,ObservationAuditV6
from .adaptive_v5 import AdaptiveV5
from .adaptive_prototype import center,iou
from .nested_fixture_v6 import containment,area
from night_lamp.quality import StreamQuality


class ContentEvidenceQuality(StreamQuality):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.full_previous=None

    def observe(self,frame,source_ts,frame_id):
        repeated=self.full_previous is not None and np.array_equal(frame,self.full_previous)
        result=super().observe(frame,source_ts,frame_id)
        self.full_previous=frame.copy()
        result['exact_content_repeated']=bool(repeated)
        # Exact repeat is not proof of network failure, but supplies no new
        # visual evidence. Pause immediately instead of crediting the 5-second
        # suspicion delay. A first new image and subsequent changes remain usable.
        if repeated:result['fresh']=False
        return result


class AdaptiveV8(AdaptiveV7):
    def __init__(self,backend,params=None,vision=None):
        super().__init__(backend,params,vision)
        self.quality=ContentEvidenceQuality(max_gap_s=self.p.max_gap_s)

    def associate(self,frame,detections,ts,night):
        consumed=set();owned=set()
        for key in list(self.fixture_regions):
            if key not in self.tracks:
                self.fixture_regions.pop(key);continue
            t=self.tracks[key];outer=self.envelope(t);matches=[]
            for j,box in enumerate(detections):
                if j in consumed:continue
                family=(iou(t.body_box,box)>=.5
                    or (iou(outer,box)>=.5 and containment(outer,box)>=.9)
                    or (area(box)<=area(t.body_box) and containment(t.body_box,box)>=.9))
                if family and np.linalg.norm(center(box)-center(t.body_box))<=.4*max(outer[2]-outer[0],outer[3]-outer[1]):
                    matches.append(j)
            if not matches:continue
            best=max(matches,key=lambda j:(iou(t.body_box,detections[j]),detections[j][4]))
            box=detections[best]
            # Ownership is already proved by the full fixture relation. A
            # body-only detection need not pass a second, smaller head-center
            # distance gate; that was respawning the same body every few seconds.
            t.hits+=1;t.last_seen=ts;t.confidence=max(t.confidence,box[4])
            t.fine_hits+=int(box[5]=='fine')
            owned.add(key);consumed.update(matches)
        remaining=[box for j,box in enumerate(detections) if j not in consumed]
        AdaptiveV5.associate(self,frame,remaining,ts,night,reserved_ids=owned)


class LampPipelineV8:
    def __init__(self,backend,params=None,vision=None):
        self.detector=AdaptiveV8(backend,params or V6Params(),vision)
        self.audit=ObservationAuditV6()

    def step(self,frame,ts,frame_id,night=False):
        return self.audit.apply(self.detector.step(frame,ts,frame_id,night))
