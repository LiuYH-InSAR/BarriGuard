"""Quality evidence for a lamp window; no claim of semantic occlusion detection."""
from dataclasses import dataclass
import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class ImageMetrics:
    """Stateless image work shared only inside one controller frame call."""
    image_id: int
    source_ts: float
    frame_id: object
    gray: object
    small: object
    blur: object
    contrast: object

    def matches(self,frame,source_ts,frame_id):
        return self.image_id==id(frame) and self.source_ts==source_ts and self.frame_id==frame_id


@dataclass(frozen=True)
class ObservationQuality:
    fresh: bool
    source_continuous: bool
    clear: bool
    position_trusted: bool
    visible: bool | None
    reason: str = ''

    @property
    def usable(self):
        return bool(self.fresh and self.source_continuous and self.clear
                    and self.position_trusted and self.visible is True)


class StreamQuality:
    def __init__(self, max_gap_s=.5, freeze_s=5., blur_floor=12., contrast_floor=12.):
        self.max_gap_s,self.freeze_s=max_gap_s,freeze_s
        self.blur_floor,self.contrast_floor=blur_floor,contrast_floor
        self.previous=None
        self.previous_id=None
        self.previous_ts=None
        self.unchanged_since=None
        self.prepared_metrics=None
        self.last_metrics=None

    def observe(self, frame, source_ts, frame_id):
        fresh=(frame_id is not None and frame_id!=self.previous_id and
               (self.previous_ts is None or source_ts>self.previous_ts))
        gap=(None if self.previous_ts is None else source_ts-self.previous_ts)
        continuous=gap is not None and 0<gap<=self.max_gap_s
        metrics=self.prepared_metrics
        if metrics is None or not metrics.matches(frame,source_ts,frame_id):
            gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            small=cv.resize(gray,(160,90))
            blur=cv.Laplacian(gray,cv.CV_32F).var()
            contrast=gray.std() if blur>=self.blur_floor else None
            metrics=ImageMetrics(id(frame),source_ts,frame_id,gray,small,blur,contrast)
        self.last_metrics=metrics
        small=metrics.small
        same=self.previous is not None and np.array_equal(small,self.previous)
        if not same:self.unchanged_since=source_ts
        frozen=same and self.unchanged_since is not None and source_ts-self.unchanged_since>=self.freeze_s
        clear=bool(metrics.blur>=self.blur_floor and
                   (metrics.contrast if metrics.contrast is not None else metrics.gray.std())>=self.contrast_floor)
        if fresh:
            self.previous,self.previous_id,self.previous_ts=small,frame_id,source_ts
        # Exact content repetition is a conservative freeze suspicion, not proof.
        return dict(fresh=bool(fresh and not frozen),continuous=bool(continuous),clear=bool(clear),
                    freeze_suspected=bool(frozen))
