"""Production report colors follow decided health, never detector confidence."""
from collections import Counter
import cv2 as cv
import numpy as np

GREEN=(0,220,0)
RED=(0,0,255)
GRAY=(180,180,180)


def health_style(status):
    if status=='normal':return GREEN,'NORMAL','normal'
    if status=='not_flashing':return RED,'NOT FLASHING','failed'
    # Physical absence needs independent day-body/background evidence, never a
    # missed YOLO box or night non-emission alone.
    if status in ('missing','lost'):return RED,'MISSING','failed'
    label={'insufficient_observation':'INSUFFICIENT','unobservable':'UNKNOWN',
           'below_threshold':'LOW FLASHES','no_flash':'NO FLASH YET'}.get(status,'UNKNOWN')
    return GRAY,label,'unknown'


def render_lamp_report(frame,lamps,camera_id,report_at):
    """Return a new canvas and the matching machine-readable legend.

    Red is a decided failure/missing state. One zero-flash window, a gap, and an
    uncertain identity remain gray. Existing health decisions are not changed.
    """
    vis=frame.copy();counts=Counter();unplotted=[]
    ids=[str(l['lamp_id']) for l in lamps]
    suffix=6
    while len({identity[-suffix:] for identity in ids})<len(ids) and suffix<max(map(len,ids),default=0):suffix+=1
    display_ids={identity:'L-'+identity[-suffix:] for identity in ids}
    for lamp in lamps:
        color,label,group=health_style(lamp.get('status'))
        counts[group]+=1
        if not lamp.get('position_current',True):
            unplotted.append(str(lamp['lamp_id']))
            continue
        box=np.asarray(lamp['box'],float)
        if box.shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:continue
        x0,y0,x1,y1=map(int,box)
        cv.rectangle(vis,(x0,y0),(x1,y1),(0,0,0),4)
        cv.rectangle(vis,(x0,y0),(x1,y1),color,2)
        text=display_ids[str(lamp['lamp_id'])]
        location=(max(0,x0),max(13,y0-5))
        cv.putText(vis,text,location,cv.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv.LINE_AA)
        cv.putText(vis,text,location,cv.FONT_HERSHEY_SIMPLEX,.4,color,1,cv.LINE_AA)
    h,w=vis.shape[:2];header=np.zeros((76,w,3),np.uint8)
    scale=max(.3,min(.7,w/1700))
    cv.putText(header,f'Camera {camera_id} | NIGHT REPORT {report_at}',(12,24),0,scale,(255,255,255),1,cv.LINE_AA)
    phrases=[(GREEN,f'GREEN: normal {counts["normal"]}'),(RED,f'RED: confirmed failed/missing {counts["failed"]}'),
             (GRAY,f'GRAY: insufficient/unknown {counts["unknown"]}')]
    x=12
    for color,text in phrases:
        cv.putText(header,text,(x,52),0,scale*.85,color,1,cv.LINE_AA)
        x+=cv.getTextSize(text,0,scale*.85,1)[0][0]+25
    cv.putText(header,'Background = last night frame; report time is shown above',(12,70),0,scale*.65,(210,210,210),1,cv.LINE_AA)
    metadata=dict(policy='decided_health_v1',counts={k:counts[k] for k in ['normal','failed','unknown']},
        bgr_colors=dict(normal=list(GREEN),failed=list(RED),unknown=list(GRAY)),
        display_ids=display_ids,unplotted_ids=unplotted,source_frame_header_added_px=0,bottom_legend_added_px=76,
        rule='normal=green; confirmed failure/missing=red; insufficient/uncertain=gray',
        missing_detection_is_not_confirmed_missing=True)
    return np.vstack([vis,header]),metadata
