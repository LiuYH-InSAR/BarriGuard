"""Pure bounded search planning. No track, identity or flash-state dependencies."""
from dataclasses import dataclass


def overlap(a,b):
    n=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return n/max(1e-9,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-n)


@dataclass(frozen=True)
class SearchPlan:
    timestamp:float
    regions:tuple
    sources:tuple
    next_cursor:int
    budget:int
    policy:str


def plan_search(shape,proposals,tested,cursor,ts,*,size,budget,survey_budget,survey,exclusions,policy='legacy'):
    """Return a plan without modifying any input.

    legacy preserves established ordering exactly. bounded reserves independent
    survey slots first, with at most one full-grid traversal to fill them. All
    remaining image/history/verification proposals share the remaining budget.
    The bounded policy is experimental; legacy remains the production default.
    """
    if policy not in {'legacy','bounded'}:raise ValueError('Unknown search policy')
    if budget<=0 or not 0<=survey_budget<=budget:raise ValueError('Invalid search budget')
    h,w=shape[:2];size=min(size,h,w)
    ranked=[]
    for score,c,source in proposals:
        x=max(0,min(w-size,int(c[0])-size//2));y=max(0,min(h-size,int(c[1])-size//2))
        roi=(x,y,x+size,y+size);age=ts-tested.get(roi[:2],-1e6)
        ranked.append((score+min(age,5)*.2,roi,source))
    ranked.sort(key=lambda x:x[0],reverse=True)
    rois=[];sources=[]
    def add(roi,source):
        if any(overlap(roi,r)>.45 for r in rois):return False
        rois.append(tuple(roi));sources.append(source);return True
    def allowed(roi):
        x=(roi[0]+roi[2])/2;y=(roi[1]+roi[3])/2
        return not any(a*w<=x<=c*w and b*h<=y<=d*h for a,b,c,d in exclusions)
    if policy=='legacy':
        for _,roi,source in ranked:
            add(roi,source)
            if len(rois)>=budget-survey_budget:break
        for _ in range(survey_budget):
            roi=survey[cursor%len(survey)];cursor+=1
            if allowed(roi):add(roi,'survey')
    else:
        for _ in range(len(survey)):
            if len(rois)>=survey_budget:break
            roi=survey[cursor%len(survey)];cursor+=1
            if allowed(roi):add(roi,'survey')
        for _,roi,source in ranked:
            if len(rois)>=budget:break
            add(roi,source)
    return SearchPlan(ts,tuple(rois[:budget]),tuple(sources[:budget]),cursor,budget,policy)
