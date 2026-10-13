"""Equivalent background replay with optional compiled scalar arithmetic."""
import numpy as np
try:
    from numba import njit
except (ImportError, OSError):
    # Optional native DLLs may be blocked by Windows application control.
    njit = None


def numpy_replay(x, mask, initial, updates, delta):
    base=initial.copy();count=np.zeros(len(x),np.float32)
    total=np.zeros(len(x),np.int32);peak=np.zeros(len(x),np.float32)
    for i in range(x.shape[1]):
        usable=~mask[:,i];ex=x[:,i]-base;hit=usable&(ex>=delta)
        count+=hit;total+=usable;peak=np.maximum(peak,np.where(usable,ex,0))
        if updates[i,0]:base[usable]=x[usable,i]
        else:
            update=usable&~hit;alpha=np.float32(updates[i,1])
            base[update]=alpha*x[update,i]+np.float32(1-alpha)*base[update]
    return count,total,peak,base


def _scalar_replay(x, mask, initial, updates, delta):
    base=initial.copy();count=np.zeros(len(x),np.float32)
    total=np.zeros(len(x),np.int32);peak=np.zeros(len(x),np.float32)
    for j in range(len(x)):
        value=base[j]
        for i in range(x.shape[1]):
            if mask[j,i]:continue
            ex=np.float32(x[j,i]-value);hit=ex>=delta
            total[j]+=1
            if hit:count[j]+=np.float32(1)
            if ex>peak[j]:peak[j]=ex
            if updates[i,0]:value=x[j,i]
            elif not hit:
                alpha=np.float32(updates[i,1])
                left=np.float32(alpha*x[j,i])
                right=np.float32(np.float32(1-alpha)*value)
                value=np.float32(left+right)
        base[j]=value
    return count,total,peak,base


compiled_replay=njit(cache=True,fastmath=False)(_scalar_replay) if njit else None
BACKEND='numba' if compiled_replay is not None else 'numpy'


def replay(*args):
    return (compiled_replay if compiled_replay is not None else numpy_replay)(*args)
