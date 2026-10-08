"""Corrected Gray -> temporal mask -> unchanged background rules, batched."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from night_lamp.disturbance import detect_matrix
from background_kernels import replay, BACKEND

BATCH_PIXELS = 4096
CACHE_VERSION = 1


def cache_key(folder, meta):
    digest=hashlib.sha256()
    parameters=dict(meta['parameters'])
    parameters.pop('disturbance_before_background',None)
    digest.update(json.dumps([CACHE_VERSION,parameters,meta['origin_row'],meta['origin_col']],sort_keys=True).encode())
    # Large matrix identity uses filesystem metadata; small inputs use full hashes.
    stat=(folder/'gray_raw.npy').stat()
    digest.update(str((stat.st_size,stat.st_mtime_ns)).encode())
    for name in ('time_s.npy','exposure_offset.npy','background_updates.npy','valid_mask.npy','initial_background.npy'):
        digest.update((folder/name).read_bytes())
    import night_lamp.disturbance as disturbance
    import night_lamp.session as session_code
    from night_lamp.tools import nightly_map
    for path in (Path(__file__),Path(__file__).with_name('background_kernels.py'),
                 Path(__file__).with_name('gray_matrix.py'),Path(disturbance.__file__),
                 Path(session_code.__file__),Path(nightly_map.__file__)):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rebuild(session):
    started=time.perf_counter();folder=session.recorder.folder
    meta=json.loads((folder/'manifest.json').read_text())
    p=meta['parameters'];gray=np.load(folder/'gray_raw.npy',mmap_mode='r')
    t=np.load(folder/'time_s.npy');offset=np.load(folder/'exposure_offset.npy').astype(np.float32)
    updates=np.load(folder/'background_updates.npy');valid=np.load(folder/'valid_mask.npy')
    initial=np.load(folder/'initial_background.npy');h,w,n=gray.shape
    sl=(slice(meta['origin_row'],meta['origin_row']+h),slice(meta['origin_col'],meta['origin_col']+w))
    if not hasattr(session,'_unmasked_statistics'):
        session._unmasked_statistics=tuple(a[sl].copy() for a in (session._on,session._peak,session._base))
    key=cache_key(folder,meta);cache=folder/'masked_cache.json';stats=folder/'masked_statistics.npz'
    packed_path=folder/'disturbance_mask_packed.npy';cached=False
    if cache.exists() and stats.exists() and packed_path.exists():
        try:
            record=json.loads(cache.read_text());stat=packed_path.stat()
            if record=={'key':key,'mask_size':stat.st_size,'mask_mtime_ns':stat.st_mtime_ns}:
                with np.load(stats) as saved:
                    count,total,peak,final=(saved[k].copy() for k in ('count','total','peak','base'))
                cached=all(a.shape==(h,w) for a in (count,total,peak,final))
        except (ValueError,OSError,KeyError):
            cached=False
    reused=rebuilt=0
    if not cached:
        if hasattr(session,'_disturbance_packed'):
            del session._disturbance_packed
        pack=np.lib.format.open_memmap(packed_path,mode='w+',dtype=np.uint8,shape=(h,w,(n+7)//8))
        pack[:]=np.packbits(np.ones(n,bool),bitorder='little')
        count=np.zeros((h,w),np.float32);total=np.zeros((h,w),np.int32)
        peak=np.zeros((h,w),np.float32);final=initial.copy()
        ids=np.flatnonzero(valid.ravel());old_count,old_peak,old_base=session._unmasked_statistics
        can_reuse=getattr(session,'_captured_on_delta',None)==p['on_delta']
        for start in range(0,len(ids),BATCH_PIXELS):
            ys,xs=np.divmod(ids[start:start+BATCH_PIXELS],w)
            corrected=gray[ys,xs].astype(np.float32)-offset[None,:]
            mask=detect_matrix(corrected,t,p['disturbance'])
            pack[ys,xs]=np.packbits(mask,axis=1,bitorder='little')
            affected=mask.any(axis=1) if can_reuse else np.ones(len(ys),bool)
            unmasked=~affected;uy,ux=ys[unmasked],xs[unmasked]
            count[uy,ux]=old_count[uy,ux];total[uy,ux]=n
            peak[uy,ux]=old_peak[uy,ux];final[uy,ux]=old_base[uy,ux]
            reused+=int(unmasked.sum());rebuilt+=int(affected.sum())
            if affected.any():
                ay,ax=ys[affected],xs[affected]
                values=replay(corrected[affected],mask[affected],initial[ay,ax],updates,p['on_delta'])
                for target,value in zip((count,total,peak,final),values):target[ay,ax]=value
            if start%(BATCH_PIXELS*10)==0:
                print(f'  mask/background pixels={start}/{len(ids)} backend={BACKEND}',flush=True)
        pack.flush();del pack
        np.savez_compressed(stats,count=count,total=total,peak=peak,base=final)
        stat=packed_path.stat()
        cache.write_text(json.dumps({'key':key,'mask_size':stat.st_size,'mask_mtime_ns':stat.st_mtime_ns}),encoding='utf-8')
    full_count=np.zeros(session._on.shape,np.float32);full_total=np.zeros(session._on.shape,np.int32)
    full_count[sl]=count;full_total[sl]=total
    session._on=full_count;session._valid_sample_count=full_total
    session._peak[sl]=peak;session._base[sl]=final
    session._disturbance_packed=np.load(packed_path,mmap_mode='r')
    session._disturbance_origin=(meta['origin_row'],meta['origin_col'])
    np.save(folder.parent/'valid_sample_count.npy',full_total)
    meta['processing_order']='exposure correction -> per-pixel disturbance mask -> protected EMA and hits/effective ratio; original warmup median unchanged'
    meta['parameters']['disturbance_before_background']=True
    (folder/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    session.recorder.parameters.update(meta['parameters'])
    performance=dict(backend=BACKEND,batch_pixels=BATCH_PIXELS,cache_hit=cached,
        reused_pixels=reused,replayed_pixels=rebuilt,seconds=time.perf_counter()-started)
    (folder/'masked_performance.json').write_text(json.dumps(performance,indent=2),encoding='utf-8')
    print(f'  masked background/count complete: {performance}',flush=True)
