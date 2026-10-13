"""Export Gray diagnostics at a zero-based original-video row/column."""
import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from night_lamp.evidence import REGION_RADIUS, state_sequence, state_means
from night_intermediates import _plot_svg
from gray_stages import stage_details


def export_position(folder, row, col, output=None, *, compare_actual=True):
    folder = Path(folder)
    meta = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    gray = np.load(folder / 'gray_raw.npy', mmap_mode='r')
    valid = np.load(folder / 'valid_mask.npy')
    y, x = row - meta['origin_row'], col - meta['origin_col']
    if not (0 <= y < valid.shape[0] and 0 <= x < valid.shape[1]) or not valid[y, x]:
        raise ValueError('Position is outside saved valid ROI; use original-video zero-based coordinates.')
    times = np.load(folder / 'time_s.npy')
    offset = np.load(folder / 'exposure_offset.npy')
    raw = gray[y, x].astype(np.float64)
    corrected = raw - offset
    y0, y1 = max(0, y - REGION_RADIUS), min(valid.shape[0], y + REGION_RADIUS + 1)
    x0, x1 = max(0, x - REGION_RADIUS), min(valid.shape[1], x + REGION_RADIUS + 1)
    region = gray[y0:y1, x0:x1][valid[y0:y1, x0:x1]].mean(axis=0)
    regional = region - offset
    bgr_path = folder / 'bgr_raw.npy'
    pixel_bgr = region_bgr = None
    if bgr_path.exists():
        bgr = np.load(bgr_path, mmap_mode='r')
        pixel_bgr = bgr[y,x].astype(np.float64) - offset[:,None]
        region_bgr = bgr[y0:y1,x0:x1][valid[y0:y1,x0:x1]].mean(axis=0) - offset[:,None]
        # Match detector's weighted regional BGR Gray (OpenCV pixel Gray rounds).
        regional = region_bgr @ np.array([.114,.587,.299])
    background = excess = hit = None
    saved_mask=None
    if (folder/'disturbance_mask_packed.npy').exists():
        packed=np.load(folder/'disturbance_mask_packed.npy',mmap_mode='r')
        saved_mask=np.unpackbits(packed[y,x],bitorder='little')[:len(raw)].astype(bool)
    if (folder / 'initial_background.npy').exists() and (folder / 'background_updates.npy').exists():
        base = np.float32(np.load(folder / 'initial_background.npy', mmap_mode='r')[y,x])
        updates = np.load(folder / 'background_updates.npy')
        delta = meta['parameters']['on_delta']
        background = np.empty(len(raw)); excess = np.empty(len(raw)); hit = np.empty(len(raw),bool)
        for i,(reset,alpha) in enumerate(updates[:,:2]):
            det = np.float32(corrected[i])
            background[i] = base
            excess[i] = np.float32(det-base)
            usable=saved_mask is None or not saved_mask[i]
            hit[i] = usable and excess[i]>=delta
            if not usable:continue
            if reset:
                base = det
            elif not hit[i]:
                base = np.float32(np.float32(alpha)*det+np.float32(1-alpha)*base)
    states = state_sequence(regional)
    out = Path(output) if output else folder / 'positions' / f'row_{row}_col_{col}'
    out.mkdir(parents=True, exist_ok=True)
    values = np.column_stack((times, raw, corrected, region, regional, offset,
                              np.r_[np.nan, np.diff(corrected)], states))
    np.savetxt(out / 'series.csv', values, delimiter=',', fmt='%.6f', comments='',
        header='time_s,pixel_gray_raw,pixel_gray_corrected,region_gray_raw,region_gray_corrected,exposure_offset,pixel_gray_change,region_on_state')
    np.savez_compressed(out / 'series.npz', time_s=times, pixel_gray_raw=raw,
        pixel_gray_corrected=corrected, region_gray_raw=region,
        region_gray_corrected=regional, exposure_offset=offset, region_on_state=states)
    extra = {}
    if pixel_bgr is not None:
        extra.update(pixel_bgr_corrected=pixel_bgr, region_bgr_corrected=region_bgr)
    if background is not None:
        extra.update(pixel_background=background, pixel_excess=excess, pixel_on_hit=hit)
        extra.update(background_reset_after_hit=updates[:,0], background_ema_alpha=updates[:,1])
        if updates.shape[1]>=3:
            extra['frame_global_median'] = updates[:,2]
        np.savetxt(out/'background_series.csv',np.column_stack((times,background,excess,hit,updates)), delimiter=',',comments='',fmt='%.8f',header='time_s,pixel_background,pixel_excess,on_hit,'+','.join(meta['background_update_columns']))
    if extra:
        np.savez_compressed(out / 'diagnostics.npz', time_s=times, **extra)
    item = {'reason': f'row={row}, col={col}; state assignment is not a lamp verdict'}
    report_path = folder.parent / 'final.json'
    mref = meta.get('mref')
    if mref is None and report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        mref = report.get('metadata', {}).get('night_med')
    if mref is not None:
        item['mref'] = float(mref)
    regional_means = state_means(regional)
    pixel_means = state_means(corrected)
    discovery = None if hit is None else {'exceed_count':int(hit.sum()), 'total':len(hit), 'ratio':float(hit.mean()), 'source':'replayed saved online background','_hits':hit,'_mask':saved_mask}
    region_steps, region_lines = stage_details(folder, meta, raw, corrected, regional, mref, row, col, region_bgr, discovery)
    pixel_steps, pixel_lines = stage_details(folder, meta, raw, corrected, corrected, mref, row, col, pixel_bgr, discovery)
    from night_lamp.disturbance import detect
    mask_data={}
    for label,signal,steps in [('pixel',corrected,pixel_steps),('region',regional,region_steps)]:
        masked,mean,info=detect(signal,times,steps['disturbance']['parameters'])
        info.update({k:steps['disturbance'][k] for k in ('suspected_stutter','normal_mask_duration_ratio','mask_duration_ratio','final_mask_duration_ratio','trigger_pixel_full_run_mask_ratio','trigger_mask_threshold','fallback_settings') if k in steps['disturbance']})
        if saved_mask is not None:
            from night_lamp.disturbance import describe_mask
            masked|=saved_mask;info.update(describe_mask(masked,times))
        off_mean=steps.get('states',{}).get('off_mean')
        if off_mean is not None:
            info.update(off_mean=off_mean,c_reference=info['reference_mean']-off_mean)
        mask_data[label+'_mask']=masked;mask_data[label+'_window_mean']=mean
        deviation=mean-info.get('reference_mean',np.nan)
        np.savetxt(out/(label+'_disturbance.csv'),np.column_stack((times,signal,mean,deviation,masked)),delimiter=',',comments='',header='time_s,gray_corrected,window_mean,deviation_from_reference,masked',fmt='%.8f')
        (out/(label+'_mask_intervals.json')).write_text(json.dumps(info,indent=2),encoding='utf-8')
    np.savez_compressed(out/'disturbance.npz',time_s=times,**mask_data)
    for path,signal,steps,lines in [(out/'region_gray.svg',regional,region_steps,region_lines),(out/'pixel_gray.svg',corrected,pixel_steps,pixel_lines)]:
        means={k:steps.get('states',{}).get(k) for k in ('off_mean','on_mean')}
        info=steps['disturbance']
        mask_label=f'{"fallback" if info.get("suspected_stutter") else "normal"} {info["parameters"]["window_s"]:g}s final mask (including pixel mask union)'
        from plot_table import report_table
        stats=steps.get('discovery',{})
        if (folder.parent/'on_count.npy').exists() and (folder.parent/'valid_sample_count.npy').exists():
            count=float(np.load(folder.parent/'on_count.npy',mmap_mode='r')[row,col])
            total=int(np.load(folder.parent/'valid_sample_count.npy',mmap_mode='r')[row,col])
            stats=dict(count=count,total=total,ratio=count/total if total else 0.)
        table=report_table(steps['parameters'],info,steps.get('states',{}),steps.get('periodicity',{}),stats,mref,float(signal.mean()))
        extra=[f'A: col={col}, row={row}; path={"fallback" if info.get("suspected_stutter") else "normal"}; observed={times[0]:.2f}..{times[-1]:.2f}s',
               'Ratio: full-run pixel background counts. Fit/AC: plotted signal with final mask.',
               f'Query temporal verdict: {"PASS" if steps.get("query_temporal_pass") else "FAIL"}; diagnostic query, not archived detector verdict.',
               f'Lag={steps["parameters"]["lag_lo"]}..{steps["parameters"]["lag_hi"]}; yellow={steps.get("states",{}).get("yellow_score")}; morphology label={steps.get("candidate_mask",{}).get("component_label")}']
        _plot_svg(path,times,signal,None,None,{**item,**means,'stage_table':table,'stage_lines':extra,'mask_label':mask_label,'mask_intervals':info['intervals'],'fit_scope':'effective unmasked samples'})
    (out / 'stage_details.json').write_text(json.dumps({'pixel':pixel_steps,'region':region_steps}, indent=2), encoding='utf-8')
    image = cv.imread(str(folder / 'reference_gray.jpg'))
    if image is not None:
        cv.circle(image, (col, row), 8, (0, 0, 255), 2)
        cv.putText(image, f'row={row} col={col}', (max(0, col - 60), max(20, row - 12)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv.imwrite(str(out / 'position.jpg'), image)
    (out / 'position.json').write_text(json.dumps({'row': row, 'col': col,
        'index_base': 0, 'samples': len(times), 'source': str(folder.resolve()),
        'region_pixels': int(valid[y0:y1, x0:x1].sum()),
        'state_means': {'pixel': pixel_means, 'region': regional_means},
        'mref': mref,
        'mref_scope': 'one global reference for the whole video; not a pixel background or ON threshold',
        'mean_scope': 'full sampled sequence; same reference lines in all four panels'
        }, indent=2), encoding='utf-8')
    if compare_actual:
        from representative_plot import add_actual_plot
        add_actual_plot(folder,out,row,col,meta)
        comparison=json.loads((out/'representative_comparison.json').read_text(encoding='utf-8'))
        if comparison.get('track_available'):
            bx,by=comparison['tracked_coordinate']
            export_position(folder,by,bx,out/'B_full',compare_actual=False)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True, help='Video output gray_matrix folder')
    parser.add_argument('--row', required=True, type=int)
    parser.add_argument('--col', required=True, type=int)
    parser.add_argument('--out')
    args = parser.parse_args()
    try:
        print(export_position(args.matrix, args.row, args.col, args.out))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
