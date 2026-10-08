"""Stack the queried point with the actual archived detector track, without replay."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from night_lamp import evidence as ev
from night_lamp.disturbance import detect, describe_mask, periodic_features
from night_intermediates import _plot_svg


def stack_svg(top, bottom, target):
    a=ET.parse(top).getroot();b=ET.parse(bottom).getroot()
    width=max(float(a.attrib['width']),float(b.attrib['width']))
    height=float(a.attrib['height']);b.set('y',str(height))
    root=ET.Element('{http://www.w3.org/2000/svg}svg',width=str(width),
                    height=str(height+float(b.attrib['height'])))
    root.extend((a,b));ET.ElementTree(root).write(target,encoding='utf-8',xml_declaration=True)


def add_actual_plot(folder, out, row, col, meta):
    report=folder.parent/'final.json';labels_path=folder.parent/'duty_candidate_labels.npy'
    evidence=None
    if report.exists():
        records=json.loads(report.read_text(encoding='utf-8')).get('metadata',{}).get('lamp_evidence',[])
        if labels_path.exists():
            labels=np.load(labels_path,mmap_mode='r');label=int(labels[row,col])
            evidence=next((e for e in records if label>0 and e.get('kind')=='candidate'
                           and int(labels[e['y'],e['x']])==label),None)
        if evidence is None:
            evidence=next((e for e in records if (e.get('x'),e.get('y'))==(col,row)),None)
    track=evidence.get('series_row') if evidence else None
    archive=folder.parent/'intermediate/series.npz'
    mapping=dict(query=[col,row],representative=evidence,track_available=False)
    if track is None or not archive.exists():
        explanation='No detector representative for this component.' if evidence is None else 'Detector representative has no recorded series.'
        for kind in ('pixel','region'):
            path=out/f'{kind}_gray.svg';tree=ET.parse(path);root=tree.getroot()
            height=float(root.attrib['height']);root.set('height',str(height+90))
            view=root.attrib.get('viewBox','').split()
            if view:view[3]=str(height+90);root.set('viewBox',' '.join(view))
            text=ET.SubElement(root,'{http://www.w3.org/2000/svg}text',x='25',y=str(height+35),fill='black')
            text.text='B: '+explanation+' No nearby substitute was fabricated.'
            tree.write(path,encoding='utf-8',xml_declaration=True)
        mapping['explanation']=explanation
        (out/'representative_comparison.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
        return
    p=meta['parameters'];period=SimpleNamespace(**{k:p[k] for k in ('onset_min','peak_floor','min_peaks','gap_tol')})
    with np.load(archive) as saved:
        start=int(saved['tracking_start_index'][track]);x,y=map(int,saved['xy'][track])
        times=saved['time_s'][start:].copy();pixel=saved['pixel_gray_corrected'][track,start:].copy()
        bgr=saved['bgr_corrected'][track,start:].copy();region=bgr@ev.GRAY_BGR
    packed_path=folder/'disturbance_mask_packed.npy'
    external=np.zeros(len(times),bool)
    if packed_path.exists():
        packed=np.load(packed_path,mmap_mode='r')
        external=np.unpackbits(packed[y-meta['origin_row'],x-meta['origin_col']],bitorder='little')[start:start+len(times)].astype(bool)
    mapping.update(track_available=True,tracked_coordinate=[x,y],tracking_start_index=start,
                   tracking_start_s=float(times[0]),scope='actual recorded suffix; archived verdict applies to regional BGR')
    ratio_line='Step 2: historical B-coordinate counts unavailable; UNKNOWN.'
    if (folder.parent/'on_count.npy').exists() and (folder.parent/'valid_sample_count.npy').exists():
        count=float(np.load(folder.parent/'on_count.npy',mmap_mode='r')[y,x])
        total=int(np.load(folder.parent/'valid_sample_count.npy',mmap_mode='r')[y,x])
        ratio=count/total if total else 0.
        mapping['full_run_discovery']=dict(count=count,total=total,ratio=ratio)
        ratio_line=f'Step 2 B-coordinate full-run ratio: hits={count:g}; effective total={total}; ratio={ratio:.4%}>={p["duty_floor"]:.2%}: {"PASS" if ratio>=p["duty_floor"] else "FAIL"}; ON_DELTA>={p["on_delta"]}'
    arrays=dict(time_s=times,pixel_gray_corrected=pixel,region_gray_corrected=region,bgr_corrected=bgr)
    for kind,signal in [('pixel',pixel),('region',region)]:
        # The actual detector's mask comes from regional Gray plus its core mask.
        mask_signal=pixel if evidence.get('kind')=='single_pixel' else region
        mask,mean,info=detect(mask_signal,times,p.get('disturbance',{'enabled':False}))
        mask|=external;clean=~mask;info.update(describe_mask(mask,times))
        lines=[f'B actual track: col={x}, row={y}; representative col={evidence["x"]}, row={evidence["y"]}; series_row={track}',
               f'Scope: actual recorded sequence starts at {times[0]:.3f}s; no earlier samples inserted.',
               f'Archived detector verdict: {"PASS" if evidence["flashing"] else "FAIL"}; reason={evidence["reason"]}; samples={evidence.get("samples")}',
               f'Saved thresholds; window={info["parameters"]["window_s"]}s; enter>{info["parameters"]["enter_delta"]}; exit<={info["parameters"]["exit_delta"]}; reference={info.get("reference_mean"):.3f}',
               f'Masked={int(mask.sum())}; effective={int(clean.sum())}/{len(times)}; min samples>={p["min_samples"]}: {"PASS" if clean.sum()>=p["min_samples"] else "FAIL"}',
               f'Step 1: Mref={meta.get("mref")}; recorded Gray mean={float(np.mean(signal)):.3f}',
               ratio_line,
               'Archived lamp signal: '+('pixel Gray (single-pixel supplement).' if evidence.get('kind')=='single_pixel' else 'regional BGR; pixel fit is diagnostic.')]
        if evidence.get('kind')=='single_pixel':
            limit=evidence.get('min_effective_seconds')
            duration=evidence.get('effective_seconds')
            lines.append(f'Single-pixel effective duration={duration}s >= {limit}s: '+('PASS' if evidence.get('effective_duration_pass') else 'FAIL'))
            minimum=p.get('single_pixel',{}).get('min_hits')
            hits=evidence.get('discovery_hits')
            lines.append(f'Single-pixel discovery hits={hits} >= {minimum}: '+('PASS' if minimum is not None and hits is not None and hits>=minimum else 'UNKNOWN'))
        details=dict(mask=info,archived_verdict=evidence);means={}
        if clean.sum()>=p['min_samples'] and np.isfinite(signal).all():
            mu,var,weights,bic,on_clean=ev._fit_states(signal[clean]);gap=float(mu[1]-mu[0]);sep=gap/np.sqrt(var.mean())
            on=np.zeros(len(signal),bool);on[clean]=on_clean
            delta=bgr[clean&on].mean(axis=0)-bgr[clean&~on].mean(axis=0) if on_clean.any() and (~on_clean).any() else np.zeros(3)
            blue,green,red=delta;rg=min(red,green)
            yellow=float(np.clip((rg-max(blue,0))/max(rg,p['min_colour_gain']),0,1)) if rg>=p['min_colour_gain'] else 0.
            limit=p['yellow_separation'] if yellow>=p['yellow_support'] else p['separation_min']
            checks={}
            for name,value,threshold in [('mean gap',gap,p['gap_min']),('min state weight',float(weights.min()),p['weight_min']),('separation',float(sep),limit),('BIC gain',bic,p['bic_min'])]:
                checks[name]=dict(value=float(value),threshold=threshold,passed=bool(value>=threshold))
                lines.append(f'Step 3 {name}={value:.4f}; threshold>={threshold}: {"PASS" if value>=threshold else "FAIL"}')
            lines.append(f'Yellow score={yellow:.4f}>={p["yellow_support"]}: {"PASS" if yellow>=p["yellow_support"] else "FAIL (auxiliary only)"}; RG gain={rg:.4f}>={p["min_colour_gain"]}')
            onset,peak,lag,curve,passed,peaks=periodic_features(on,clean,p['lag_lo'],p['lag_hi'],period)
            spreads=[int(np.ptp(np.diff(peaks[j:j+p['min_peaks']]))) for j in range(len(peaks)-p['min_peaks']+1)]
            spread=min(spreads) if spreads else None
            lines.extend([f'Step 4 onsets={onset}>={p["onset_min"]}: {"PASS" if onset>=p["onset_min"] else "FAIL"}',
                f'AC max={peak:.4f}; lag={p["lag_lo"]}..{p["lag_hi"]}; peak floor>={p["peak_floor"]}; peaks={peaks}',
                f'Peak count={len(peaks)}>={p["min_peaks"]}: {"PASS" if len(peaks)>=p["min_peaks"] else "FAIL"}; best consecutive gap spread={spread}<={p["gap_tol"]}: {"PASS" if passed else "FAIL"}'])
            means=dict(off_mean=float(mu[0]),on_mean=float(mu[1]))
            details.update(checks=checks,means=means,onsets=onset,periodic=bool(passed),peak_lags=peaks)
        else:lines.append('Steps 3/4: FAIL or unavailable due to insufficient/nonfinite actual observations.')
        arrays[kind+'_mask']=mask;mapping[kind]=details
        b_path=out/f'{kind}_actual_B.svg'
        _plot_svg(b_path,times,signal,None,None,dict(reason=f'B actual track col={x}, row={y}',
                  stage_lines=lines,mref=meta.get('mref'),mask_intervals=info['intervals'],**means))
        a_path=out/f'{kind}_gray.svg';query_path=out/f'{kind}_query_A.svg'
        query_path.write_bytes(a_path.read_bytes());stack_svg(query_path,b_path,a_path)
    np.savez_compressed(out/'actual_B_series.npz',**arrays)
    (out/'representative_comparison.json').write_text(json.dumps(mapping,indent=2),encoding='utf-8')
