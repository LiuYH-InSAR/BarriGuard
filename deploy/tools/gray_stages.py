"""Full queried Gray diagnostics; distinct from archived representative verdicts."""
import json
from pathlib import Path
import numpy as np
import yaml
from night_lamp import evidence as ev
from night_lamp.periodicity import ac_limited, clean_train, onsets_of


def stage_details(folder, meta, raw, corrected, signal, mref, row, col, bgr=None, discovery=None):
    from night_lamp.tools import nightly_map as nm
    folder = Path(folder)
    defaults = dict(on_delta=nm.ON_DELTA, duty_floor=nm.CAND_DUTY_LO,
        baseline_frames=60, gap_min=ev.MIN_STATE_GAP, weight_min=ev.MIN_STATE_WEIGHT,
        separation_min=ev.MIN_SEPARATION, bic_min=ev.MIN_BIC_GAIN,
        onset_min=20, peak_floor=.2, min_peaks=3, gap_tol=1, lag_lo=2, lag_hi=44)
    camera_id = {'1002490':'1749', '1002491':'1750'}.get(folder.parent.name.split('_')[0])
    config_path = Path(__file__).resolve().parents[1] / 'config.yaml'
    if camera_id and config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        camera = next(c for c in config['cameras'] if str(c['id']) == camera_id)
        spec = camera['algorithms']['night_lamp']
        period = spec['periodicity']; rate = 1000 / spec['sampling']['interval_ms']
        defaults.update({k:period[k] for k in ('onset_min','peak_floor','min_peaks','gap_tol')})
        defaults['baseline_frames'] = spec['baseline']['frames']
        defaults['lag_lo'] = max(1, int(round(period['lag_lo_s'] * rate)))
        defaults['lag_hi'] = max(defaults['lag_lo']+1, int(round(period['lag_hi_s'] * rate)))
    saved = meta.get('parameters')
    p = {**defaults, **(saved or {})}
    from night_lamp.disturbance import detect, periodic_features
    from types import SimpleNamespace
    times=np.load(folder/'time_s.npy') if (folder/'time_s.npy').exists() else np.arange(len(signal))*.16
    enabled = p.get('disturbance', {'enabled':False})
    from night_lamp.stutter import detect_path,select_parameters
    branch_path=folder/'stutter_map.npy'
    force=bool(np.load(branch_path,mmap_mode='r')[row-meta['origin_row'],col-meta['origin_col']]) if branch_path.exists() else None
    disturbance_mask,window_mean,disturbance=detect_path(signal,times,enabled,p.get('stutter'),force)
    if discovery is not None and discovery.get('_mask') is not None:
        from night_lamp.disturbance import describe_mask
        disturbance_mask|=discovery['_mask'];disturbance.update(describe_mask(disturbance_mask,times))
    clean=~disturbance_mask
    known = set(saved or {})
    report_path = folder.parent / 'final.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8')).get('metadata', {})
        p['on_delta'] = report.get('delta', p['on_delta'])
        if 'delta' in report:
            known.add('on_delta')
        p['baseline_frames'] = report.get('warmup', {}).get('frames', p['baseline_frames'])
        if 'lag_frames' in report:
            p['lag_lo'], p['lag_hi'] = report['lag_frames']
            known.update(['lag_lo','lag_hi'])
    p=select_parameters(p,disturbance['suspected_stutter'])
    def verdict(passed, *keys):
        return ('PASS' if passed else 'FAIL') if all(k in known for k in keys) else 'UNKNOWN (historical threshold not saved; displayed value is current fallback)'
    lines = ['Scope: full queried sequence; detector representative evidence is recorded separately below.',
             'Thresholds: saved run settings.' if saved else 'Thresholds: current defaults where historical settings were not saved.']
    base = float(np.median(corrected[:p['baseline_frames']]))
    if (folder/'initial_background.npy').exists():
        background_map=np.load(folder/'initial_background.npy',mmap_mode='r')
        base=float(background_map[row-meta['origin_row'],col-meta['origin_col']])
    result = {'parameters':p, 'parameter_scope':'saved' if saved else 'current_defaults_with_available_report_fields',
              'pixel_raw_mean':float(raw.mean()), 'pixel_corrected_mean':float(corrected.mean()),
              'initial_pixel_background':base}
    result['disturbance']=disturbance
    from night_lamp.stutter import mask_report
    lines.extend(mask_report(disturbance,disturbance_mask,times,folder,row,col))
    lines.append(f'Path: {"suspected stutter" if disturbance["suspected_stutter"] else "normal"}; normal-mask duration ratio={disturbance["normal_mask_duration_ratio"]:.2%}; fallback settings={p.get("stutter",{"enabled":False})}')
    lines.append(f'Disturbance: enabled={disturbance["parameters"]["enabled"]}; window={disturbance["parameters"]["window_s"]}s; abs(window mean-reference)>{disturbance["parameters"]["enter_delta"]}; masked={disturbance["masked_samples"]}; effective={disturbance["valid_samples"]}; reference={disturbance.get("reference_mean")}')
    if report_path.exists():
        report_all = json.loads(report_path.read_text(encoding='utf-8'))
        matches = [e for e in report_all.get('metadata',{}).get('lamp_evidence',[]) if e.get('x')==col and e.get('y')==row]
        result['detector_representatives_at_coordinate'] = matches
        lines.append('Detector representative at exact coordinate: '+ ('saved verdict='+str(matches[0].get('flashing'))+'; reason='+str(matches[0].get('reason')) if matches else 'none; query is not an actual tracked representative'))
    ref = 'not saved' if mref is None else f'{mref:.2f}'
    lines.append(f'Step 1: Mref={ref}; pixel mean raw={raw.mean():.2f}, corrected={corrected.mean():.2f}; initial background median={base:.2f}')
    updates_path = folder/'background_updates.npy'
    if updates_path.exists():
        updates = np.load(updates_path, mmap_mode='r')
        result['background'] = {'resets':int(updates[:,0].sum()), 'update_records':len(updates)}
        scope='Masked background replay' if p.get('disturbance_before_background') else 'Online background'
        lines.append(f'  {scope}: saved initial map + {len(updates)} update records; resets={int(updates[:,0].sum())}; EMA half-life={p.get("ema_half_life_s","UNKNOWN")} s (update setting, not lamp gate)')
        if updates.shape[1]>=3:
            changes = np.abs(np.diff(updates[:,2]))
            limit = p.get('change_point_delta')
            result['background']['global_median_changes'] = changes.tolist()
            if limit is not None:
                enabled = p.get('change_point_reset_enabled', 'UNKNOWN (not saved)')
                lines.append(f'  Background jump reset enabled={enabled}; abs(frame median change)>{limit}: {int((changes>limit).sum())} samples; actual resets={int(updates[:,0].sum())}; reset_after_hit in CSV is actual action, not lamp rejection')
    count_path = folder.parent / 'on_count.npy'
    stats_path = folder.parent / 'duty_stats.json'
    duty_path = folder.parent / 'duty.npy'
    if discovery is not None:
        discovery=dict(discovery)
        discovery.pop('_mask',None)
        hits=discovery.pop('_hits',None)
        if hits is not None:
            discovery.update(exceed_count=int(np.asarray(hits)[clean].sum()),total=int(clean.sum()),ratio=float(np.asarray(hits)[clean].mean()) if clean.any() else 0.,source='saved background hits, query-specific disturbance excluded')
        result['discovery'] = discovery
        ratio_pass=discovery['ratio']>p['duty_floor'] if disturbance['suspected_stutter'] else discovery['ratio']>=p['duty_floor']
        result['fallback_ratio_pass']=bool(ratio_pass)
        lines.append(f'Step 2: exceed count={discovery["exceed_count"]}; total={discovery["total"]}; ratio={discovery["ratio"]:.4%}; ON_DELTA>={p["on_delta"]}; duty {">" if disturbance["suspected_stutter"] else ">="}{p["duty_floor"]:.2%}; {verdict(ratio_pass, "duty_floor")} (before morphology)')
    elif (count_path.exists() and stats_path.exists()) or duty_path.exists():
        total = len(raw)
        if stats_path.exists():
            total = int(json.loads(stats_path.read_text())['sample_total'])
        if (folder.parent/'valid_sample_count.npy').exists():
            totals=np.load(folder.parent/'valid_sample_count.npy',mmap_mode='r')
            total=int(totals[row,col])
        if count_path.exists():
            counts = np.load(count_path, mmap_mode='r')
            count = int(round(float(counts[row,col])))
        else:
            duty = np.load(duty_path, mmap_mode='r')
            count = int(round(float(duty[row,col]) * total))
        ratio = count / total if total else 0.
        result['discovery'] = dict(exceed_count=count, total=total, ratio=ratio)
        lines.append(f'Step 2: exceed count={count}; total={total}; ratio={ratio:.4%}; ON_DELTA>={p["on_delta"]}; duty>={p["duty_floor"]:.2%}; {verdict(ratio>=p["duty_floor"], "duty_floor")} (before morphology)')
    else:
        result['discovery'] = None
        lines.append(f'Step 2: historical exceed count/ratio unavailable (online background/count not saved); total queried samples={len(raw)}; ON_DELTA>={p["on_delta"]}; duty>={p["duty_floor"]:.2%}')
    labels_path = folder.parent / 'duty_candidate_labels.npy'
    if labels_path.exists():
        labels = np.load(labels_path, mmap_mode='r')
        label = int(labels[row,col])
        result['candidate_mask'] = {'component_label':label, 'after_morphology_pass':bool(label>0)}
        lines.append(f'  Final duty component after 3x3 opening: label={label}; '+('PASS' if label>0 else 'FAIL')+'; online discovery history remains distinct')
    minimum = p.get('min_samples', ev.MIN_SAMPLES)
    if clean.sum() < minimum or not np.isfinite(signal).all():
        lines.append(f'Steps 3/4: effective samples={int(clean.sum())}; minimum={minimum}; FAIL (insufficient or missing samples)')
        return result, lines
    means, variances, weights, bic, on_clean = ev._fit_states(signal[clean])
    on=np.zeros(len(signal),bool);on[clean]=on_clean
    gap = float(means[1]-means[0]); sep = gap / float(np.sqrt(variances.mean()))
    yellow = None
    separation_limit = p['separation_min']
    if bgr is not None:
        delta = bgr[clean&on].mean(axis=0)-bgr[clean&~on].mean(axis=0) if on[clean].any() and (~on[clean]).any() else np.zeros(3)
        blue,green,red = delta
        rg = min(red,green)
        colour_min = p.get('min_colour_gain',ev.MIN_COLOUR_GAIN)
        yellow = float(np.clip((rg-max(blue,0))/max(rg,colour_min),0,1)) if rg>=colour_min else 0.
        support = p.get('yellow_support',ev.YELLOW_SUPPORT)
        if yellow>=support:
            separation_limit=p.get('yellow_separation',ev.YELLOW_SEPARATION)
        lines.append(f'Yellow auxiliary: min(R gain,G gain)={rg:.4f}>={colour_min}: {"PASS" if rg>=colour_min else "FAIL"}; yellow score={yellow:.4f}>={support}: {"PASS" if yellow>=support else "FAIL"}; selected separation threshold={separation_limit}; auxiliary FAIL does not veto')
    result['states'] = dict(off_mean=float(means[0]), on_mean=float(means[1]), gap=gap,
        weights=weights.tolist(), separation=sep, bic_gain=bic)
    result['disturbance'].update(off_mean=float(means[0]),c_reference=disturbance['reference_mean']-float(means[0]))
    result['states'].update(yellow_score=yellow, separation_threshold=separation_limit)
    lines.append(f'Step 3: effective samples={int(clean.sum())}>={minimum}; {verdict(clean.sum()>=minimum, "min_samples")}; '+('saved BGR support evaluated' if bgr is not None else 'yellow RGB support unavailable; colour-adjusted verdict UNKNOWN'))
    weight_label = f'min state weight (OFF={weights[0]:.4%}, ON={weights[1]:.4%})'
    checks = {}
    for name, value, limit, key in [('mean gap',gap,p['gap_min'],'gap_min'),(weight_label,float(weights.min()),p['weight_min'],'weight_min'),('separation',sep,separation_limit,'separation_min'),('BIC gain',bic,p['bic_min'],'bic_min')]:
        status = verdict(value>=limit,key)
        checks[name] = dict(value=value, threshold=limit, passed=bool(value>=limit) if key in known else None, status=status)
        lines.append(f'  {name}={value:.4f}; threshold>={limit}; {status}')
    result['states']['checks'] = checks
    onsets,peak,lag,curve,periodic,peaks=periodic_features(on,clean,p['lag_lo'],p['lag_hi'],SimpleNamespace(**{k:p[k] for k in ('peak_floor','min_peaks','gap_tol')}))
    result['periodicity'] = dict(onsets=onsets, ac_peak=peak, ac_lag=lag,
                                peak_lags=peaks, periodic=bool(periodic), ac_curve=curve.tolist())
    spreads = [int(np.ptp(np.diff(peaks[i:i+p['min_peaks']]))) for i in range(len(peaks)-p['min_peaks']+1)]
    best_spread = min(spreads) if spreads else None
    peak_values = [float(curve[v-p['lag_lo']]) for v in peaks]
    result['periodicity'].update(peak_values=peak_values, best_gap_spread=best_spread,
        onset_pass=bool(onsets>=p['onset_min']), peak_count_pass=bool(len(peaks)>=p['min_peaks']), gap_spread_pass=bool(best_spread is not None and best_spread<=p['gap_tol']))
    lines.append(f'Step 4: OFF->ON count={onsets}; threshold>={p["onset_min"]}; {verdict(onsets>=p["onset_min"], "onset_min")}; leading ON counts once')
    lines.append(f'  AC max={peak:.4f}; lag range={p["lag_lo"]}..{p["lag_hi"]} samples; local peak floor>={p["peak_floor"]}')
    lines.append(f'  peak lags={peaks}; consecutive peaks>={p["min_peaks"]}; max gap spread<={p["gap_tol"]} sample; {verdict(periodic,"min_peaks","gap_tol","peak_floor","lag_lo","lag_hi")}')
    lines.append(f'  qualifying local peaks={len(peaks)}>={p["min_peaks"]}: {verdict(len(peaks)>=p["min_peaks"],"min_peaks")}; best consecutive gap spread={best_spread}<={p["gap_tol"]}: {verdict(periodic,"gap_tol","min_peaks")}')
    lines.append(f'  local peak values={np.round(peak_values,4).tolist()}; floor>={p["peak_floor"]}: {verdict(bool(peak_values),"peak_floor")}; AC max has no independent threshold')
    result['query_temporal_pass'] = bool(all(c['passed'] for c in checks.values()) and onsets>=p['onset_min'] and periodic and (not disturbance['suspected_stutter'] or result.get('fallback_ratio_pass',False))) if saved else None
    lines.append('Query temporal gates: '+ ('UNKNOWN (historical run settings missing)' if result['query_temporal_pass'] is None else 'PASS' if result['query_temporal_pass'] else 'FAIL')+'; candidate discovery/morphology/tracking is separate')
    return result, lines
