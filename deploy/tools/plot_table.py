"""One shared diagnostic table schema for query A and recorded B."""
import numpy as np


def cell(text,passed=None):
    return dict(text=text+(' PASS' if passed else ' FAIL') if passed is not None else text,
                failed=passed is False)


def compare(name,value,limit,op='>=',percent=False):
    if value is None:return cell(f'{name}=N/A {op}{limit}; N/A')
    passed=value>limit if op=='>' else value<=limit if op=='<=' else value>=limit
    fmt=lambda v:f'{v:.2%}' if percent else f'{v:.3g}'
    return cell(f'{name}={fmt(value)} {op}{fmt(limit)}',bool(passed))


def report_table(p,info,states,periodicity,discovery,mref,mean):
    fallback=info.get('suspected_stutter',False)
    trigger=info.get('trigger_pixel_full_run_mask_ratio')
    threshold=info.get('fallback_settings',{}).get('mask_ratio')
    path_label='共享卡顿路径' if fallback and threshold is not None and trigger is not None and trigger<=threshold else '触发卡顿路径' if fallback else '不触发卡顿路径'
    trigger_cell=cell(f'Original mask={trigger:.2%} '+path_label) if trigger is not None else cell('Original mask=N/A')
    rows=[[cell('Mask'),trigger_cell,
           cell(f'Selected {info["parameters"]["window_s"]:g}s mask={info.get("mask_duration_ratio",0):.2%}'),
           cell(f'Final mask={info.get("final_mask_duration_ratio",0):.2%}')],
          [cell('Step 1'),cell(f'Mref={mref:.2f}' if mref is not None else 'Mref=N/A'),cell(f'Gray mean={mean:.2f}')],
          [cell('Step 2'),compare('Ratio',discovery.get('ratio'),p['duty_floor'],'>' if fallback else '>=',True),
           cell(f'Hits={discovery.get("count",discovery.get("exceed_count","N/A"))}; total={discovery.get("total","N/A")}'),cell(f'Gray-background >= {p["on_delta"]}')],
          [cell('Step 3'),compare('Samples',info.get('valid_samples'),p['min_samples']),
           compare('Mean gap',states.get('gap'),p['gap_min']),
           compare('Min weight',min(states['weights']) if states.get('weights') else None,p['weight_min'],percent=True),
           compare('Separation',states.get('separation'),states.get('separation_threshold',p['separation_min'])),
           compare('BIC gain',states.get('bic_gain'),p['bic_min'])]]
    peaks=periodicity.get('peak_lags',[]);curve=periodicity.get('ac_curve')
    local_peak=None
    if curve is not None:
        a=np.asarray(curve);values=[a[i] for i in range(1,len(a)-1) if a[i]>=a[i-1] and a[i]>=a[i+1]]
        local_peak=float(max(values)) if values else None
    rows.append([cell('Step 4'),compare('Onsets',periodicity.get('onsets'),p['onset_min']),
        compare('Local AC peak',local_peak,p['peak_floor']),
        compare('Peak count',len(peaks) if periodicity else None,p['min_peaks']),
        compare('Gap spread',periodicity.get('best_gap_spread'),p['gap_tol'],'<=')])
    return rows
