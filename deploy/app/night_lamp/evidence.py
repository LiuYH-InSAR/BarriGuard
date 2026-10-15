"""Regional two-state evidence with optional yellow support; original discovery."""
import numpy as np

from .periodicity import ac_limited, clean_train, onsets_of

REGION_RADIUS = 2
MIN_SAMPLES = 60
FIT_ITERATIONS = 40
VARIANCE_FLOOR = 1.0
MIN_STATE_WEIGHT = 0.05
MIN_STATE_GAP = 12.0
MIN_SEPARATION = 2.5
YELLOW_SEPARATION = 2.0
YELLOW_SUPPORT = 0.5
MIN_COLOUR_GAIN = 3.0
GRAY_BGR = np.array([0.114, 0.587, 0.299])
MIN_BIC_GAIN = 10.0


def _fit_states(values):
    """Compare a two-Gaussian fit with one Gaussian using BIC."""
    x = np.asarray(values, np.float64)
    means = np.percentile(x, [2, 98])
    variance = max(float(np.var(x)), VARIANCE_FLOOR)
    variances = np.full(2, variance)
    weights = np.full(2, 0.5)
    for _ in range(FIT_ITERATIONS):
        logs = (np.log(weights) - 0.5 * np.log(2 * np.pi * variances)
                - (x[:, None] - means) ** 2 / (2 * variances))
        offset = logs.max(axis=1)
        probs = np.exp(logs - offset[:, None])
        probs /= probs.sum(axis=1)[:, None]
        counts = np.maximum(probs.sum(axis=0), 1e-9)
        means = (probs * x[:, None]).sum(axis=0) / counts
        variances = np.maximum(
            (probs * (x[:, None] - means) ** 2).sum(axis=0) / counts,
            VARIANCE_FLOOR)
        weights = counts / x.size
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    logs = (np.log(weights) - 0.5 * np.log(2 * np.pi * variances)
            - (x[:, None] - means) ** 2 / (2 * variances))
    offset = logs.max(axis=1)
    ll_two = float(np.sum(offset + np.log(np.exp(logs - offset[:, None]).sum(axis=1))))
    ll_one = float(-0.5 * x.size * (np.log(2 * np.pi * variance) + 1))
    gain = 2 * (ll_two - ll_one) - 3 * np.log(x.size)
    on = logs[:, 1] > logs[:, 0]
    return means, variances, weights, float(gain), on


def analyse_region(bgr, period, lag_lo, lag_hi, times=None, external_mask=None, force_stutter=None):
    """Fit Gray states; yellow supports separation, never vetoes other lamps."""
    samples = np.asarray(bgr, np.float64).reshape(-1, 3)
    valid = np.all(np.isfinite(samples), axis=1)
    check_period = getattr(period, "enabled", True)
    result = {"samples": int(valid.sum()), "flashing": False,
              "periodicity_enabled": bool(check_period),
              "reason": "insufficient_samples", "yellow_score": 0.0}
    # Trim missing warmup prefix only; never compress internal gaps into time.
    from .stutter import SETTINGS, detect_path
    early_minimum=min(MIN_SAMPLES,SETTINGS['min_samples']) if SETTINGS['enabled'] else MIN_SAMPLES
    if valid.sum() < early_minimum:
        return result
    first = int(np.flatnonzero(valid)[0])
    samples = samples[first:]
    if not np.all(np.isfinite(samples)):
        result["reason"] = "sampling_gap"
        return result
    from .disturbance import detect, periodic_features
    t = np.arange(len(samples))*.16 if times is None else np.asarray(times)[first:]
    from types import SimpleNamespace
    disturbance_mask, _, disturbance = detect_path(samples @ GRAY_BGR,t,{},SETTINGS,force_stutter)
    fallback=disturbance['suspected_stutter']
    minimum=SETTINGS['min_samples'] if fallback else MIN_SAMPLES
    if fallback:
        period=SimpleNamespace(onset_min=SETTINGS['onset_min'],peak_floor=SETTINGS['peak_floor'],
                               min_peaks=SETTINGS['min_peaks'],gap_tol=SETTINGS['gap_tol'])
    result.update(suspected_stutter=fallback,selected_thresholds=dict(min_samples=minimum,
        onset_min=period.onset_min,peak_floor=period.peak_floor,min_peaks=period.min_peaks,gap_tol=period.gap_tol))
    if external_mask is not None:
        from .disturbance import describe_mask
        disturbance_mask|=np.asarray(external_mask,bool)[first:]
        disturbance.update(describe_mask(disturbance_mask,t))
    clean=~disturbance_mask
    result['disturbance']=disturbance
    result['samples_before_mask']=len(samples)
    result['samples']=int(clean.sum())
    if clean.sum()<minimum:return result
    means, variances, weights, bic, on_clean = _fit_states((samples @ GRAY_BGR)[clean])
    on=np.zeros(len(samples),bool);on[clean]=on_clean
    gap = float(means[1] - means[0])
    separation = gap / float(np.sqrt(np.mean(variances)))
    delta = np.zeros(3)
    if on[clean].any() and (~on[clean]).any():
        delta = samples[clean&on].mean(axis=0) - samples[clean&~on].mean(axis=0)
    blue, green, red = delta
    rg = min(red, green)
    yellow = (float(np.clip((rg - max(blue, 0)) / max(rg, MIN_COLOUR_GAIN), 0, 1))
              if rg >= MIN_COLOUR_GAIN else 0.0)
    sep_floor = YELLOW_SEPARATION if yellow >= YELLOW_SUPPORT else MIN_SEPARATION
    result.update(off_mean=round(float(means[0]), 3),
                  on_mean=round(float(means[1]), 3),
                  state_std=[round(float(v), 3) for v in np.sqrt(variances)],
                  state_weights=[round(float(v), 4) for v in weights],
                  amplitude=round(gap, 3), separation=round(separation, 3),
                  bic_gain=round(bic, 3), yellow_score=round(yellow, 3),
                  delta_rgb=[round(float(v), 3) for v in delta[::-1]],
                  duty=round(float(on[clean].mean()), 4))
    if check_period:
        onset,peak,lag,curve,periodic,peaks=periodic_features(on,clean,lag_lo,lag_hi,period)
    else:
        onset = int(np.count_nonzero(clean & on & np.r_[True, clean[:-1] & ~on[:-1]]))
        peak, lag, periodic, peaks = 0.0, -1, True, []
    result['onsets']=onset
    if (min(weights) < MIN_STATE_WEIGHT or gap < MIN_STATE_GAP
            or separation < sep_floor or bic < MIN_BIC_GAIN):
        result["reason"] = "states_not_separated"
        return result
    if result["onsets"] < period.onset_min:
        result["reason"] = "too_few_onsets"
        return result
    result.update(flashing=bool(periodic), ac_peak=round(float(peak), 4),
                  ac_lag=int(lag), peak_lags=[int(v) for v in peaks],
                  reason=("periodicity_disabled" if not check_period
                          else "periodic" if periodic else "not_periodic"))
    return result


def state_sequence(gray):
    """Posterior OFF/ON assignments for inspection, not a lamp verdict."""
    samples = np.asarray(gray, np.float64).reshape(-1)
    valid = np.isfinite(samples)
    states = np.full(len(samples), np.nan, np.float32)
    if valid.sum() < MIN_SAMPLES:
        return states
    first = int(np.flatnonzero(valid)[0])
    if not valid[first:].all():
        return states
    states[first:] = _fit_states(samples[first:])[4]
    return states


def state_means(gray):
    """Diagnostic fitted means; these alone do not establish a lamp verdict."""
    samples = np.asarray(gray, np.float64).reshape(-1)
    valid = np.flatnonzero(np.isfinite(samples))
    if len(valid) < MIN_SAMPLES:
        return {}
    samples = samples[valid[0]:]
    if not np.isfinite(samples).all():
        return {}
    means = _fit_states(samples)[0]
    return {"off_mean": float(means[0]), "on_mean": float(means[1])}
