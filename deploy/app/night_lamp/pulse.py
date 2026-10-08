"""Detect returned short pulses, then verify sustained repetition and periodicity."""
import numpy as np

from .periodicity import ac_limited, clean_train


def analyse_pulses(values, rate, spec, period):
    """Return pulse evidence and sample mask; use pre-rise level, never night base."""
    x = np.asarray(values, np.float64)
    mask = np.zeros(x.size, np.float32)
    result = {"flashing": False, "reason": "too_few_pulses", "samples": int(x.size),
              "pulses": [], "pulse_count": 0, "sustained_duration_s": 0.0}
    if not x.size or not np.isfinite(x).all():
        result["reason"] = "missing_samples"
        return result, mask
    rise_n = max(1, int(np.floor(spec.rise_max_s * rate)))
    width_n = max(1, int(np.floor(spec.width_max_s * rate)))
    tolerance = spec.amplitude_min * spec.return_ratio
    pulses = []
    i = 1
    while i < len(x):
        lo = max(0, i - rise_n)
        start = lo + int(np.argmin(x[lo:i]))
        level = x[start]
        if x[i] - level < spec.amplitude_min:
            i += 1
            continue
        # Last near-normal sample before the rise gives the actual pulse width.
        start = i - 1
        while start > lo and x[start] > level + tolerance:
            start -= 1
        peak = i
        end = None
        for j in range(i + 1, min(len(x), start + width_n + 1)):
            if x[j] > x[peak]:
                peak = j
            if x[j] <= level + tolerance:
                end = j
                break
        if end is None or peak - start > rise_n:
            i += 1
            continue
        pulse = {"start_index": int(start), "peak_index": int(peak),
                 "end_index": int(end), "level": round(float(level), 3),
                 "amplitude": round(float(x[peak] - level), 3),
                 "width_s": round((end - start) / rate, 4)}
        pulses.append(pulse)
        mask[start + 1:end] = 1
        i = end + 1
    result.update(pulses=pulses, pulse_count=len(pulses), duty=round(float(mask.mean()), 4))
    if len(pulses) < period.onset_min:
        return result, mask
    # A contiguous pulse run must last long enough; isolated bursts cannot add up.
    groups = []
    run = []
    for pulse in pulses:
        if run and (pulse["peak_index"] - run[-1]["peak_index"]) / rate > spec.max_gap_s:
            groups.append(run)
            run = []
        run.append(pulse)
    groups.append(run)
    longest = max(groups, key=lambda group: group[-1]["end_index"] - group[0]["start_index"])
    span = (longest[-1]["end_index"] - longest[0]["start_index"]) / rate
    result.update(sustained_duration_s=round(span, 4), sustained_pulse_count=len(longest))
    qualified = [group for group in groups if len(group) >= period.onset_min
                 and (group[-1]["end_index"] - group[0]["start_index"]) / rate >= spec.min_duration_s]
    if not qualified:
        result["reason"] = "not_sustained"
        return result, mask
    longest = max(qualified, key=lambda group: group[-1]["end_index"] - group[0]["start_index"])
    result.update(sustained_duration_s=round((longest[-1]["end_index"] - longest[0]["start_index"]) / rate, 4),
                  sustained_pulse_count=len(longest))
    lag_lo = max(1, int(round(period.lag_lo_s * rate)))
    lag_hi = max(lag_lo + 1, int(round(period.lag_hi_s * rate)))
    # Preserve the original autocorrelation peak-spacing check, now on pulse mask.
    segment = mask[longest[0]["start_index"]:longest[-1]["end_index"] + 1]
    peak, lag, curve = ac_limited(segment, lag_lo, lag_hi)
    periodic, peaks = clean_train(curve, lag_lo, period.peak_floor,
                                 period.min_peaks, period.gap_tol)
    result.update(flashing=bool(periodic), ac_peak=round(float(peak), 4),
                  ac_lag=int(lag), peak_lags=[int(v) for v in peaks],
                  reason="repeated_pulses" if periodic else "not_periodic")
    return result, mask
