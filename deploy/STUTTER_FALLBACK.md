# High-mask fallback

Normal processing and autocorrelation formulas remain unchanged. The fallback is a parameter branch, not a verified freeze detector.

After exposure correction, compute the original 5s per-pixel mask. Mask duration uses each observation's interval up to the next observation, divided by the recorded sequence duration. Trigger strictly above 80%. Select once, recompute that pixel's mask with window_s=20; do not re-test the branch using the replacement mask. Replay original background updates and hits using the selected mask.

The recorded pixel's branch selects the 5s/20s regional mask window and downstream parameters. Regional masks still combine with the selected pixel mask, preserving the existing two-mask process. Selection uses the complete saved pixel sequence, not the shorter representative suffix. Keep original initial background and reference calculation, morphology, tracking, colour support and lag range.

Fallback: ratio > 0.20 at the tracked pixel over its full saved sequence; valid regional samples >=40; onsets >=5; AC local peak floor >=0.1; at least3 peaks; gap spread <=5 samples. No new independent 20-hit gate. All state thresholds remain unchanged. Single-pixel supplement retains its original 20-hit and 20-effective-second requirements.

Saved matrix outputs: stutter_map.npy (ROI-sized branch flags), normal_mask_duration_ratio.npy (original 5s mask duration fractions), selected disturbance_mask_packed.npy, masked statistics, and manifest parameters.stutter. Final evidence saves suspected_stutter, selected_thresholds, and fallback ratio value/threshold/pass. A/B plots use saved branch/settings; archived older runs do not enable this path retrospectively.

Migration reference: BarriGuard0 is read-only. Original-file backups and the migration manifest are in output/migration; no old project's rollback script is copied.
