# v9c vision runtime

Current business policy (2026-11-02): 15 valid complete cycles immediately pass
one lamp. A 300-second valid duration is the minimum for zero-flash evidence and
scene discovery, not a prerequisite for normal. Unpassed counters continue past
300 seconds, up to the 1500-second deadline. Vision lifetime counters also keep
sampling so their identity evidence does not freeze at five minutes. Old
300-seconds-AND-15 descriptions below document earlier validation stages.

Per-window accepted positives retain only their own admitted evidence. Ordinary
photometric recalibration does not erase an accepted physical lamp; a new source,
unsafe identity merge, or invalidated geometry still resets it. Restart does not
restore active windows. `pulse_first_window_v4` retires prior-policy failure
history. Quality/identity/geometry checks and six-valid-zero-window failure remain.

This package contains the vision code extracted from the reviewed offline v9c
pipeline. Imports are package relative; no annotation coordinates, training
directory, review IDs, model downloading or offline report runner are required.
The versioned filenames describe implementation layers, not selectable backends.
`LampPipelineV9` is the only production pipeline.

`EXTRACTION.json` records source and extracted hashes. Crop input size/coarse
confidence are configurable (defaults 640/.35). Subsequent behavior fixes are
described below. `deploy/tests/test_adaptive_runtime.py` compares
default extracted outputs with the offline oracle when that local oracle exists.

The pipeline's lifetime `pulses`/`valid_s` fields are identity diagnostics only.
`night_lamp.runtime.AdaptiveLampController` samples audited per-frame signals
into separate business-window counters, handles shared reference permission,
within-view persistent IDs, evidence resets, persistence and archive reports.
The pipeline's private frozen image supports subpixel sampling/jitter only; it
is not another shared reference library and cannot commit camera movement.

The performance pass reuses stateless same-frame pixel metrics without sharing
continuity history, vectorizes identical background-mask bounds, memoizes pure
geometry in a bounded cache, and omits centers provably outside the influence
of the square sampler. Mask pixels, thresholds and model requests are unchanged.
The original and optimized pipelines/business evidence matched exactly on 934
real source samples and 448 cached model requests. Cached model outputs are an
offline test oracle only; production still calls the local YOLO backend.

The 2026-10-31 night-video review found a passing truck could invalidate a single
background alignment and erase every lamp's evidence. Invalid alignment now
pauses discovery/sampling and retains the frozen reference, identities and prior
evidence. Recovery must verify that reference; it never credits the blocked
interval or completes a pulse across it. A valid transform above the movement
threshold still resets local geometry and waits for the shared reference owner.
Support-mask changes, emitter corrections and ambiguous identities retain their
existing evidence-reset rules. This is a behavior fix, beyond the earlier
performance-only parity result.

The same-day old-video review found a one-hit broad body variant could quarantine
an already optically proved small head indefinitely. Such a pair can now be
reverified when the child has at least five detector hits and the tracker has a
prior optical fixture proof in this view. All existing optical requirements
remain: stable local background, continuous valid samples, fresh family detector
updates, a single child/core, repeated cycles, amplitude and neighbor veto.
The auditor independently remembers successful prior optical resolutions; a
detector flag alone cannot release quarantine. Unproved pairs still require
five parent hits. A successful resolution resets identity and business-window
evidence before observation resumes; it never sums ambiguous histories. Track
expiry removes audit provenance. Camera movement still invalidates local IDs.

The per-lamp follow-up also found coarse hypotheses below .65 could enter the
base quarantine while the optical resolver refused to examine them. V9 now uses
the same .65 lamp-type entry requirement in both quarantine paths; older audit
classes keep their original geometry predicate. A new sparse-body path accepts
an independently stored, previously admitted nighttime emitter epoch as prior
evidence. Registration requires a locked core, stable background, at least five
YOLO hits, fifteen valid pulses and current emission support; a daytime or
quarantined confirmation cannot register it. Recenter/support changes and track
expiry invalidate this record. A sparse body still needs a new uninterrupted
single-core optical proof and three detector updates inside that proof window.
The auditor checks its own prior epoch before release, and merging resets all
observation evidence. No reference coordinates, training labels or review IDs
are runtime inputs to this path.

The sampling-region follow-up keeps a recently flashing confirmed sampler when
the proposed optical core already has a full 3x3 neighborhood in its trusted
mask. A proposed core inside this sampler but excluded by that mask is rejected
instead of moving the sampler into excluded pixels. The mask is never reopened.
This requires a currently observable position, at least fifteen complete pulses,
calibrated amplitude and a complete flash within the existing 3-second maximum
pulse-width bound. Cold, stale, unobservable or outside-region cases still use
the existing correction/reset path. The original locator is retained: a separate
candidate-lifetime cleanup was tested and withdrawn after control regressions.

`exposure_recovery.RecoverableFlashSignal` can rebuild a stale local photometric
baseline after a persistent background-level change. Permission requires night,
clear/fresh/continuous registered frames, a locked emitter, at least five YOLO
hits, confidence at least .65, a recent body detection and no prior audit
quarantine. Local position/background gates still apply. The new evidence must
span at least 3 seconds and 12 samples, with gaps at most .5 seconds, at least one
new detector hit in that interval and background percentile ranges at most 3
gray levels (one quarter of the existing exposure-change limit). Detection age
is limited to the coarse-search period plus the continuity limit (5.5 seconds
with defaults). Waiting for this proof supplies no observation credit.

A `signal_reference_changed` event erases old templates, thresholds, pulses and
current valid time; it increments the signal/support and observation epochs,
invalidates audit provenance and restarts calibration and emitter localization.
The trigger frame is unusable. The controller resets both pending and persistent
business-window counters, retaining the within-view identity and location. The
shared camera reference library is unaffected. Repeated baseline changes may
therefore extend or prevent window completion; longer real videos must validate
these recovery parameters. Neither this recovery nor an identity green box is
proof of a complete 300-effective-second healthy window.

Compact-core ownership can also reverify a previously established fixture when
diffuse halo/background variation makes the original total-variation ownership
fraction fall below .70. The original .70 path is unchanged. The alternative
still requires the original unique component, continuous sampling, at least two
complete optical cycles, amplitude, new detector updates and neighbor veto. At
least .90 of the strong component's variation must lie inside the body; its peak
must be strictly inside that body. Strong pixels outside the body must all
belong to that same component and correlate with the body series by at least
.90. A separate strong pixel outside the body vetoes this path, even if too small
to qualify as a second full component. With no strong outside pixels, no halo
correlation is required. Cold candidates cannot use this alternative.

`ownership_prior` records either an earlier successful optical fixture or an
admitted emitter epoch. The auditor checks its own stored provenance again and
verifies the compact-core metrics; a detector flag alone cannot release it.
Successful reconciliation retains the existing identity and starts a fresh
sampler/calibration and zeroed business evidence. Multiple optical cores remain
unresolved. The .90 concentration/correlation parameters require longer-video
validation and do not change the 300-second/15-pulse business requirements.
## Search planning boundary

`search_plan.py` is a pure ROI planner. It accepts image proposals, optional
history/verification suggestions, timestamps, immutable prior-search metadata,
survey position and budgets. It does not access tracks, flash counters or health
state. The caller owns discovery timing and applies the returned search cursor.

The default `legacy` policy preserves existing request ordering. The word
`legacy` here names only the prior YOLO ROI ordering, not the retired non-YOLO
light backend (which has not been restored). `bounded` is an offline experimental
policy which reserves independent survey slots before
ranking suggestions; it is not enabled by the production configuration. Both
policies respect the existing fine-search cap. Quality gates, flash thresholds,
shared reference handling and water-barrier algorithms are unchanged.

Offline `training/lamps/detection_tape.py` records and verifies the complete
detector request (frame hash/shape/dtype, ordered ROIs, weight hash, input size,
confidence) and its raw detections. Replay rejects mismatches and extra/missing
requests; it does not silently return cached detections for a different crop.

## Bounded observation and optical source separation (2026-11-02)

Low-texture immediate backgrounds may be verified with frozen 64/96-pixel-radius
alternatives captured when the anchor was created. Local residual checks still
reject nearby changes (median absolute difference >18, or >25% of pixels differ
by >35). At least three stable textured tiles are still required. A textured
disagreement stops expansion; widened context additionally requires a recent
body detection. Registration, stream continuity and frame-quality gates remain.

`source_separation.py` distinguishes a spurious wide group box from independent
flashing sources. Each child needs prior flashes, stable detection and a trusted
sampler. Both child crops and the shared parent region must provide fresh optical
proof (16 samples, at least 3 seconds, at least two complete cycles); the shared
region must have separately owned connected cores. The two checks run concurrently
in the same usable window. Children retain separate IDs; the group is not aliased
to either child. Newly separated source counters and business epochs restart.

An optically verified repeated box variant may retain an unchanged sampler only
with the consumer's own earlier admitted epoch and exactly identical sample box.
Its signal calibration/mask are preserved, without adding any alias's evidence.
Known business counters pause during ambiguity and do not count that interval;
changed regions, new independent sources and unknown candidates still reset.
The evidence-policy marker retires old-policy failure history while retaining IDs
and diagnostic history. The 300-effective-second/15-cycle and failure-window rules
are unchanged. Some ambiguous detections may remain unobservable.

## Single-source revalidation (2026-11-02)

`single_source.py` treats highly overlapping, slightly displaced detector boxes
as a hypothesis to verify, rather than two independent fixtures. A localized
single temporal core must lie inside both bodies, with fresh detector updates,
at least 16 valid samples, three effective seconds and two complete cycles.
The consumer checks prior identities/epochs and geometry independently before
releasing quarantine. A second strong core vetoes a merge, including synchronized
adjacent lights. Proven detector variants are routed back to the same fixture.

A quarantined fixture whose competitor disappeared may be reobserved only with
prior physical-lamp evidence and a fresh single-core proof; disappearance alone
does not release it. Proof uses at most 64 samples within 12 source seconds.
Invalid intervals split segments: neither elapsed time nor a flash cycle crosses
a gap. This does not alter the business 300-second/15-cycle criteria.

An unchanged, independently admitted sampler can retain only its own evidence.
Merging different persistent business IDs resets the resulting business window;
unknown or changed samplers start a new epoch. No alias counts/time are added.
The visual policy is `single_source_revalidation_v3`; older-policy failure
history remains diagnostic and is not reused after upgrade.

