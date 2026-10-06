# Schema audit — mouse arena visual discrimination

Source: `/Users/danieldenman/behavior_data` (bindfs read-only mirror of
`/Volumes/s1/behavior/mouse_arena_discrimination/data`). Audited 23 sessions across
4 animals. **No files in the source were modified; the mount is `-r`.**

## Repository layout

```
<root>/<animal>/<YYYY-MM-DD_HH-MM-SS>/
    events.csv              one row per event (trial, fall, timeout, session bounds)
    samples.csv             ~32 Hz behavioural trace
    session_meta.json       mouse id, weight, task config
    sync_events.csv         DAQ sync pulses (not used here)
    capture_manifest.json   video capture manifest (absent in 1 session)
    camera/ , mirror_capture/   video (not used here)
    flir_diagnostics.log
```

Column sets are **identical across all 23 sessions** — no schema drift.

## events.csv

| column | meaning |
|---|---|
| `t_start`, `t_end` | trial window, seconds from session start |
| `event_type` | `trial` (4197), `fall_trial` (528), `timeout_start` (115), `timeout_end` (114), `session_start`/`session_end` (23 each) |
| `trial_id` | index into `samples.trial_id` |
| `rewarded` | identical to `extra.outcome == "correct"` |
| `stim_id` | e.g. `target_60`; NaN on fall trials |
| `orientation` | identical to `extra.target_orientation` |
| `contrast` | 1.0 throughout |
| `x`, `y`, `heading` | pose at trial end |
| `extra` | JSON, the substantive trial record (below) |

### `extra` JSON
`outcome` (`correct` / `incorrect` / `fall` / `omission`), `chosen_box`, `correct_box`,
`target_box`, `distractor_box`, `chosen_side`, `correct_side`, `target_side`,
`distractor_side`, `target_orientation`, `distractor_orientation`, `choice_latency_s`.

Verified relationships (session jlh62/2026-10-01, all exact):
- `correct_box == target_box` on every trial — the rewarded stimulus is always the target.
- `rewarded == (outcome == "correct")`.
- `events.orientation == extra.target_orientation`.
- **`t_end - t_start == choice_latency_s + 1.000 s`** (sd 0.007), the configured `iti_s`.
  So `choice_latency_s` is the stimulus-presentation-to-choice/reward latency and is the
  reaction time used throughout; `t_end - t_start` would carry the ITI.

## Stimulus axis

`config.task.vr_task_v0.orientation_options = [90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10, 5, 0]`
with `distractor_orientation` fixed at **0** in every session. The **target** orientation varies;
the distractor does not. Discriminability is therefore

    delta = |target_orientation - distractor_orientation| = target_orientation

13 levels, evenly sampled (290-340 pooled choice trials per level). Pooled accuracy runs
48.0% at delta=0 (chance, since the two stimuli are then identical) to ~87% at delta=90,
saturating near 85-89% — i.e. a lapse rate of roughly 12-15%.

Two legacy sessions (2026-09-11) store the same quantity under the key
`changing_orientation_target` instead of `target_orientation`; the loader falls back to it.

## samples.csv

`t`, `phase_id`, `trial_id`, `x`, `y`, `heading`, `fwd_v`, `yaw_v`, `fwd_raw`, `yaw_raw`,
`fwd_cmd`, `yaw_cmd`, `current_stim_id`. Median sample interval **0.031 s (~32 Hz)**, despite
`config.task.sample_interval = 0.02`. `trial_id` is present on every sample, so trial
trajectories are extracted by a direct join — the example module's
`add_trial_bounds_from_rewards` fallback is not needed.

## Task geometry

Arena radius 100, `center_radius` 12, two choice boxes `box_9` (left) and `box_3` (right),
`choice_half_angle_deg` 12, `target_left_prob` 0.5, `choice_window_s` 60,
`timeout_after_incorrect` 3, `timeout_after_falls` 2.

## Inclusion

19 of 23 sessions pass (>= 5 orientation levels and >= 50 choice trials),
totalling 4024 choice trials.

Excluded:
- `jlh60/2026-09-11` — legacy variant, 2 choice trials, 29 omissions. **This is jlh60's only
  session, so the animal drops out of the analysis entirely.**
- `jlh62/2026-09-11` — legacy variant, 30 choice trials, 32 falls.
- `jlh62/2026-09-22_14-41-40` — `fixed_90` variant, 3 trials, 1.4 min (aborted).
- `jlh62/2026-09-22_12-09-17` — 38 choice trials with 30 falls.

Included animals: bp1 (7 sessions, 2338 trials), bp2 (1 session, 259 trials),
jlh62 (11 sessions, 1427 trials). **bp2 contributes a single session**, so its per-level counts
are 14-29 trials and its curve will be correspondingly noisy.

Falls (528 trials) and omissions are excluded from accuracy and reaction time, but retained
in the trial table as outcome categories for the path analysis.
