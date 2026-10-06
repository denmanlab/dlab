"""Generate a small synthetic repository with the audited schema.

Purpose is testing, not analysis: the psychometric parameters are known by
construction, so fits can be checked against ground truth, and the degenerate
cases that break plotting code are included deliberately.

    python make_synthetic_fixture.py [outdir]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

LEVELS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]

# Ground truth per animal: (threshold_deg, slope, lapse, rightward_bias)
GROUND_TRUTH = {
    "syn01": (20.0, 0.12, 0.08, 0.00),
    "syn02": (35.0, 0.07, 0.15, 0.10),
    "syn03": (12.0, 0.20, 0.05, -0.05),
}

ARENA_RADIUS = 100.0
CENTER_RADIUS = 12.0
ITI_S = 1.0
SAMPLE_DT = 0.031


def p_correct(delta, threshold, slope, lapse):
    """Logistic with guess rate fixed at 0.5 and a free lapse rate."""
    upper = 1.0 - lapse
    return 0.5 + (upper - 0.5) / (1.0 + np.exp(-slope * (delta - threshold)))


def _trajectory(t0, duration, target_side, correct, rng):
    """A plausible centre-out trajectory ending at the chosen box."""
    n = max(int(duration / SAMPLE_DT), 8)
    t = t0 + np.arange(n) * SAMPLE_DT
    chosen_right = (target_side == "right") == bool(correct)
    end_angle = np.deg2rad(-35.0 if chosen_right else 35.0)
    # Radius grows with a slight overshoot; angle drifts toward the chosen box.
    frac = np.linspace(0, 1, n) ** 0.8
    r = CENTER_RADIUS * 0.2 + (ARENA_RADIUS * 0.92 - CENTER_RADIUS * 0.2) * frac
    wobble = rng.normal(0, 0.05, n).cumsum() * (0.4 if correct else 1.0)
    ang = end_angle * frac + wobble * 0.1
    x, y = r * np.sin(ang), r * np.cos(ang)
    heading = np.rad2deg(ang) + rng.normal(0, 2, n)
    return t, x, y, heading


def make_session(animal, session_name, n_trials, rng, levels=LEVELS, degenerate=None):
    threshold, slope, lapse, bias = GROUND_TRUTH[animal]
    events, samples = [], []
    t = 0.0
    events.append(dict(t_start=0.0, t_end=0.0, event_type="session_start",
                       phase_id="vr_task_v0"))

    for i in range(1, n_trials + 1):
        delta = float(rng.choice(levels))
        target_side = "right" if rng.random() < 0.5 + bias else "left"
        target_box = "box_3" if target_side == "right" else "box_9"
        distractor_box = "box_9" if target_side == "right" else "box_3"

        roll = rng.random()
        if degenerate == "no_errors":
            outcome = "correct"
        elif roll < 0.10:
            outcome = "fall"
        elif roll < 0.12:
            outcome = "omission"
        else:
            outcome = "correct" if rng.random() < p_correct(delta, threshold, slope, lapse) \
                else "incorrect"

        # RT slows near threshold, floor ~3 s.
        rt = 3.0 + 6.0 * np.exp(-((delta - threshold) ** 2) / (2 * 25.0 ** 2)) \
            + rng.gamma(2.0, 1.2)
        if outcome in ("fall", "omission"):
            rt_field = None
            duration = rt
        else:
            rt_field = float(rt)
            duration = rt + ITI_S

        correct = outcome == "correct"
        chosen_box = (target_box if correct else distractor_box) \
            if outcome in ("correct", "incorrect") else None
        chosen_side = None
        if chosen_box is not None:
            chosen_side = "right" if chosen_box == "box_3" else "left"

        tt, x, y, heading = _trajectory(t, min(duration, 25.0), target_side, correct, rng)
        if degenerate == "missing_samples" and i % 7 == 0:
            keep = rng.random(len(tt)) > 0.5
            tt, x, y, heading = tt[keep], x[keep], y[keep], heading[keep]
        for k in range(len(tt)):
            samples.append(dict(t=tt[k], phase_id="vr_task_v0", trial_id=i,
                                x=x[k], y=y[k], heading=heading[k],
                                fwd_v=rng.normal(12, 3), yaw_v=rng.normal(0, 20),
                                fwd_raw=rng.normal(0.1, 0.05), yaw_raw=rng.normal(0, 0.05),
                                fwd_cmd=0.0, yaw_cmd=0.0,
                                current_stim_id=f"target_{int(delta)}"))

        extra = dict(outcome=outcome, chosen_box=chosen_box, correct_box=target_box,
                     target_box=target_box, distractor_box=distractor_box,
                     chosen_side=chosen_side, correct_side=target_side,
                     target_side=target_side,
                     distractor_side="left" if target_side == "right" else "right",
                     target_orientation=delta, distractor_orientation=0.0,
                     choice_latency_s=rt_field)
        events.append(dict(
            t_start=t, t_end=t + duration,
            event_type="fall_trial" if outcome == "fall" else "trial",
            phase_id="vr_task_v0", trial_id=i, rewarded=correct,
            stim_id=None if outcome == "fall" else f"target_{int(delta)}",
            contrast=None if outcome == "fall" else 1.0, orientation=delta,
            user_assisted=False, x=float(x[-1]), y=float(y[-1]),
            heading=float(heading[-1]), extra=json.dumps(extra)))
        t += duration

    events.append(dict(t_start=t, t_end=t, event_type="session_end", phase_id="vr_task_v0"))

    meta = dict(session_id=f"{animal}_{session_name}", mouse_id=animal,
                mouse_weight=25.0 + hash(animal) % 5, task_name="vr_task_v0",
                task_variant="varying_orientation", datetime=session_name,
                config=dict(movement=dict(move_speed=30.0, yaw_speed=130.0),
                            arena=dict(radius=ARENA_RADIUS, center_radius=CENTER_RADIUS),
                            task={"sample_interval": 0.02, "phase_id": "vr_task_v0",
                                  "vr_task_v0": dict(choice_boxes=["box_9", "box_3"],
                                                     choice_half_angle_deg=12.0,
                                                     target_left_prob=0.5,
                                                     orientation_options=[float(v) for v in levels],
                                                     distractor_orientation=0.0,
                                                     contrast=1.0, iti_s=ITI_S,
                                                     choice_window_s=60.0, timeout_s=15.0)}))
    return pd.DataFrame(events), pd.DataFrame(samples), meta


def build(outdir="synthetic_fixture", seed=0):
    rng = np.random.default_rng(seed)
    spec = [
        ("syn01", "2026-01-05_10-00-00", 300, None),
        ("syn01", "2026-01-06_10-00-00", 280, None),
        ("syn01", "2026-01-07_10-00-00", 320, "missing_samples"),
        ("syn02", "2026-01-05_11-00-00", 260, None),
        ("syn02", "2026-01-06_11-00-00", 240, None),
        ("syn03", "2026-01-05_12-00-00", 300, None),
        # 'no_errors' must stay small: a ceiling-performance session large enough
        # to dominate an animal would flatten its psychometric curve and break the
        # ground-truth recovery test. At 30 trials it is also excluded by QC,
        # which is the point -- it tests the exclusion path too.
        ("syn03", "2026-01-06_12-00-00", 30, "no_errors"),
        ("syn03", "2026-01-08_12-00-00", 290, None),
        # Degenerate cases that must not crash any panel:
        ("syn03", "2026-01-07_12-00-00", 4, None),            # aborted session
    ]
    for animal, session, n, degen in spec:
        levels = LEVELS
        if n <= 5:
            levels = [90]                                      # single level, 2-trial cells
        sd = os.path.join(outdir, animal, session)
        os.makedirs(sd, exist_ok=True)
        ev, sa, meta = make_session(animal, session, n, rng, levels, degen)
        ev.to_csv(os.path.join(sd, "events.csv"), index=False)
        sa.to_csv(os.path.join(sd, "samples.csv"), index=False)
        with open(os.path.join(sd, "session_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
    with open(os.path.join(outdir, "ground_truth.json"), "w") as fh:
        json.dump({k: dict(zip(("threshold", "slope", "lapse", "bias"), v))
                   for k, v in GROUND_TRUTH.items()}, fh, indent=1)
    return outdir


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "synthetic_fixture"))
