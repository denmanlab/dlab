"""Ingestion and trial-level metrics for the mouse arena visual-discrimination task.

Repository contract (one folder per animal, one folder per session):

    <root>/<animal>/<YYYY-MM-DD_HH-MM-SS>/{events.csv, samples.csv, session_meta.json}

Extends the two-condition example module to the full orientation range. The
discriminability axis is

    delta = |target_orientation - distractor_orientation|

with the distractor fixed at 0 deg in this dataset, so delta == target orientation.

Trial duration is ``extra.choice_latency_s`` -- the interval from stimulus
presentation to the choice, and so to reward on correct trials. It is called a
duration rather than a reaction time because it contains the whole locomotor
traverse to the box, not just a decision latency. ``t_end - t_start`` is kept
separately as ``trial_window_s``; it adds the configured inter-trial interval
(verified: the difference equals ``iti_s`` to within 7 ms).

All reads are read-only; nothing in the source repository is ever written.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
CHOICE_OUTCOMES = ("correct", "incorrect")
ALL_OUTCOMES = ("correct", "incorrect", "fall", "omission")

# Inclusion thresholds; see schema_report.md.
MIN_LEVELS = 5
MIN_CHOICE_TRIALS = 50

# "Easy" trials: the three largest orientation differences. Accuracy here is a
# ceiling measure -- it reflects engagement, lapses and whether the animal knows
# the rule, largely independent of perceptual sensitivity near threshold.
EASY_MIN_DELTA = 70.0


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def list_animals(root):
    """Animal folder names under `root`, sorted."""
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )


def list_session_folders(animal_dir, newest_first=False):
    """Session folder names under `animal_dir`, sorted by parsed datetime.

    Falls back to string sort when folder names do not parse. Ported from the
    example module, defaulting to chronological order because session index is
    used as a learning axis here.
    """
    subfolders = [
        name for name in os.listdir(animal_dir)
        if os.path.isdir(os.path.join(animal_dir, name)) and not name.startswith(".")
    ]
    parsed = []
    for name in subfolders:
        try:
            parsed.append((name, datetime.strptime(name, DATETIME_FORMAT)))
        except ValueError:
            pass
    if parsed:
        return [n for n, _ in sorted(parsed, key=lambda kv: kv[1], reverse=newest_first)]
    return sorted(subfolders, reverse=newest_first)


def iter_sessions(root):
    """Yield (animal, session, session_dir) for every session in the repository."""
    for animal in list_animals(root):
        animal_dir = os.path.join(root, animal)
        for session in list_session_folders(animal_dir):
            yield animal, session, os.path.join(animal_dir, session)


# --------------------------------------------------------------------------- #
# raw loading
# --------------------------------------------------------------------------- #
def load_behavior_data(session_dir):
    """Return (events, samples, session_meta) for one session folder."""
    events = pd.read_csv(os.path.join(session_dir, "events.csv"))
    samples = pd.read_csv(os.path.join(session_dir, "samples.csv"))
    with open(os.path.join(session_dir, "session_meta.json")) as fh:
        session_meta = json.load(fh)
    return events, samples, session_meta


def parse_extra(value):
    """Parse the `extra` JSON column, tolerating NaN and malformed rows."""
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}


def task_config(session_meta):
    """The per-task config block, e.g. config.task.vr_task_v0."""
    cfg = session_meta.get("config", {}) or {}
    task = cfg.get("task", {}) or {}
    return task.get(session_meta.get("task_name", ""), {}) or {}


def arena_radius(session_meta, samples=None, default=100.0):
    """Arena radius from metadata, falling back to the sample extent."""
    try:
        radius = ((session_meta.get("config", {}) or {}).get("arena", {}) or {}).get("radius")
        if radius is not None and float(radius) > 0:
            return float(radius)
    except (TypeError, ValueError, AttributeError):
        pass
    if samples is not None and {"x", "y"}.issubset(samples.columns):
        r_max = np.nanmax(np.hypot(samples["x"].to_numpy(), samples["y"].to_numpy()))
        if np.isfinite(r_max) and r_max > 0:
            return float(np.ceil(r_max))
    return float(default)


# --------------------------------------------------------------------------- #
# trial table
# --------------------------------------------------------------------------- #
def _target_orientation(extra):
    """Target orientation, with the legacy key fallback.

    Sessions recorded before the `varying_orientation` variant was named store
    the same quantity under `changing_orientation_target`.
    """
    value = extra.get("target_orientation")
    if value is None:
        value = extra.get("changing_orientation_target")
    return value


def build_session_trials(session_dir, animal=None, session=None):
    """One row per trial event (including falls and omissions) for one session."""
    events, samples, meta = load_behavior_data(session_dir)
    trial_rows = events[events["event_type"].isin(["trial", "fall_trial"])].copy()
    extra = trial_rows["extra"].map(parse_extra)

    cfg = task_config(meta)
    iti_s = float(cfg.get("iti_s", np.nan))

    out = pd.DataFrame({
        "animal": animal or meta.get("mouse_id"),
        "session": session or os.path.basename(session_dir.rstrip(os.sep)),
        "trial_id": trial_rows["trial_id"].to_numpy(),
        "event_type": trial_rows["event_type"].to_numpy(),
        "t_start": trial_rows["t_start"].to_numpy(),
        "t_end": trial_rows["t_end"].to_numpy(),
        "target_ori": extra.map(_target_orientation).to_numpy(),
        "distractor_ori": extra.map(lambda d: d.get("distractor_orientation")).to_numpy(),
        "outcome": extra.map(lambda d: d.get("outcome")).to_numpy(),
        "chosen_box": extra.map(lambda d: d.get("chosen_box")).to_numpy(),
        "correct_box": extra.map(lambda d: d.get("correct_box")).to_numpy(),
        "target_box": extra.map(lambda d: d.get("target_box")).to_numpy(),
        "chosen_side": extra.map(lambda d: d.get("chosen_side")).to_numpy(),
        "target_side": extra.map(lambda d: d.get("target_side")).to_numpy(),
        "correct_side": extra.map(lambda d: d.get("correct_side")).to_numpy(),
        "trial_duration_s": pd.to_numeric(
            extra.map(lambda d: d.get("choice_latency_s")), errors="coerce"
        ).to_numpy(),
        "stim_id": trial_rows["stim_id"].to_numpy(),
        "contrast": trial_rows["contrast"].to_numpy(),
        "x_end": trial_rows["x"].to_numpy(),
        "y_end": trial_rows["y"].to_numpy(),
        "heading_end": trial_rows["heading"].to_numpy(),
    })

    out["delta_ori"] = (out["target_ori"] - out["distractor_ori"]).abs()
    # Signed difference relative to the target side: positive = target on the right.
    out["signed_delta"] = np.where(
        out["target_side"].eq("right"), out["delta_ori"], -out["delta_ori"]
    )
    out["is_choice"] = out["outcome"].isin(CHOICE_OUTCOMES)
    out["correct"] = np.where(out["is_choice"], out["outcome"].eq("correct"), np.nan)
    out["chose_right"] = np.where(
        out["is_choice"], out["chosen_side"].eq("right").astype(float), np.nan
    )
    out["trial_window_s"] = out["t_end"] - out["t_start"]
    out["iti_s"] = iti_s
    out["trial_in_session"] = np.arange(1, len(out) + 1)
    out["mouse_weight_g"] = meta.get("mouse_weight")
    out["task_variant"] = meta.get("task_variant") or "legacy_no_variant"
    out["arena_radius"] = arena_radius(meta, samples)
    return out


def build_trial_table(root, sessions=None):
    """Trial table for the whole repository (or a given (animal, session) subset)."""
    wanted = set(map(tuple, sessions)) if sessions is not None else None
    frames = []
    for animal, session, session_dir in iter_sessions(root):
        if wanted is not None and (animal, session) not in wanted:
            continue
        frames.append(build_session_trials(session_dir, animal, session))
    if not frames:
        return pd.DataFrame()
    trials = pd.concat(frames, ignore_index=True)

    order = (
        trials[["animal", "session"]].drop_duplicates()
        .sort_values(["animal", "session"]).reset_index(drop=True)
    )
    order["session_index"] = order.groupby("animal").cumcount() + 1
    trials = trials.merge(order, on=["animal", "session"], how="left")
    trials["date"] = pd.to_datetime(trials["session"], format=DATETIME_FORMAT)
    return trials


# --------------------------------------------------------------------------- #
# session inventory
# --------------------------------------------------------------------------- #
def build_session_table(root, trials=None):
    """One row per session: counts, rates, QC flags and the inclusion decision."""
    if trials is None:
        trials = build_trial_table(root)

    def _agg(g):
        choice = g[g["is_choice"]]
        easy = choice[choice["delta_ori"] >= EASY_MIN_DELTA]
        n_choice = len(choice)
        levels = sorted(g["delta_ori"].dropna().unique())
        return pd.Series({
            "session_index": g["session_index"].iloc[0],
            "date": g["date"].iloc[0],
            "task_variant": g["task_variant"].iloc[0],
            "weight_g": g["mouse_weight_g"].iloc[0],
            "duration_min": round(float(g["t_end"].max()) / 60.0, 2),
            "n_trial_rows": len(g),
            "n_choice_trials": n_choice,
            "n_correct": int((g["outcome"] == "correct").sum()),
            "n_incorrect": int((g["outcome"] == "incorrect").sum()),
            "n_fall": int((g["outcome"] == "fall").sum()),
            "n_omission": int((g["outcome"] == "omission").sum()),
            "pct_correct": round(100 * choice["correct"].mean(), 1) if n_choice else np.nan,
            "median_duration_s": round(float(choice["trial_duration_s"].median()), 2)
                                 if n_choice else np.nan,
            "n_easy_trials": int(len(easy)),
            "pct_correct_easy": round(100 * easy["correct"].mean(), 1) if len(easy) else np.nan,
            "median_window_s": round(float(choice["trial_window_s"].median()), 2)
                               if n_choice else np.nan,
            "n_levels": len(levels),
            "levels": ",".join(str(int(v)) for v in levels),
        })

    sessions = (
        trials.groupby(["animal", "session"], sort=True)
        .apply(_agg, include_groups=False).reset_index()
    )

    flags = []
    for _, r in sessions.iterrows():
        f = []
        if r["task_variant"] == "legacy_no_variant":
            f.append("legacy_variant")
        if r["n_levels"] <= 1:
            f.append("single_level")
        if r["n_choice_trials"] < MIN_CHOICE_TRIALS:
            f.append("few_trials")
        if r["duration_min"] < 10:
            f.append("short_session")
        if r["n_trial_rows"] and r["n_omission"] / r["n_trial_rows"] > 0.15:
            f.append("high_omission")
        if r["n_trial_rows"] and r["n_fall"] / r["n_trial_rows"] > 0.25:
            f.append("high_fall")
        flags.append(",".join(f) or "ok")
    sessions["qc_flags"] = flags
    sessions["include"] = (
        (sessions["n_levels"] >= MIN_LEVELS)
        & (sessions["n_choice_trials"] >= MIN_CHOICE_TRIALS)
    )
    return sessions.sort_values(["animal", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# psychometric fitting
# --------------------------------------------------------------------------- #
def _link_logistic(z):
    from scipy.special import expit
    return expit(z)


def _link_gauss(z):
    from scipy.stats import norm
    return norm.cdf(z)


def _link_gumbel(z):
    return 1.0 - np.exp(-np.exp(np.clip(z, -50, 5)))


# Psychometric function families. Each maps (delta, mu, slope) to a proportion in
# (0, 1) that is then scaled between the fixed 0.5 guess rate and the 1 - lapse
# asymptote. `mu` is a location (or scale, for Weibull) in degrees and `slope` a
# steepness; `bounds` and `inits` are per-family because the two parameters are
# not on the same scale across families.
def _f_logistic(delta, mu, slope):
    return _link_logistic(slope * (np.asarray(delta, float) - mu))


def _f_gauss(delta, mu, slope):
    return _link_gauss(slope * (np.asarray(delta, float) - mu))


def _f_gumbel(delta, mu, slope):
    return _link_gumbel(slope * (np.asarray(delta, float) - mu))


def _f_weibull(delta, mu, slope):
    d = np.clip(np.asarray(delta, float), 0.0, None)
    mu = max(float(mu), 1e-6)
    return 1.0 - np.exp(-np.power(d / mu, slope))


FAMILIES = {
    "logistic": dict(
        fn=_f_logistic, label="logistic",
        equation=r"$p(\Delta)=0.5+(0.5-\lambda)\,/\,[1+e^{-\beta(\Delta-\mu)}]$",
        slope_bounds=(1e-3, 5.0), slope_inits=(0.05, 0.15, 0.4),
        mu_label="midpoint", slope_label=r"deg$^{-1}$",
        mu_positive=False),
    "gauss": dict(
        fn=_f_gauss, label="cumulative Gaussian",
        equation=r"$p(\Delta)=0.5+(0.5-\lambda)\,\Phi[\beta(\Delta-\mu)]$",
        slope_bounds=(1e-3, 5.0), slope_inits=(0.03, 0.09, 0.25),
        mu_label="midpoint", slope_label=r"deg$^{-1}$",
        mu_positive=False),
    "gumbel": dict(
        fn=_f_gumbel, label="Gumbel",
        equation=r"$p(\Delta)=0.5+(0.5-\lambda)\,[1-e^{-e^{\beta(\Delta-\mu)}}]$",
        slope_bounds=(1e-3, 5.0), slope_inits=(0.03, 0.08, 0.2),
        mu_label="location", slope_label=r"deg$^{-1}$",
        mu_positive=False),
    "weibull": dict(
        fn=_f_weibull, label="Weibull",
        equation=r"$p(\Delta)=0.5+(0.5-\lambda)\,[1-e^{-(\Delta/\mu)^{\beta}}]$",
        slope_bounds=(0.2, 12.0), slope_inits=(1.0, 2.0, 4.0),
        mu_label="scale", slope_label=r"shape, unitless",
        mu_positive=True),
}
DEFAULT_FAMILY = "logistic"


def psychometric(delta, mu, slope, lapse, family=DEFAULT_FAMILY):
    """Psychometric function with the guess rate fixed at 0.5 and a free lapse.

        p(delta) = 0.5 + (0.5 - lapse) * F(delta; mu, slope)

    `F` is the family's shape function -- see FAMILIES for the available
    choices (logistic, cumulative Gaussian, Gumbel, Weibull).

    The guess rate is fixed rather than fitted because at delta = 0 the target
    and distractor are physically identical, so chance performance is a property
    of the task, not a free parameter. The 75%-correct threshold is derived
    separately by `threshold_at`.
    """
    spec = FAMILIES[family]
    return 0.5 + (0.5 - lapse) * spec["fn"](delta, mu, slope)


def threshold_at(mu, slope, lapse, level=0.75, family=DEFAULT_FAMILY,
                 lo=0.0, hi=90.0):
    """Stimulus difference at which the fitted curve reaches `level` correct.

    Solved numerically by bisection so it works for every family, and returns
    NaN when the curve does not cross `level` inside [lo, hi] -- which happens
    when the lapse rate puts the asymptote below the criterion.
    """
    from scipy.optimize import brentq

    if not np.isfinite([mu, slope, lapse]).all():
        return np.nan
    if not (0.5 < level < 1.0 - lapse):
        return np.nan

    def g(x):
        return float(psychometric(x, mu, slope, lapse, family)) - level

    try:
        if g(lo) > 0 or g(hi) < 0:
            return np.nan
        return float(brentq(g, lo, hi, xtol=1e-4))
    except (ValueError, RuntimeError):
        return np.nan


def fit_psychometric(delta, n_correct, n_total, max_lapse=0.4,
                     family=DEFAULT_FAMILY):
    """Maximum-likelihood fit of one psychometric family to binomial counts.

    The estimator is the binomial likelihood for every family, because the data
    are counts of correct responses out of n at each level; least-squares on the
    proportions would ignore the unequal trial counts per level.

    Parameters
    ----------
    delta, n_correct, n_total : array-like, one entry per stimulus level.
    family : one of FAMILIES.

    Returns a dict with family, mu, slope, lapse, threshold_75, nll, aic,
    converged, reliable, fit_flags, n_trials, n_levels. Parameters are NaN when
    fewer than 4 levels carry trials, since a three-parameter fit is not
    identifiable below that.
    """
    from scipy.optimize import minimize

    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {list(FAMILIES)}")
    spec = FAMILIES[family]

    delta = np.asarray(delta, dtype=float)
    k = np.asarray(n_correct, dtype=float)
    n = np.asarray(n_total, dtype=float)
    keep = n > 0
    delta, k, n = delta[keep], k[keep], n[keep]
    nan = dict(family=family, mu=np.nan, slope=np.nan, lapse=np.nan,
               threshold_75=np.nan, nll=np.nan, aic=np.nan, converged=False,
               reliable=False, fit_flags="too_few_levels",
               n_trials=int(n.sum()), n_levels=len(delta))
    if len(delta) < 4:
        return nan

    def nll(theta):
        mu, log_slope, lapse = theta
        p = psychometric(delta, mu, np.exp(log_slope), lapse, family)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.sum(k * np.log(p) + (n - k) * np.log1p(-p)))

    span = max(delta.max() - delta.min(), 1.0)
    lo_mu = max(delta.min() - span, 1e-3) if spec["mu_positive"] else delta.min() - span
    s_lo, s_hi = spec["slope_bounds"]
    bounds = [(lo_mu, delta.max() + span),
              (np.log(s_lo), np.log(s_hi)), (0.0, max_lapse)]

    best = None
    for frac in (0.25, 0.5, 0.75):
        mu0 = max(delta.min() + frac * span, lo_mu)
        for s0 in spec["slope_inits"]:
            for lp0 in (0.02, 0.10, 0.20):
                try:
                    r = minimize(nll, [mu0, np.log(s0), lp0], method="L-BFGS-B",
                                 bounds=bounds)
                except Exception:
                    continue
                if r.success and (best is None or r.fun < best.fun):
                    best = r
    if best is None:
        return nan

    mu, slope, lapse = best.x[0], float(np.exp(best.x[1])), best.x[2]
    thr = threshold_at(mu, slope, lapse, 0.75, family,
                       lo=float(delta.min()), hi=float(delta.max()))

    # Identifiability. The likelihood is shallow in `slope` at realistic trial
    # counts, so a fit can "converge" on a parameter set that says little. Flag
    # the cases where the returned numbers should not be read as estimates.
    reasons = []
    if not (delta.min() <= mu <= delta.max()):
        reasons.append("mu_outside_range")
    if slope <= s_lo * 1.1 or slope >= s_hi * 0.9:
        reasons.append("slope_at_bound")
    if not np.isfinite(thr):
        reasons.append("no_75pct_crossing")
    if n.sum() < 200:
        reasons.append("few_trials")

    return dict(family=family, mu=float(mu), slope=slope, lapse=float(lapse),
                threshold_75=thr, nll=float(best.fun),
                aic=float(2 * 3 + 2 * best.fun), converged=True,
                reliable=not reasons, fit_flags=",".join(reasons) or "ok",
                n_trials=int(n.sum()), n_levels=len(delta))


def compare_families(delta, n_correct, n_total, families=None, max_lapse=0.4):
    """Fit every family to the same counts and rank them by AIC."""
    families = families or list(FAMILIES)
    rows = [fit_psychometric(delta, n_correct, n_total, max_lapse, f)
            for f in families]
    out = pd.DataFrame(rows)
    if "aic" in out and out["aic"].notna().any():
        out["delta_aic"] = out["aic"] - out["aic"].min()
        out = out.sort_values("aic")
    return out.reset_index(drop=True)


def level_counts(trials, by=None, level_col="delta_ori"):
    """Per-level correct/total counts with the binomial SEM, for choice trials."""
    df = trials[trials["is_choice"]].copy()
    keys = ([] if by is None else list(by)) + [level_col]
    g = df.groupby(keys, dropna=True)["correct"].agg(n="size", k="sum").reset_index()
    g["p"] = g["k"] / g["n"]
    g["sem"] = np.sqrt(g["p"] * (1 - g["p"]) / g["n"])
    return g


def fit_by(trials, by=("animal",), n_boot=0, seed=0, level_col="delta_ori",
           family=DEFAULT_FAMILY):
    """Fit the psychometric curve within each group of `by`.

    With n_boot > 0, adds bootstrap 95% CIs for mu, slope, lapse and
    threshold_75 by resampling trials with replacement within the group.
    """
    rng = np.random.default_rng(seed)
    df = trials[trials["is_choice"]]
    rows = []
    for key, g in df.groupby(list(by), dropna=True):
        counts = level_counts(g, level_col=level_col)
        fit = fit_psychometric(counts[level_col], counts["k"], counts["n"],
                               family=family)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(fit)
        if n_boot and fit["converged"]:
            boot = {p: [] for p in ("mu", "slope", "lapse", "threshold_75")}
            vals = g[[level_col, "correct"]].to_numpy()
            for _ in range(n_boot):
                idx = rng.integers(0, len(vals), len(vals))
                bs = pd.DataFrame(vals[idx], columns=[level_col, "correct"])
                c = bs.groupby(level_col)["correct"].agg(n="size", k="sum").reset_index()
                f = fit_psychometric(c[level_col], c["k"], c["n"], family=family)
                if f["converged"]:
                    for p in boot:
                        boot[p].append(f[p])
            for p, v in boot.items():
                v = np.asarray(v, dtype=float)
                v = v[np.isfinite(v)]
                rec[f"{p}_lo"] = float(np.percentile(v, 2.5)) if len(v) > 10 else np.nan
                rec[f"{p}_hi"] = float(np.percentile(v, 97.5)) if len(v) > 10 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# trajectories
# --------------------------------------------------------------------------- #
def session_trajectories(session_dir):
    """Per-sample trajectory frame for one session, keyed by trial_id.

    `samples.trial_id` is populated on every sample, so trial trajectories are a
    direct join; the example module's add_trial_bounds_from_rewards fallback is
    not needed for this repository.
    """
    samples = pd.read_csv(os.path.join(session_dir, "samples.csv"))
    samples = samples.dropna(subset=["trial_id", "x", "y"]).copy()
    samples["trial_id"] = samples["trial_id"].astype(int)
    return samples


def path_metrics(samples, trials, center_radius=12.0):
    """Per-trial path metrics from the sample trace.

    Returns one row per trial_id present in both inputs:
      path_length      cumulative distance travelled
      net_displacement straight-line start-to-end distance
      straightness     net_displacement / path_length  (1 = perfectly direct)
      mean_speed, peak_speed
      t_leave_center   time from trial start to first exit of the centre zone
      total_turning    cumulative |heading change|, degrees
      n_samples
    """
    out = []
    by_trial = samples.groupby("trial_id")
    for trial_id, g in by_trial:
        g = g.sort_values("t")
        x, y, t = g["x"].to_numpy(), g["y"].to_numpy(), g["t"].to_numpy()
        if len(g) < 3:
            continue
        step = np.hypot(np.diff(x), np.diff(y))
        dt = np.diff(t)
        path_length = float(step.sum())
        net = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        # Sample timing jitters; intervals far below the median produce spurious
        # speeds (values >1000 units/s against a 30 units/s move_speed config).
        # Ignore intervals shorter than half the session's median interval.
        dt_floor = 0.5 * float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > dt_floor, step / dt, np.nan)
        r = np.hypot(x, y)
        outside = np.flatnonzero(r > center_radius)
        t_leave = float(t[outside[0]] - t[0]) if outside.size else np.nan
        if "heading" in g:
            dh = np.diff(np.unwrap(np.deg2rad(g["heading"].to_numpy())))
            turning = float(np.abs(np.rad2deg(dh)).sum())
        else:
            turning = np.nan
        out.append({
            "trial_id": int(trial_id),
            "path_length": path_length,
            "net_displacement": net,
            "straightness": net / path_length if path_length > 0 else np.nan,
            "mean_speed": float(np.nanmean(speed)),
            # 95th percentile rather than the maximum: robust to residual jitter.
            "peak_speed": float(np.nanpercentile(speed, 95)) if np.isfinite(speed).any() else np.nan,
            "t_leave_center": t_leave,
            "total_turning": turning,
            "n_samples": int(len(g)),
        })
    metrics = pd.DataFrame(out)
    if metrics.empty:
        return metrics
    return trials.merge(metrics, on="trial_id", how="inner")


# --------------------------------------------------------------------------- #
# within-trial heading and commitment
# --------------------------------------------------------------------------- #
# Heading convention, established empirically against the sample traces rather
# than assumed: travel direction in the standard math convention equals
# `heading + 90`, i.e. heading 0 points along +y, straight ahead toward the
# choice boxes. Verified at 0.00 deg median error across 60 trials.
HEADING_OFFSET_DEG = 90.0


def box_positions(arena_radius=100.0, half_angle_deg=12.0, inset=1.0):
    """World coordinates of the two choice boxes, per the task's arena layout."""
    r = arena_radius - inset
    out = {}
    for box, sign in (("box_9", +1.0), ("box_3", -1.0)):
        t = np.deg2rad(90.0 + sign * half_angle_deg)
        out[box] = (r * np.cos(t), r * np.sin(t))
    return out


def _wrap180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def heading_error_to(x, y, heading_deg, target_xy):
    """Signed angle between the animal's heading and the direction to a point.

    Zero means heading straight at the target; the sign gives which side it is
    on. Returned in degrees in (-180, 180].
    """
    bx, by = target_xy
    bearing = np.degrees(np.arctan2(by - np.asarray(y), bx - np.asarray(x)))
    return _wrap180(bearing - (np.asarray(heading_deg) + HEADING_OFFSET_DEG))


def trial_heading_trace(g, chosen_xy, other_xy, n_grid=60):
    """Heading error toward the chosen and unchosen box over one trial.

    Returns (t_rel, frac, err_chosen, err_other) resampled onto `n_grid` points
    of normalised within-trial time, plus the raw arrays.
    """
    g = g.sort_values("t")
    t = g["t"].to_numpy()
    x, y, h = g["x"].to_numpy(), g["y"].to_numpy(), g["heading"].to_numpy()
    t_rel = t - t[0]
    ec = heading_error_to(x, y, h, chosen_xy)
    eo = heading_error_to(x, y, h, other_xy)
    if t_rel[-1] <= 0:
        return None
    frac = t_rel / t_rel[-1]
    grid = np.linspace(0, 1, n_grid)
    return dict(frac=grid,
                err_chosen=np.interp(grid, frac, np.abs(ec)),
                err_other=np.interp(grid, frac, np.abs(eo)),
                t_rel=t_rel, raw_chosen=ec, raw_other=eo, dur=t_rel[-1])


def commitment_metrics(g, chosen_xy, other_xy, align_thresh_deg=30.0,
                       smooth_n=5, arrival_radius=15.0, dead_zone_deg=5.0):
    """When, within a trial, the animal committed to the box it chose.

    Two independent estimators, reported together because they answer slightly
    different questions:

    t_commit_align
        Last time the heading crossed into sustained alignment with the chosen
        box: the earliest time after which |heading error| stays below
        `align_thresh_deg` for the rest of the trial. This is the commitment
        that stuck.
    t_commit_turn
        Time of the peak smoothed angular speed, i.e. the main steering turn.
        Normally precedes t_commit_align.

    n_reversals
        How many times the box the animal was heading more directly toward
        switched. Zero means a single committed run; higher values mean the
        animal changed its mind en route, which is the signature that
        distinguishes late switching from early commitment.
    """
    tr = trial_heading_trace(g, chosen_xy, other_xy)
    if tr is None:
        return None
    t_rel, ec, eo, dur = tr["t_rel"], tr["raw_chosen"], tr["raw_other"], tr["dur"]

    # Drop the arrival phase. Within `arrival_radius` of the box the bearing to
    # it is geometrically unstable -- it swings through large angles for small
    # movements -- so heading error there is noise, not steering. Without this
    # mask the "stays aligned to the end" criterion fails on most trials.
    gs = g.sort_values("t")
    dist_chosen = np.hypot(gs["x"].to_numpy() - chosen_xy[0],
                           gs["y"].to_numpy() - chosen_xy[1])
    approach = dist_chosen > arrival_radius
    if approach.sum() < 4:
        approach = np.ones_like(dist_chosen, dtype=bool)
    last = int(np.flatnonzero(approach)[-1])          # final pre-arrival sample
    ec_a, eo_a, t_a = ec[:last + 1], eo[:last + 1], t_rel[:last + 1]

    # Decision variable: how much more directly the heading points at the chosen
    # box than at the other one. Positive = leaning toward the box eventually
    # chosen. This is used instead of raw alignment because the two boxes sit
    # only 2 x choice_half_angle_deg apart (24 deg), so any fixed alignment
    # threshold wide enough to be robust also admits the other box.
    dv = np.abs(eo_a) - np.abs(ec_a)
    if len(dv) >= smooth_n:
        kern = np.ones(smooth_n) / smooth_n
        dv_s = np.convolve(dv, kern, mode="same")
    else:
        dv_s = dv

    # A dead zone keeps heading jitter from registering as a change of mind.
    committed = np.where(dv_s > dead_zone_deg, 1,
                         np.where(dv_s < -dead_zone_deg, -1, 0))

    # Earliest time after which the animal stays committed to the chosen box
    # for the remainder of the approach.
    # Also record where in the arena the animal was at that moment: `t_a` is a
    # prefix of the time-sorted arrays, so the same index addresses x and y.
    t_align, commit_x, commit_y = np.nan, np.nan, np.nan
    not_chosen = np.flatnonzero(committed != 1)
    idx = 0 if not_chosen.size == 0 else (not_chosen[-1] + 1)
    if idx < len(t_a):
        t_align = float(t_a[idx])
        commit_x = float(gs["x"].to_numpy()[idx])
        commit_y = float(gs["y"].to_numpy()[idx])

    # Peak angular speed (the main turn).
    t_turn = np.nan
    if len(t_rel) > smooth_n + 2:
        h = np.unwrap(np.deg2rad(g.sort_values("t")["heading"].to_numpy()))
        dt = np.diff(t_rel)
        with np.errstate(divide="ignore", invalid="ignore"):
            omega = np.abs(np.diff(h) / np.where(dt > 0, dt, np.nan))
        k = np.ones(smooth_n) / smooth_n
        if np.isfinite(omega).sum() > smooth_n:
            sm = np.convolve(np.nan_to_num(omega), k, mode="same")
            t_turn = float(t_rel[1:][int(np.nanargmax(sm))])

    # Changes of mind: transitions between the two committed states, ignoring
    # passages through the neutral dead zone.
    states = committed[committed != 0]
    n_reversals = int(np.sum(np.diff(states) != 0)) if states.size > 1 else 0
    toward_chosen = committed == 1

    return dict(t_commit_align=t_align, t_commit_turn=t_turn,
                commit_x=commit_x, commit_y=commit_y,
                commit_r=float(np.hypot(commit_x, commit_y))
                         if np.isfinite(commit_x) else np.nan,
                frac_commit_align=t_align / dur if np.isfinite(t_align) and dur > 0 else np.nan,
                n_reversals=n_reversals,
                frac_toward_chosen=float(np.mean(toward_chosen)),
                mean_abs_err_chosen=float(np.nanmean(np.abs(ec_a))),
                approach_s=float(t_a[-1]) if len(t_a) else np.nan,
                duration_s=float(dur))


def build_commitment_table(root, sessions=None, align_thresh_deg=30.0,
                           n_grid=60, return_traces=True):
    """Per-trial commitment metrics, and the mean heading traces for plotting.

    Returns (metrics_df, traces_df). `traces_df` is long-form with one row per
    (animal, session, trial_id, normalised-time bin), which is what the
    within-trial panels average over.
    """
    wanted = set(map(tuple, sessions)) if sessions is not None else None
    met_rows, trace_rows = [], []
    for animal, session, session_dir in iter_sessions(root):
        if wanted is not None and (animal, session) not in wanted:
            continue
        trials = build_session_trials(session_dir, animal, session)
        choice = trials[trials["is_choice"] & trials["chosen_box"].notna()]
        if choice.empty:
            continue
        samples = session_trajectories(session_dir)
        if samples.empty:
            continue
        radius = float(trials["arena_radius"].dropna().iloc[0]) if len(trials) else 100.0
        boxes = box_positions(radius)
        by_trial = dict(tuple(samples.groupby("trial_id")))

        for _, r in choice.iterrows():
            g = by_trial.get(int(r["trial_id"]))
            if g is None or len(g) < 6:
                continue
            chosen = boxes.get(r["chosen_box"])
            other = boxes.get("box_3" if r["chosen_box"] == "box_9" else "box_9")
            if chosen is None or other is None:
                continue
            m = commitment_metrics(g, chosen, other, align_thresh_deg)
            if m is None:
                continue
            m.update(animal=animal, session=session, trial_id=int(r["trial_id"]),
                     outcome=r["outcome"], delta_ori=r["delta_ori"],
                     correct=r["correct"], trial_duration_s=r["trial_duration_s"],
                     session_index=None)
            met_rows.append(m)
            if return_traces:
                tr = trial_heading_trace(g, chosen, other, n_grid=n_grid)
                if tr is not None:
                    trace_rows.append(pd.DataFrame({
                        "animal": animal, "session": session,
                        "trial_id": int(r["trial_id"]), "outcome": r["outcome"],
                        "delta_ori": r["delta_ori"], "frac": tr["frac"],
                        "t_rel": tr["frac"] * tr["dur"],
                        "err_chosen": tr["err_chosen"], "err_other": tr["err_other"]}))

    metrics = pd.DataFrame(met_rows)
    traces = pd.concat(trace_rows, ignore_index=True) if trace_rows else pd.DataFrame()
    if not metrics.empty:
        order = (metrics[["animal", "session"]].drop_duplicates()
                 .sort_values(["animal", "session"]).reset_index(drop=True))
        order["session_index"] = order.groupby("animal").cumcount() + 1
        metrics = metrics.drop(columns=["session_index"]).merge(
            order, on=["animal", "session"], how="left")
    return metrics, traces


def build_path_table(root, sessions=None, center_radius=12.0):
    """Path metrics joined to trial records, for the whole repository or a subset."""
    wanted = set(map(tuple, sessions)) if sessions is not None else None
    frames = []
    for animal, session, session_dir in iter_sessions(root):
        if wanted is not None and (animal, session) not in wanted:
            continue
        trials = build_session_trials(session_dir, animal, session)
        samples = session_trajectories(session_dir)
        got = path_metrics(samples, trials, center_radius=center_radius)
        if not got.empty:
            frames.append(got)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
