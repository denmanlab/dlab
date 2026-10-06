"""Panel library for the arena discrimination dashboard.

Every panel is a pure function

    panel(ds: Dataset, params: dict) -> matplotlib.figure.Figure

with no Streamlit import, no file writing and no global state, registered in
`PANELS`. Two thin renderers consume the registry: `preview.py` (standalone
HTML) and `dashboard_app.py` (Streamlit). Whatever is settled in the preview is
literally what the app renders.

Stimulus convention (verified against the task source, mouse_arena_app):
0 deg is a vertical grating, 90 deg horizontal. The distractor is fixed at
0 deg; the rewarded target carries the varying orientation. Difficulty is
delta = |target - distractor|, so delta = 0 means the two gratings are
identical and performance is at chance by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.figure import Figure

import arena_discrim as ad

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
ANIMAL_PALETTE = ["#1b6ca8", "#d1495b", "#1a936f", "#8c5e8a", "#e58f2a", "#4a4e69"]
OUTCOME_COLORS = {"correct": "#1a936f", "incorrect": "#d1495b",
                  "fall": "#9a9a9a", "omission": "#c9c9c9"}
GROUP_COLOR = "#222222"

# Arena geometry, read from mouse_arena_app/arena/arena_scene.py:
# both choice boxes sit in the front field of view at +/- choice_half_angle_deg.
BOX_HALF_ANGLE_DEG = 12.0
BOX_INSET = 1.0


def apply_style():
    """Base style: Arial throughout, no grid, no top/right spines, set type sizes."""
    mpl.rcParams.update({
        # Arial everywhere, with fallbacks if a machine lacks it. mathtext is set
        # to the same family so the Greek in the fit equation matches the axis
        # labels instead of falling back to DejaVu.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "Arimo", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "it",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",     # keep text as text in exported SVG
    })


def animal_colors(animals):
    return {a: ANIMAL_PALETTE[i % len(ANIMAL_PALETTE)] for i, a in enumerate(sorted(animals))}


# Which spread measure the error bars show. Set once per panel by `render()` /
# `panel_data()` from params["errorbar"], rather than threaded through every
# aggregation call site. Panels call `spread()` and never read this directly.
_SPREAD = "sem"


def _set_spread(params):
    global _SPREAD
    _SPREAD = "sd" if (params or {}).get("errorbar") == "sd" else "sem"


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def _sd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.std(x, ddof=1) if len(x) > 1 else np.nan


def spread(x):
    """SEM or SD of `x`, per the current setting."""
    return _sd(x) if _SPREAD == "sd" else _sem(x)


def spread_label():
    return "SD" if _SPREAD == "sd" else "SEM"


def binom_spread(p, n):
    """Spread of a binomial proportion.

    SEM is sqrt(p(1-p)/n); the SD counterpart is the per-trial sqrt(p(1-p)),
    which does not shrink with sample size.
    """
    p = np.asarray(p, dtype=float)
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    var = p * (1.0 - p)
    return np.sqrt(var) if _SPREAD == "sd" else np.sqrt(var / n)


def caption_for(name, params=None):
    """Panel caption with the error-bar measure substituted in."""
    _set_spread(params)
    return PANELS[name]["caption"].replace("{err}", spread_label())


def _empty(fig, message):
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, color="#666")
    ax.set_axis_off()
    return fig


# --------------------------------------------------------------------------- #
# selection + dataset
# --------------------------------------------------------------------------- #
@dataclass
class Selection:
    """What the sidebar controls. The only input both renderers filter on."""
    animals: list | None = None                  # None = all
    sessions: list | None = None                 # list of (animal, session); None = all
    date_range: tuple | None = None              # (start, end) inclusive, as Timestamps
    min_trials_per_level: int = 0
    include_only_qc_pass: bool = True
    outcomes: tuple = ("correct", "incorrect")   # which outcomes count as choices

    def describe(self):
        n_a = len(self.animals) if self.animals else "all"
        n_s = len(self.sessions) if self.sessions else "all"
        return f"{n_a} animals, {n_s} sessions"


@dataclass
class Dataset:
    """Filtered trial, path and session tables plus cached psychometric fits."""
    trials: pd.DataFrame
    paths: pd.DataFrame
    sessions: pd.DataFrame
    selection: Selection
    commit: pd.DataFrame = field(default_factory=pd.DataFrame)
    traces: pd.DataFrame = field(default_factory=pd.DataFrame)
    root: str | None = None
    _fits: dict = field(default_factory=dict, repr=False)
    _samples: dict = field(default_factory=dict, repr=False)

    def samples(self, animal, session):
        """Raw sample trace for one session, loaded on demand and cached.

        Only the trajectory panels need this; everything else runs off the
        cached trial tables, so a selection change does not re-read samples.
        """
        import os
        key = (animal, session)
        if key not in self._samples:
            if self.root is None:
                return pd.DataFrame()
            self._samples[key] = ad.session_trajectories(
                os.path.join(self.root, animal, session))
        return self._samples[key]

    @property
    def animals(self):
        return sorted(self.trials["animal"].unique())

    @property
    def colors(self):
        return animal_colors(self.animals)

    @property
    def choice(self):
        return self.trials[self.trials["is_choice"]]

    FIT_COLS = ("mu", "slope", "lapse", "threshold_75", "nll", "converged",
                "reliable", "fit_flags", "n_trials", "n_levels")

    def fits(self, by=("animal",), n_boot=0, family=None):
        family = family or ad.DEFAULT_FAMILY
        key = (tuple(by), n_boot, family)
        if key not in self._fits:
            if self.choice.empty:
                # An over-restrictive filter can empty the selection; return the
                # expected shape so panels degrade to an empty message rather
                # than raising on a missing column.
                self._fits[key] = pd.DataFrame(columns=list(by) + list(self.FIT_COLS))
            else:
                self._fits[key] = ad.fit_by(self.choice, by=by, n_boot=n_boot,
                                            family=family)
        return self._fits[key]


def load_tables(root=None, trials_path="trials.parquet", paths_path="path_metrics.parquet",
                commit_path="commitment.parquet", traces_path="heading_traces.parquet"):
    """Load cached tables, rebuilding from `root` if the caches are absent.

    Returns (trials, paths, commit, traces). The commitment tables are optional:
    the within-trial panels degrade to a message if they are missing.
    """
    import os
    if os.path.exists(trials_path) and os.path.exists(paths_path):
        trials = pd.read_parquet(trials_path)
        paths = pd.read_parquet(paths_path)
    elif root is not None:
        trials = ad.build_trial_table(root)
        paths = ad.build_path_table(root)
    else:
        raise FileNotFoundError("No cached tables and no data root given.")

    commit = pd.read_parquet(commit_path) if os.path.exists(commit_path) else pd.DataFrame()
    traces = pd.read_parquet(traces_path) if os.path.exists(traces_path) else pd.DataFrame()
    return trials, paths, commit, traces


def make_dataset(trials, paths, sessions, selection, root=None,
                 commit=None, traces=None):
    """Apply a Selection to the full tables."""
    s = selection
    keep_sessions = sessions.copy()
    if s.include_only_qc_pass:
        keep_sessions = keep_sessions[keep_sessions["include"]]
    if s.animals:
        keep_sessions = keep_sessions[keep_sessions["animal"].isin(s.animals)]
    if s.sessions:
        want = set(map(tuple, s.sessions))
        keep_sessions = keep_sessions[
            [(a, ss) in want for a, ss in zip(keep_sessions["animal"], keep_sessions["session"])]
        ]
    if s.date_range:
        lo, hi = s.date_range
        keep_sessions = keep_sessions[
            (keep_sessions["date"] >= lo) & (keep_sessions["date"] <= hi)
        ]

    key = keep_sessions[["animal", "session"]]
    t = trials.merge(key, on=["animal", "session"], how="inner")
    p = paths.merge(key, on=["animal", "session"], how="inner")
    cm = (commit.merge(key, on=["animal", "session"], how="inner")
          if commit is not None and not commit.empty else pd.DataFrame())
    tc = (traces.merge(key, on=["animal", "session"], how="inner")
          if traces is not None and not traces.empty else pd.DataFrame())

    if s.min_trials_per_level > 0 and not t.empty:
        counts = (t[t["is_choice"]].groupby(["animal", "delta_ori"]).size()
                  .rename("n").reset_index())
        good = counts[counts["n"] >= s.min_trials_per_level][["animal", "delta_ori"]]
        t = t.merge(good, on=["animal", "delta_ori"], how="inner")
        p = p.merge(good, on=["animal", "delta_ori"], how="inner")
        if not cm.empty:
            cm = cm.merge(good, on=["animal", "delta_ori"], how="inner")
        if not tc.empty:
            tc = tc.merge(good, on=["animal", "delta_ori"], how="inner")

    return Dataset(trials=t, paths=p, sessions=keep_sessions, selection=s,
                   commit=cm, traces=tc, root=root)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
PANELS = {}


def panel(name, title, view, section=None, figsize=(6.4, 4.4), caption=""):
    """Register a panel function under `name`."""
    def deco(fn):
        PANELS[name] = dict(name=name, title=title, view=view, section=section or view,
                            figsize=figsize, caption=caption, fn=fn,
                            data=globals().get(f"data_{name}"))
        return fn
    return deco


def render(name, ds, params=None):
    """Render one registered panel to a Figure."""
    spec = PANELS[name]
    _set_spread(params)
    apply_style()
    fig = Figure(figsize=spec["figsize"], constrained_layout=True)
    spec["fn"](fig, ds, params or {})
    return fig


def panel_data(name, ds, params=None):
    """The dataframe behind a panel, for the per-panel CSV export."""
    spec = PANELS[name]
    _set_spread(params)
    fn = spec.get("data")
    if fn is None:
        return None
    try:
        return fn(ds, params or {})
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# view: task design
# --------------------------------------------------------------------------- #
def _grating(ax, orientation_deg, n=160, sf=3.0, contrast=1.0):
    """Render the actual stimulus: luminance = 0.5 + 0.5*c*sin(2*pi*sf*x_rot).

    Matches the task's fragment shader, where x_rot = u*cos(ori) + v*sin(ori),
    so 0 deg gives vertical bars and 90 deg horizontal bars.
    """
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, n), np.linspace(-0.5, 0.5, n))
    ang = np.deg2rad(orientation_deg)
    xr = u * np.cos(ang) + v * np.sin(ang)
    img = 0.5 + 0.5 * contrast * np.sin(2 * np.pi * sf * xr)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, origin="lower", extent=(0, 1, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#999"); sp.set_linewidth(0.6)


@panel("stimulus_ladder", "Stimulus set: the discrimination axis", "design",
       figsize=(9.0, 2.6),
       caption="The distractor is fixed at 0 deg (vertical). The rewarded target "
               "rotates toward horizontal; difficulty is the difference between them.")
def stimulus_ladder(fig, ds, params):
    levels = sorted(ds.choice["delta_ori"].dropna().unique())
    if not levels:
        return _empty(fig, "No trials in selection")
    show = levels if len(levels) <= 13 else levels[:: max(1, len(levels) // 13)]
    gs = gridspec.GridSpec(2, len(show) + 1, figure=fig, height_ratios=[1, 0.22],
                           wspace=0.12, hspace=0.05)
    ax0 = fig.add_subplot(gs[0, 0])
    _grating(ax0, 0.0)
    ax0.set_title("distractor\n(fixed 0$\\degree$)", fontsize=8.5, color="#d1495b")
    for i, lv in enumerate(show):
        ax = fig.add_subplot(gs[0, i + 1])
        _grating(ax, lv)
        ax.set_title(f"{int(lv)}$\\degree$", fontsize=8.5)
    axl = fig.add_subplot(gs[1, 1:])
    axl.set_axis_off()
    axl.annotate("", xy=(1, 0.6), xytext=(0, 0.6), xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
    axl.text(0.0, 0.05, "identical to distractor\n(chance by construction)",
             fontsize=8, ha="left", color="#666")
    axl.text(1.0, 0.05, "orthogonal\n(easiest)", fontsize=8, ha="right", color="#666")
    fig.suptitle("Rewarded target rotates from vertical to horizontal against a fixed "
                 "vertical distractor", fontsize=10.5)
    return fig


def data_level_coverage(ds, params):
    c = ds.choice.groupby(["animal", "delta_ori"]).size().rename("n_trials").reset_index()
    return c.pivot(index="animal", columns="delta_ori", values="n_trials")


@panel("level_coverage", "Trials per stimulus level", "design", figsize=(7.6, 3.0),
       caption="Trial counts per animal and orientation difference, in the current "
               "selection. Thin cells make that animal's curve unreliable at that level.")
def level_coverage(fig, ds, params):
    piv = data_level_coverage(ds, params)
    if piv is None or piv.empty:
        return _empty(fig, "No trials in selection")
    ax = fig.add_subplot(111)
    sns.heatmap(piv, annot=True, fmt=".0f", cmap="Blues", ax=ax,
                cbar_kws=dict(label="trials"), linewidths=0.5, linecolor="white",
                annot_kws=dict(fontsize=8))
    ax.set_xlabel("orientation difference (deg)")
    ax.set_ylabel("")
    ax.set_title(f"{int(piv.sum().sum())} choice trials in selection")
    return fig


@panel("arena_geometry", "Arena and choice geometry", "design", figsize=(4.6, 4.6),
       caption="Arena drawn to the task configuration: both choice boxes sit in the "
               "front field of view at the configured half-angle.")
def arena_geometry(fig, ds, params):
    R = float(ds.trials["arena_radius"].dropna().iloc[0]) if len(ds.trials) else 100.0
    half = BOX_HALF_ANGLE_DEG
    ax = fig.add_subplot(111)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(th), R * np.sin(th), color="#333", lw=1.2)
    ax.plot(12 * np.cos(th), 12 * np.sin(th), color="#888", lw=0.9, ls="--")
    ax.text(0, 0, "start", ha="center", va="center", fontsize=8, color="#666")
    r_front = R - BOX_INSET
    for side, sign, color in (("left (box_9)", +1, "#1b6ca8"), ("right (box_3)", -1, "#e58f2a")):
        t = np.deg2rad(90.0 + sign * half)
        bx, by = r_front * np.cos(t), r_front * np.sin(t)
        ax.plot([0, bx], [0, by], color=color, lw=1.0, alpha=0.5)
        ax.scatter([bx], [by], s=160, marker="s", color=color, zorder=3)
        ax.annotate(side, (bx, by), textcoords="offset points",
                    xytext=(0, 12 if sign > 0 else 12), ha="center", fontsize=8.5, color=color)
    ax.annotate(f"$\\pm${half:g}$\\degree$", (0, R * 0.45), ha="center", fontsize=9, color="#444")
    ax.set_aspect("equal")
    ax.set_xlim(-R * 1.15, R * 1.15); ax.set_ylim(-R * 1.15, R * 1.25)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"arena radius {R:g}, centre zone 12")
    return fig


def data_session_overview(ds, params):
    cols = ["animal", "session", "session_index", "date", "duration_min",
            "n_choice_trials", "pct_correct", "n_fall", "n_omission", "median_duration_s", "qc_flags"]
    return ds.sessions[[c for c in cols if c in ds.sessions.columns]]


@panel("session_overview", "Sessions in selection", "design", figsize=(7.6, 3.4),
       caption="Trials and accuracy per session. Marker area is session duration.")
def session_overview(fig, ds, params):
    s = ds.sessions
    if s.empty:
        return _empty(fig, "No sessions in selection")
    colors = ds.colors
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0, 0])
    for a, g in s.groupby("animal"):
        ax.plot(g["session_index"], g["n_choice_trials"], "-o", ms=4,
                color=colors.get(a, "#666"), label=a)
    ax.set_xlabel("session index"); ax.set_ylabel("choice trials")
    ax.legend(title=None)
    ax2 = fig.add_subplot(gs[0, 1])
    for a, g in s.groupby("animal"):
        ax2.scatter(g["session_index"], g["pct_correct"],
                    s=np.clip(g["duration_min"], 5, 90) * 1.6,
                    color=colors.get(a, "#666"), alpha=0.75, edgecolor="white", lw=0.5)
    ax2.axhline(50, color="#999", ls=":", lw=0.9)
    ax2.set_xlabel("session index"); ax2.set_ylabel("% correct")
    ax2.set_ylim(40, 100)
    return fig


# --------------------------------------------------------------------------- #
# view: analysis / psychometric
# --------------------------------------------------------------------------- #
def data_psychometric_per_animal(ds, params):
    c = ad.level_counts(ds.choice, by=["animal"])
    if not c.empty:
        c["sem"] = binom_spread(c["p"], c["n"])   # honour the SEM/SD setting
    return c


@panel("psychometric_per_animal", "Psychometric curve per animal", "analysis",
       section="psychometric", figsize=(9.0, 3.4),
       caption="Proportion correct against orientation difference, binomial {err}. "
               "Curve is a logistic with guess rate fixed at 0.5 and free lapse. "
               "Dashed line marks the fitted 75% threshold.")
def psychometric_per_animal(fig, ds, params):
    animals = ds.animals
    if not animals:
        return _empty(fig, "No trials in selection")
    counts = data_psychometric_per_animal(ds, params)
    fits = ds.fits(by=("animal",), n_boot=params.get("n_boot", 0),
                   family=params.get("family"))
    colors = ds.colors
    gs = gridspec.GridSpec(1, len(animals), figure=fig)
    xx = np.linspace(0, max(counts["delta_ori"].max(), 1), 200)
    for i, a in enumerate(animals):
        ax = fig.add_subplot(gs[0, i])
        c = counts[counts["animal"] == a]
        ax.errorbar(c["delta_ori"], c["p"], yerr=c["sem"], fmt="o", ms=4.5,
                    color=colors[a], ecolor=colors[a], elinewidth=1, capsize=0, alpha=0.9)
        f = fits[fits["animal"] == a]
        if len(f) and bool(f.iloc[0]["converged"]):
            r = f.iloc[0]
            ax.plot(xx, ad.psychometric(xx, r["mu"], r["slope"], r["lapse"],
                                    params.get("family") or ad.DEFAULT_FAMILY),
                    color=colors[a], lw=1.6)
            if bool(r.get("reliable", True)) and np.isfinite(r["threshold_75"]):
                ax.axvline(r["threshold_75"], color=colors[a], ls="--", lw=0.9, alpha=0.7)
                ax.text(r["threshold_75"], 0.42, f" {r['threshold_75']:.0f}$\\degree$",
                        fontsize=8, color=colors[a])
            if not bool(r.get("reliable", True)):
                ax.text(0.97, 0.06, "fit unreliable", transform=ax.transAxes,
                        ha="right", fontsize=7.5, color="#b00", style="italic")
        ax.axhline(0.5, color="#999", ls=":", lw=0.9)
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"{a}  (n={int(c['n'].sum())})", color=colors[a])
        ax.set_xlabel("orientation difference (deg)")
        ax.set_ylabel("proportion correct" if i == 0 else "")
        if i:
            ax.set_yticklabels([])
    return fig


def group_fit(ds, params=None):
    """Psychometric fit to the trials pooled across the selected animals.

    Pooling rather than averaging per-animal fits: with few animals the pooled
    likelihood is better conditioned. The caveat is that pooling animals with
    different thresholds shallows the slope, so the pooled slope is a lower
    bound on the typical individual slope.
    """
    counts = ad.level_counts(ds.choice)
    if counts.empty:
        return None
    return ad.fit_psychometric(counts["delta_ori"], counts["k"], counts["n"],
                               family=(params or {}).get("family")
                                      or ad.DEFAULT_FAMILY)


def data_psychometric_group(ds, params):
    per = data_psychometric_per_animal(ds, params)
    grp = per.groupby("delta_ori")["p"].agg(mean="mean", sem=spread, n_animals="size")
    return grp.reset_index()


@panel("psychometric_group", "Group psychometric", "analysis", section="psychometric",
       figsize=(6.2, 4.8),
       caption="Individual animals in colour; black points are the mean +/- {err} "
               "across animals (not across trials). The black curve is a single "
               "fit to all selected trials pooled. The curve can sit above the "
               "black points because pooling weights by trial count, so it follows "
               "the heavily sampled animals, whereas the points weight each animal "
               "equally. Pooling animals of differing threshold also shallows the "
               "slope, making the pooled beta a lower bound on the typical "
               "individual slope.")
def psychometric_group(fig, ds, params):
    per = ad.level_counts(ds.choice, by=["animal"])
    if per.empty:
        return _empty(fig, "No trials in selection")
    grp = data_psychometric_group(ds, params)
    colors = ds.colors
    ax = fig.add_subplot(111)
    for a, g in per.groupby("animal"):
        ax.plot(g["delta_ori"], g["p"], "-o", ms=3.5, lw=1.0, alpha=0.55,
                color=colors[a], label=a)
    ax.errorbar(grp["delta_ori"], grp["mean"], yerr=grp["sem"], fmt="o", ms=5.5,
                lw=0, color=GROUP_COLOR, ecolor=GROUP_COLOR, capsize=0,
                label=f"mean $\\pm$ {spread_label()} across animals "
                      f"(n={int(grp['n_animals'].max())})",
                zorder=5)

    f = group_fit(ds, params)
    if f and f["converged"]:
        xx = np.linspace(0, max(per["delta_ori"].max(), 1), 200)
        ax.plot(xx, ad.psychometric(xx, f["mu"], f["slope"], f["lapse"], f["family"]),
                color=GROUP_COLOR, lw=2.0, zorder=4, label="pooled fit")
        # The equation follows the chosen family. Written inline rather than
        # with \dfrac: a tall fraction glyph collides with the line beneath it.
        spec = ad.FAMILIES[f["family"]]
        eq = f"{spec['label']}:  {spec['equation']}"
        txt = (f"{eq}\n"
               f"$\\mu$ = {f['mu']:.1f}$\\degree$ ({spec['mu_label']})\n"
               f"$\\beta$ = {f['slope']:.3f} {spec['slope_label']}\n"
               f"$\\lambda$ = {f['lapse']:.3f}\n"
               f"75% threshold = {f['threshold_75']:.1f}$\\degree$\n"
               f"n = {int(f['n_trials'])} trials, {int(f['n_levels'])} levels")
        ax.text(0.975, 0.045, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                          edgecolor="#ddd", alpha=0.92))
        if not f["reliable"]:
            ax.text(0.5, 0.02, f"pooled fit flagged: {f['fit_flags']}",
                    transform=ax.transAxes, ha="center", fontsize=7.5,
                    color="#b00", style="italic")
    ax.axhline(0.5, color="#999", ls=":", lw=0.9)
    ax.set_ylim(0.3, 1.02)
    ax.set_xlabel("orientation difference (deg)")
    ax.set_ylabel("proportion correct")
    ax.legend(loc="upper left")
    return fig


def data_side_bias(ds, params):
    c = ds.choice.copy()
    g = c.groupby(["animal", "signed_delta"])["chose_right"].agg(n="size", p="mean").reset_index()
    g["sem"] = binom_spread(g["p"], g["n"])
    return g


@panel("side_bias", "Side bias", "analysis", section="psychometric", figsize=(5.4, 4.2),
       caption="Probability of choosing the right box against signed orientation "
               "difference (positive = target on the right). A vertical offset at "
               "zero is a side bias; chance is 0.5.")
def side_bias(fig, ds, params):
    g = data_side_bias(ds, params)
    if g.empty:
        return _empty(fig, "No trials in selection")
    colors = ds.colors
    ax = fig.add_subplot(111)
    for a, gg in g.groupby("animal"):
        gg = gg.sort_values("signed_delta")
        ax.errorbar(gg["signed_delta"], gg["p"], yerr=gg["sem"], fmt="-o", ms=3.5,
                    lw=1.1, color=colors[a], ecolor=colors[a], capsize=0, label=a, alpha=0.85)
    ax.axhline(0.5, color="#999", ls=":", lw=0.9)
    ax.axvline(0.0, color="#999", ls=":", lw=0.9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("signed orientation difference (deg)\n$-$ target left   |   target right $+$")
    ax.set_ylabel("P(chose right)")
    ax.legend()
    return fig


def data_threshold_by_session(ds, params):
    f = ds.fits(by=("animal", "session_index"), n_boot=0,
                family=(params or {}).get("family"))
    want = ["animal", "session_index", "mu", "threshold_75", "lapse", "slope",
            "n_trials", "reliable", "fit_flags"]
    if f.empty:
        return pd.DataFrame(columns=want)
    return f[want]


@panel("threshold_by_session", "Learning: threshold across sessions", "analysis",
       section="psychometric", figsize=(6.6, 3.6),
       caption="Per-session 75% threshold. Open markers are fits flagged "
               "unreliable (too few trials or a non-identifiable curve) and should "
               "not be read as estimates.")
def threshold_by_session(fig, ds, params):
    f = data_threshold_by_session(ds, params)
    if f is None or f.empty:
        return _empty(fig, "No fits in selection")
    colors = ds.colors
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    for a, g in f.groupby("animal"):
        g = g.sort_values("session_index")
        ok = g[g["reliable"]]
        ax.plot(g["session_index"], g["threshold_75"], "-", lw=1.0,
                color=colors.get(a, "#666"), alpha=0.5)
        ax.scatter(ok["session_index"], ok["threshold_75"], s=30,
                   color=colors.get(a, "#666"), label=a, zorder=3)
        bad = g[~g["reliable"]]
        ax.scatter(bad["session_index"], bad["threshold_75"], s=30, facecolors="none",
                   edgecolors=colors.get(a, "#666"), zorder=3)
    ax.set_xlabel("session index"); ax.set_ylabel("75% threshold (deg)")
    ax.legend()
    ax2 = fig.add_subplot(gs[0, 1])
    for a, g in f.groupby("animal"):
        g = g.sort_values("session_index")
        ax2.plot(g["session_index"], g["lapse"], "-o", ms=3.5, lw=1.0,
                 color=colors.get(a, "#666"), alpha=0.8)
    ax2.set_xlabel("session index"); ax2.set_ylabel("lapse rate")
    ax2.set_ylim(0, None)
    return fig


# --------------------------------------------------------------------------- #
# view: analysis / learning
# --------------------------------------------------------------------------- #
LEARNING_METRICS = [
    ("pct_correct_easy", "% correct at 70-90$\\degree$", (40, 102)),
    ("lapse", "lapse rate", (0, None)),
    ("n_choice_trials", "choice trials", (0, None)),
    ("median_duration_s", "time per trial (s)", (0, None)),
]


def data_learning(ds, params):
    """Per-session learning metrics, with the per-session lapse rate joined on."""
    cols = ["animal", "session", "session_index", "date", "n_choice_trials",
            "pct_correct", "pct_correct_easy", "n_easy_trials",
            "median_duration_s", "median_window_s"]
    s = ds.sessions[[c for c in cols if c in ds.sessions.columns]].copy()
    fits = ds.fits(by=("animal", "session_index"), n_boot=0)
    if not fits.empty:
        s = s.merge(fits[["animal", "session_index", "lapse", "threshold_75",
                          "mu", "reliable"]],
                    on=["animal", "session_index"], how="left")
    return s


@panel("learning", "Learning across sessions", "analysis", section="learning",
       figsize=(9.0, 5.4),
       caption="One line per animal across sessions. Accuracy at 70-90 deg is a "
               "ceiling measure (does the animal know the rule and stay engaged), "
               "largely separable from threshold sensitivity. Lapse comes from the "
               "per-session psychometric fit; open markers are unreliable fits. "
               "Error bars on accuracy are binomial {err}.")
def learning(fig, ds, params):
    s = data_learning(ds, params)
    if s is None or s.empty:
        return _empty(fig, "No sessions in selection")
    colors = ds.colors
    gs = gridspec.GridSpec(2, 2, figure=fig)
    for i, (col, label, ylim) in enumerate(LEARNING_METRICS):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        if col not in s.columns:
            ax.set_axis_off()
            continue
        for a, g in s.groupby("animal"):
            g = g.sort_values("session_index")
            ax.plot(g["session_index"], g[col], "-", lw=1.1,
                    color=colors.get(a, "#666"), alpha=0.75)
            if col == "pct_correct_easy" and "n_easy_trials" in g:
                p = g[col] / 100.0
                yerr = 100 * binom_spread(p, g["n_easy_trials"])
                ax.errorbar(g["session_index"], g[col], yerr=yerr, fmt="o", ms=4,
                            color=colors.get(a, "#666"), ecolor=colors.get(a, "#666"),
                            capsize=0, elinewidth=0.9, label=a)
            elif col == "lapse" and "reliable" in g:
                good = g[g["reliable"].fillna(False).astype(bool)]
                bad = g[~g["reliable"].fillna(False).astype(bool)]
                ax.scatter(good["session_index"], good[col], s=28,
                           color=colors.get(a, "#666"), label=a)
                ax.scatter(bad["session_index"], bad[col], s=28, facecolors="none",
                           edgecolors=colors.get(a, "#666"))
            else:
                ax.plot(g["session_index"], g[col], "o", ms=4,
                        color=colors.get(a, "#666"), label=a)
        if col == "pct_correct_easy":
            ax.axhline(50, color="#999", ls=":", lw=0.9)
        ax.set_xlabel("session index")
        ax.set_ylabel(label)
        if ylim:
            ax.set_ylim(*ylim)
        if i == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    return fig


# --------------------------------------------------------------------------- #
# view: analysis / within-trial dynamics
# --------------------------------------------------------------------------- #
def data_heading_trace(ds, params):
    tc = ds.traces
    if tc is None or tc.empty:
        return pd.DataFrame()
    long = tc.melt(id_vars=["frac", "outcome"], value_vars=["err_chosen", "err_other"],
                   var_name="target", value_name="abs_err")
    return (long.groupby(["outcome", "target", "frac"])["abs_err"]
            .agg(mean="mean", sem=spread, n="size").reset_index())


@panel("heading_trace", "Heading toward the chosen box over the trial", "analysis",
       section="within", figsize=(8.6, 3.6),
       caption="Absolute heading error toward the box the animal chose (solid) and "
               "toward the box it rejected (dashed), against normalised within-trial "
               "time. Mean +/- {err} across trials. Early separation of the two traces "
               "means an early commitment; late separation means a late switch.")
def heading_trace(fig, ds, params):
    g = data_heading_trace(ds, params)
    if g.empty:
        return _empty(fig, "Commitment traces unavailable in selection")
    gs = gridspec.GridSpec(1, 2, figure=fig)
    for j, outcome in enumerate(("correct", "incorrect")):
        ax = fig.add_subplot(gs[0, j])
        for target, ls in (("err_chosen", "-"), ("err_other", "--")):
            gg = g[(g["outcome"] == outcome) & (g["target"] == target)].sort_values("frac")
            if gg.empty:
                continue
            ax.plot(gg["frac"], gg["mean"], ls, lw=1.6,
                    color=OUTCOME_COLORS[outcome],
                    label="chosen box" if target == "err_chosen" else "rejected box")
            ax.fill_between(gg["frac"], gg["mean"] - gg["sem"], gg["mean"] + gg["sem"],
                            color=OUTCOME_COLORS[outcome], alpha=0.2, lw=0)
        ax.set_xlabel("normalised time within trial")
        ax.set_ylabel("|heading error| (deg)" if j == 0 else "")
        ax.set_title(outcome, color=OUTCOME_COLORS[outcome])
        if j == 0:
            ax.legend()
    ymax = max(ax.get_ylim()[1] for ax in fig.axes)
    for ax in fig.axes:
        ax.set_ylim(0, ymax)
    return fig


def data_commitment(ds, params):
    cm = ds.commit
    if cm is None or cm.empty:
        return pd.DataFrame()
    return (cm.groupby(["delta_ori", "outcome"])
            .agg(n=("t_commit_align", "size"),
                 t_commit=("t_commit_align", "mean"),
                 t_commit_sem=("t_commit_align", spread),
                 frac_commit=("frac_commit_align", "mean"),
                 reversals=("n_reversals", "mean"),
                 reversals_sem=("n_reversals", spread),
                 pct_single_run=("n_reversals", lambda s: 100 * (s == 0).mean()))
            .reset_index())


@panel("commitment", "Inferred decision point", "analysis", section="within",
       figsize=(9.0, 3.6),
       caption="Commitment time is the earliest moment after which the heading "
               "stays closer to the chosen box than the rejected one for the rest "
               "of the approach, on a 5-sample-smoothed decision variable with a "
               "5 deg dead zone. Arrival within 15 units of the box is excluded, "
               "where bearing becomes geometrically unstable. Mean +/- {err}.")
def commitment(fig, ds, params):
    g = data_commitment(ds, params)
    if g.empty:
        return _empty(fig, "Commitment metrics unavailable in selection")
    gs = gridspec.GridSpec(1, 3, figure=fig)
    specs = [("t_commit", "t_commit_sem", "commitment time (s)"),
             ("reversals", "reversals_sem", "changes of mind per trial"),
             ("pct_single_run", None, "% trials with a single committed run")]
    for i, (col, semcol, label) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        for outcome in ("correct", "incorrect"):
            gg = g[g["outcome"] == outcome].sort_values("delta_ori")
            yerr = gg[semcol] if semcol else None
            ax.errorbar(gg["delta_ori"], gg[col], yerr=yerr, fmt="-o", ms=4, lw=1.3,
                        color=OUTCOME_COLORS[outcome], ecolor=OUTCOME_COLORS[outcome],
                        capsize=0, label=outcome)
        ax.set_xlabel("orientation difference (deg)")
        ax.set_ylabel(label)
        # Latencies, counts and percentages all start at 0; never let a clipped
        # axis exaggerate the size of a difference.
        ax.set_ylim(0, ax.get_ylim()[1])
        if i == 0:
            ax.legend()
    return fig


MIND_STYLE = {"single run": dict(ls="-", marker="o"),
              "changed mind": dict(ls="--", marker="s")}


def data_commitment_by_mind(ds, params):
    cm = ds.commit
    if cm is None or cm.empty:
        return pd.DataFrame()
    cm = cm.copy()
    cm["mind"] = np.where(cm["n_reversals"] == 0, "single run", "changed mind")
    return (cm.groupby(["outcome", "mind", "delta_ori"])["t_commit_align"]
            .agg(n="size", mean="mean", sem=spread, median="median").reset_index())


@panel("commitment_by_mind", "Decision time by difficulty, outcome and change of mind",
       "analysis", section="within", figsize=(8.8, 3.8),
       caption="Commitment time against orientation difference, split by outcome "
               "(panels) and by whether the animal changed its mind en route "
               "(line style). Mean +/- {err} across trials; marker area is trial "
               "count, because the incorrect/changed-mind cells are thin - as few "
               "as 8 trials at some levels against ~160 for correct/single-run - "
               "so points there should not be read as firmly as the rest. "
               "Note the offset between the two line styles is largely "
               "definitional: a trial that changed its mind must commit after its "
               "reversal, so it cannot commit early. The informative comparison is "
               "the trend across difficulty within each line, and correct against "
               "incorrect within the same line style.")
def commitment_by_mind(fig, ds, params):
    g = data_commitment_by_mind(ds, params)
    if g.empty:
        return _empty(fig, "Commitment metrics unavailable in selection")
    nmax = max(float(g["n"].max()), 1.0)
    gs = gridspec.GridSpec(1, 2, figure=fig)
    axes = []
    for j, outcome in enumerate(("correct", "incorrect")):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        for mind, style in MIND_STYLE.items():
            gg = g[(g["outcome"] == outcome) & (g["mind"] == mind)].sort_values("delta_ori")
            if gg.empty:
                continue
            ax.errorbar(gg["delta_ori"], gg["mean"], yerr=gg["sem"],
                        ls=style["ls"], lw=1.3, marker="none",
                        color=OUTCOME_COLORS[outcome], ecolor=OUTCOME_COLORS[outcome],
                        capsize=0, elinewidth=0.9, zorder=2,
                        label=f"{mind} (n={int(gg['n'].sum())})")
            ax.scatter(gg["delta_ori"], gg["mean"],
                       s=18 + 70 * (gg["n"] / nmax), marker=style["marker"],
                       facecolors=OUTCOME_COLORS[outcome] if mind == "single run" else "white",
                       edgecolors=OUTCOME_COLORS[outcome], lw=1.1, zorder=3)
        ax.set_xlabel("orientation difference (deg)")
        ax.set_ylabel("commitment time (s)" if j == 0 else "")
        ax.set_title(outcome, color=OUTCOME_COLORS[outcome])
        ax.legend(fontsize=8)
    # Commitment time is a latency: the axis starts at 0 regardless of the data,
    # so the size of an effect is never visually exaggerated by a clipped axis.
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(0, hi)
    for a in axes[1:]:
        a.set_yticklabels([])
    return fig


def data_commit_positions(ds, params):
    cm = ds.commit
    if cm is None or cm.empty or "commit_x" not in cm.columns:
        return pd.DataFrame()
    cm = cm[np.isfinite(cm["commit_x"]) & np.isfinite(cm["commit_y"])].copy()
    cm["mind"] = np.where(cm["n_reversals"] == 0, "single run", "changed mind")
    return cm[["animal", "session", "trial_id", "outcome", "mind", "delta_ori",
               "t_commit_align", "commit_x", "commit_y", "commit_r"]]


@panel("commit_position_maps", "Where in the arena commitment happens", "analysis",
       section="within", figsize=(8.8, 5.0),
       caption="Density of the animal's position at the inferred commitment "
               "moment, on the full arena. Each panel is normalised to its own "
               "maximum, so compare shapes rather than absolute shading; n differs "
               "a lot between panels. The cross marks the median commitment "
               "position and the dashed circle the centre zone. Commitment inside "
               "the centre zone means the animal had chosen before setting off; "
               "commitment far out means it steered late. n counts trials with a "
               "detected commitment, so it is slightly below the trial counts in "
               "the timing panels, where trials without one still contribute.")
def commit_position_maps(fig, ds, params):
    d = data_commit_positions(ds, params)
    if d.empty:
        return _empty(fig, "Commit positions unavailable "
                           "(rebuild commitment.parquet to add them)")
    R = float(ds.trials["arena_radius"].dropna().iloc[0]) if len(ds.trials) else 100.0
    bins = int(params.get("map_bins", 28))
    edges = np.linspace(-R, R, bins + 1)
    cx = 0.5 * (edges[:-1] + edges[1:])
    XX, YY = np.meshgrid(cx, cx)
    outside = np.hypot(XX, YY) > R

    gs = gridspec.GridSpec(2, 2, figure=fig)
    th = np.linspace(0, 2 * np.pi, 300)
    for i, outcome in enumerate(("correct", "incorrect")):
        for j, mind in enumerate(("single run", "changed mind")):
            ax = fig.add_subplot(gs[i, j])
            sub = d[(d["outcome"] == outcome) & (d["mind"] == mind)]
            if len(sub) >= 5:
                H, _, _ = np.histogram2d(sub["commit_x"], sub["commit_y"],
                                         bins=[edges, edges])
                H = H.T                      # histogram2d returns x along axis 0
                H = np.ma.masked_array(H, mask=outside | (H == 0))
                # Light-to-dark: a sparse cell must not look dense. With a
                # dark-low colormap the single-trial cells read as hot spots.
                cmap = mpl.colormaps["magma_r"].copy()
                cmap.set_bad("#fbfbfc")        # empty cells read as background
                ax.pcolormesh(edges, edges, H / H.max(), cmap=cmap,
                              vmin=0, vmax=1, shading="flat")
                ax.plot(sub["commit_x"].median(), sub["commit_y"].median(),
                        "+", color="#1b6ca8", ms=12, mew=2.4, zorder=5)
            else:
                ax.text(0, 0, f"n = {len(sub)}", ha="center", va="center",
                        fontsize=9, color="#666")
            ax.plot(R * np.cos(th), R * np.sin(th), color="#333", lw=1.0)
            ax.plot(12 * np.cos(th), 12 * np.sin(th), color="#888", lw=0.8, ls="--")
            r_front = R - BOX_INSET
            for sign in (+1, -1):
                t = np.deg2rad(90.0 + sign * BOX_HALF_ANGLE_DEG)
                ax.scatter([r_front * np.cos(t)], [r_front * np.sin(t)], s=70,
                           marker="s", facecolors="none", edgecolors="#444", zorder=4)
            ax.set_aspect("equal")
            ax.set_xlim(-R * 1.08, R * 1.08)
            ax.set_ylim(-R * 1.08, R * 1.08)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{outcome}, {mind}  (n={len(sub)})", fontsize=9,
                         color=OUTCOME_COLORS[outcome])
    return fig


def data_commit_radius(ds, params):
    d = data_commit_positions(ds, params)
    if d.empty:
        return pd.DataFrame()
    return (d.groupby(["outcome", "mind", "delta_ori"])["commit_r"]
            .agg(n="size", mean="mean", sem=spread, median="median").reset_index())


@panel("commit_radius", "Distance from centre at commitment", "analysis",
       section="within", figsize=(8.8, 3.8),
       caption="The same four splits collapsed to one number per trial: how far "
               "from the arena centre the animal had travelled when it committed. "
               "Mean +/- {err}; marker area is trial count. Reads the heatmaps "
               "quantitatively and against difficulty.")
def commit_radius(fig, ds, params):
    g = data_commit_radius(ds, params)
    if g.empty:
        return _empty(fig, "Commit positions unavailable in selection")
    nmax = max(float(g["n"].max()), 1.0)
    gs = gridspec.GridSpec(1, 2, figure=fig)
    axes = []
    for j, outcome in enumerate(("correct", "incorrect")):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        for mind, style in MIND_STYLE.items():
            gg = g[(g["outcome"] == outcome) & (g["mind"] == mind)].sort_values("delta_ori")
            if gg.empty:
                continue
            ax.errorbar(gg["delta_ori"], gg["mean"], yerr=gg["sem"], ls=style["ls"],
                        lw=1.3, marker="none", color=OUTCOME_COLORS[outcome],
                        ecolor=OUTCOME_COLORS[outcome], capsize=0, elinewidth=0.9,
                        label=f"{mind} (n={int(gg['n'].sum())})", zorder=2)
            ax.scatter(gg["delta_ori"], gg["mean"], s=18 + 70 * (gg["n"] / nmax),
                       marker=style["marker"],
                       facecolors=OUTCOME_COLORS[outcome] if mind == "single run" else "white",
                       edgecolors=OUTCOME_COLORS[outcome], lw=1.1, zorder=3)
        ax.axhline(12, color="#888", ls=":", lw=0.9)
        ax.set_xlabel("orientation difference (deg)")
        ax.set_ylabel("distance from centre at commitment" if j == 0 else "")
        ax.set_title(outcome, color=OUTCOME_COLORS[outcome])
        ax.legend(fontsize=8)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(0, hi)
    for a in axes[1:]:
        a.set_yticklabels([])
    return fig


@panel("decision_examples", "Example trials with the detected decision point",
       "analysis", section="within", figsize=(9.0, 3.2),
       caption="Decision variable over the approach: positive means the heading "
               "points more directly at the box eventually chosen. Shaded band is "
               "the dead zone; the vertical line is the inferred commitment.")
def decision_examples(fig, ds, params):
    cm, tc = ds.commit, ds.traces
    if cm is None or cm.empty or tc is None or tc.empty:
        return _empty(fig, "Commitment traces unavailable in selection")
    rng = np.random.default_rng(int(params.get("example_seed", 1)))
    picks = []
    for outcome in ("correct", "incorrect"):
        pool = cm[(cm["outcome"] == outcome) & cm["t_commit_align"].notna()]
        early = pool[pool["n_reversals"] == 0]
        late = pool[pool["n_reversals"] >= 1]
        for sub, tag in ((early, "single run"), (late, "changed mind")):
            if len(sub):
                picks.append((sub.iloc[rng.integers(0, len(sub))], outcome, tag))
    if not picks:
        return _empty(fig, "No trials with a detected commitment")
    gs = gridspec.GridSpec(1, len(picks), figure=fig)
    dz = float(params.get("dead_zone_deg", 5.0))
    for i, (r, outcome, tag) in enumerate(picks):
        ax = fig.add_subplot(gs[0, i])
        tr = tc[(tc["animal"] == r["animal"]) & (tc["session"] == r["session"])
                & (tc["trial_id"] == r["trial_id"])].sort_values("frac")
        if tr.empty:
            ax.set_axis_off()
            continue
        dv = tr["err_other"] - tr["err_chosen"]
        ax.axhspan(-dz, dz, color="#eee", lw=0)
        ax.axhline(0, color="#999", lw=0.8)
        ax.plot(tr["t_rel"], dv, color=OUTCOME_COLORS[outcome], lw=1.5)
        if np.isfinite(r["t_commit_align"]):
            ax.axvline(r["t_commit_align"], color="#333", ls="--", lw=1.0)
        ax.set_xlabel("time from stimulus onset (s)")
        ax.set_ylabel("toward chosen $-$ toward rejected (deg)" if i == 0 else "")
        ax.set_title(f"{outcome}, {tag}\n{r['animal']} trial {int(r['trial_id'])}, "
                     f"$\\Delta$={int(r['delta_ori'])}$\\degree$",
                     fontsize=8.5, color=OUTCOME_COLORS[outcome])
    return fig


# --------------------------------------------------------------------------- #
# view: analysis / trial duration
# --------------------------------------------------------------------------- #
def _rt(ds, params):
    c = ds.choice.copy()
    c = c[np.isfinite(c["trial_duration_s"])]
    trim = float(params.get("dur_trim_pct", 1.0))
    if trim > 0 and len(c):
        lo, hi = np.percentile(c["trial_duration_s"], [trim, 100 - trim])
        c = c[(c["trial_duration_s"] >= lo) & (c["trial_duration_s"] <= hi)]
    return c


def data_duration_distributions(ds, params):
    c = _rt(ds, params)
    return c.groupby(["animal", "outcome"])["trial_duration_s"].describe().reset_index()


@panel("duration_distributions", "Trial duration distributions", "analysis", section="duration",
       figsize=(8.6, 3.2),
       caption="Stimulus-to-reward latency on correct vs incorrect trials. "
               "Trimmed at the configured percentile; vertical lines are medians.")
def duration_distributions(fig, ds, params):
    c = _rt(ds, params)
    if c.empty:
        return _empty(fig, "No trial durations in selection")
    animals = ds.animals
    logx = bool(params.get("dur_log", False))
    gs = gridspec.GridSpec(1, len(animals), figure=fig)
    for i, a in enumerate(animals):
        ax = fig.add_subplot(gs[0, i])
        g = c[c["animal"] == a]
        for outcome in ("correct", "incorrect"):
            x = g.loc[g["outcome"] == outcome, "trial_duration_s"].to_numpy()
            if len(x) < 5:
                continue
            sns.kdeplot(x=np.log10(x) if logx else x, ax=ax, fill=True, alpha=0.3,
                        color=OUTCOME_COLORS[outcome], lw=1.3, label=outcome,
                        warn_singular=False)
            ax.axvline(np.median(np.log10(x) if logx else x),
                       color=OUTCOME_COLORS[outcome], ls="--", lw=1.0)
        ax.set_title(a, color=ds.colors[a])
        ax.set_xlabel("log$_{10}$ duration (s)" if logx else "duration (s)")
        ax.set_ylabel("density" if i == 0 else "")
        if i == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend()
    return fig


def data_duration_vs_difficulty(ds, params):
    c = _rt(ds, params)
    g = c.groupby(["delta_ori", "outcome"])["trial_duration_s"].agg(n="size", mean="mean", sem=spread)
    return g.reset_index()


@panel("duration_vs_difficulty", "Trial duration vs difficulty", "analysis", section="duration",
       figsize=(5.6, 4.0),
       caption="Mean +/- {err} across trials. Slowing near threshold and faster "
               "responses at easy levels is the expected signature.")
def duration_vs_difficulty(fig, ds, params):
    g = data_duration_vs_difficulty(ds, params)
    if g.empty:
        return _empty(fig, "No trial durations in selection")
    ax = fig.add_subplot(111)
    for outcome in ("correct", "incorrect"):
        gg = g[g["outcome"] == outcome].sort_values("delta_ori")
        ax.errorbar(gg["delta_ori"], gg["mean"], yerr=gg["sem"], fmt="-o", ms=4.5,
                    lw=1.4, color=OUTCOME_COLORS[outcome], ecolor=OUTCOME_COLORS[outcome],
                    capsize=0, label=outcome)
    ax.set_xlabel("orientation difference (deg)")
    ax.set_ylabel("trial duration (s)")
    ax.legend()
    return fig


def data_duration_across_sessions(ds, params):
    c = _rt(ds, params)
    return c.groupby(["animal", "session_index"])["trial_duration_s"].agg(
        n="size", mean="mean", sem=spread, median="median").reset_index()


@panel("duration_across_sessions", "Trial duration across and within sessions", "analysis",
       section="duration", figsize=(8.0, 3.4),
       caption="Left: mean +/- {err} per session. Right: duration against trial number "
               "within a session, binned, pooled over sessions - a rising "
               "trace indicates satiety or disengagement.")
def duration_across_sessions(fig, ds, params):
    c = _rt(ds, params)
    if c.empty:
        return _empty(fig, "No trial durations in selection")
    colors = ds.colors
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    per = data_duration_across_sessions(ds, params)
    for a, g in per.groupby("animal"):
        g = g.sort_values("session_index")
        ax.errorbar(g["session_index"], g["mean"], yerr=g["sem"], fmt="-o", ms=4,
                    lw=1.1, color=colors[a], ecolor=colors[a], capsize=0, label=a)
    ax.set_xlabel("session index"); ax.set_ylabel("trial duration (s)"); ax.legend()
    ax2 = fig.add_subplot(gs[0, 1])
    nb = 12
    for a, g in c.groupby("animal"):
        g = g.copy()
        g["frac"] = g.groupby("session")["trial_in_session"].transform(
            lambda s: (s - s.min()) / max(s.max() - s.min(), 1))
        g["bin"] = np.clip((g["frac"] * nb).astype(int), 0, nb - 1)
        b = g.groupby("bin")["trial_duration_s"].agg(mean="mean", sem=spread).reset_index()
        ax2.errorbar((b["bin"] + 0.5) / nb, b["mean"], yerr=b["sem"], fmt="-o", ms=3.5,
                     lw=1.1, color=colors[a], ecolor=colors[a], capsize=0)
    ax2.set_xlabel("position within session"); ax2.set_ylabel("trial duration (s)")
    return fig


# --------------------------------------------------------------------------- #
# view: analysis / path
# --------------------------------------------------------------------------- #
def _difficulty_bin(delta):
    """Three difficulty bands over the orientation-difference axis."""
    return pd.cut(delta, bins=[-0.1, 15, 40, 90.1],
                  labels=["hard (0-15$\\degree$)", "medium (20-40$\\degree$)",
                          "easy (50-90$\\degree$)"])


def data_path_metrics_vs_difficulty(ds, params):
    p = ds.paths[ds.paths["is_choice"]]
    metric = params.get("path_metric", "straightness")
    g = p.groupby(["delta_ori", "outcome"])[metric].agg(n="size", mean="mean", sem=spread)
    return g.reset_index().assign(metric=metric)


@panel("path_metrics_vs_difficulty", "Path metrics vs difficulty", "analysis",
       section="path", figsize=(8.8, 3.4),
       caption="Mean +/- {err} across trials, split by outcome. Straightness is net "
               "displacement over path length: 1.0 is a direct run to the box.")
def path_metrics_vs_difficulty(fig, ds, params):
    p = ds.paths[ds.paths["is_choice"]]
    if p.empty:
        return _empty(fig, "No trajectories in selection")
    metrics = params.get("path_metrics",
                         ["straightness", "path_length", "t_leave_center"])
    labels = {"straightness": "straightness", "path_length": "path length",
              "t_leave_center": "time to leave centre (s)",
              "total_turning": "cumulative turning (deg)",
              "mean_speed": "mean speed", "peak_speed": "peak speed (95th pct)"}
    gs = gridspec.GridSpec(1, len(metrics), figure=fig)
    for i, m in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        for outcome in ("correct", "incorrect"):
            g = (p[p["outcome"] == outcome].groupby("delta_ori")[m]
                 .agg(mean="mean", sem=spread).reset_index())
            ax.errorbar(g["delta_ori"], g["mean"], yerr=g["sem"], fmt="-o", ms=4,
                        lw=1.3, color=OUTCOME_COLORS[outcome],
                        ecolor=OUTCOME_COLORS[outcome], capsize=0, label=outcome)
        ax.set_xlabel("orientation difference (deg)")
        ax.set_ylabel(labels.get(m, m))
        if i == 0:
            ax.legend()
    return fig


def data_path_by_outcome(ds, params):
    p = ds.paths[ds.paths["is_choice"]].copy()
    p["difficulty"] = _difficulty_bin(p["delta_ori"])
    return (p.groupby(["animal", "difficulty", "outcome"], observed=True)["straightness"]
            .agg(n="size", mean="mean", sem=spread).reset_index())


@panel("path_by_outcome", "Straightness by outcome and difficulty", "analysis",
       section="path", figsize=(6.4, 3.8),
       caption="Mean +/- {err} across trials. If error trials are less direct at "
               "every difficulty, the errors carry a motor signature rather than "
               "being purely perceptual.")
def path_by_outcome(fig, ds, params):
    g = data_path_by_outcome(ds, params)
    if g.empty:
        return _empty(fig, "No trajectories in selection")
    ax = fig.add_subplot(111)
    order = [c for c in g["difficulty"].cat.categories if c in set(g["difficulty"])]
    width = 0.36
    for j, outcome in enumerate(("correct", "incorrect")):
        gg = (g[g["outcome"] == outcome].groupby("difficulty", observed=True)
              .apply(lambda d: pd.Series({
                  "mean": np.average(d["mean"], weights=d["n"]),
                  "sem": spread(d["mean"])}), include_groups=False).reindex(order))
        xs = np.arange(len(order)) + (j - 0.5) * width
        ax.bar(xs, gg["mean"], width=width, color=OUTCOME_COLORS[outcome],
               alpha=0.85, label=outcome)
        ax.errorbar(xs, gg["mean"], yerr=gg["sem"], fmt="none", ecolor="#333",
                    elinewidth=1, capsize=3)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("straightness")
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    return fig


@panel("trajectory_bundles", "Trajectories to correct and incorrect choices",
       "analysis", section="path", figsize=(8.4, 4.4),
       caption="Single-trial paths from one session, coloured by outcome, with "
               "the mean path overlaid. Choose the session in the sidebar.")
def trajectory_bundles(fig, ds, params):
    sess = params.get("example_session")
    if sess is None:
        if ds.sessions.empty:
            return _empty(fig, "No sessions in selection")
        row = ds.sessions.sort_values("n_choice_trials", ascending=False).iloc[0]
        sess = (row["animal"], row["session"])
    animal, session = sess
    samples = ds.samples(animal, session)
    if samples.empty:
        return _empty(fig, "Sample trace unavailable (no data root)")
    tr = ds.trials[(ds.trials["animal"] == animal) & (ds.trials["session"] == session)]
    R = float(tr["arena_radius"].dropna().iloc[0]) if len(tr) else 100.0
    lut = tr.set_index("trial_id")["outcome"].to_dict()
    max_n = int(params.get("max_trials_drawn", 120))

    gs = gridspec.GridSpec(1, 2, figure=fig)
    for j, outcome in enumerate(("correct", "incorrect")):
        ax = fig.add_subplot(gs[0, j])
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(R * np.cos(th), R * np.sin(th), color="#333", lw=1.0)
        ax.plot(12 * np.cos(th), 12 * np.sin(th), color="#bbb", lw=0.8, ls="--")
        ids = [t for t, o in lut.items() if o == outcome][:max_n]
        xs_all, ys_all = [], []
        for tid in ids:
            g = samples[samples["trial_id"] == tid].sort_values("t")
            if len(g) < 3:
                continue
            n_interp = 60
            f = np.linspace(0, 1, len(g))
            xi = np.interp(np.linspace(0, 1, n_interp), f, g["x"])
            yi = np.interp(np.linspace(0, 1, n_interp), f, g["y"])
            # Draw the downsampled path: visually identical at this scale and it
            # keeps exported SVG files to a workable size.
            ax.plot(xi, yi, color=OUTCOME_COLORS[outcome], lw=0.5, alpha=0.18)
            xs_all.append(xi)
            ys_all.append(yi)
        if xs_all:
            ax.plot(np.mean(xs_all, axis=0), np.mean(ys_all, axis=0),
                    color=OUTCOME_COLORS[outcome], lw=2.4)
        r_front = R - BOX_INSET
        for sign in (+1, -1):
            t = np.deg2rad(90.0 + sign * BOX_HALF_ANGLE_DEG)
            ax.scatter([r_front * np.cos(t)], [r_front * np.sin(t)], s=90, marker="s",
                       facecolors="none", edgecolors="#444", zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-R * 1.1, R * 1.1); ax.set_ylim(-R * 1.1, R * 1.15)
        ax.set_title(f"{outcome}  (n={len(xs_all)} drawn)", color=OUTCOME_COLORS[outcome])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{animal} / {session}", fontsize=10)
    return fig


# --------------------------------------------------------------------------- #
# view: cross-animal comparison
# --------------------------------------------------------------------------- #
def data_animal_comparison(ds, params):
    if ds.choice.empty:
        return pd.DataFrame()
    fits = ds.fits(by=("animal",), n_boot=params.get("n_boot", 0),
                   family=params.get("family"))
    c = ds.choice
    base = c.groupby("animal").agg(
        n_trials=("correct", "size"), pct_correct=("correct", "mean"),
        median_duration_s=("trial_duration_s", "median")).reset_index()
    p = ds.paths[ds.paths["is_choice"]].groupby("animal").agg(
        straightness=("straightness", "mean"),
        path_length=("path_length", "mean")).reset_index()
    out = base.merge(fits, on="animal", how="left").merge(p, on="animal", how="left")
    out["n_sessions"] = out["animal"].map(ds.sessions.groupby("animal").size())
    return out


@panel("animal_comparison", "Per-animal summary", "compare", figsize=(9.0, 3.4),
       caption="Each point is one animal; the black marker is the mean +/- {err} "
               "across animals. Hollow points are fits flagged unreliable.")
def animal_comparison(fig, ds, params):
    t = data_animal_comparison(ds, params)
    if t.empty:
        return _empty(fig, "No animals in selection")
    colors = ds.colors
    specs = [("pct_correct", "proportion correct", None),
             ("threshold_75", "75% threshold (deg)", "reliable"),
             ("lapse", "lapse rate", "reliable"),
             ("median_duration_s", "median trial duration (s)", None),
             ("straightness", "straightness", None)]
    gs = gridspec.GridSpec(1, len(specs), figure=fig)
    for i, (col, label, gate) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        vals = t.copy()
        for _, r in vals.iterrows():
            if not np.isfinite(r.get(col, np.nan)):
                continue
            solid = True if gate is None else bool(r.get(gate, True))
            ax.scatter([0], [r[col]], s=55, color=colors.get(r["animal"], "#666"),
                       facecolors=colors.get(r["animal"], "#666") if solid else "none",
                       edgecolors=colors.get(r["animal"], "#666"), zorder=3,
                       label=r["animal"] if i == 0 else None)
        use = vals[col]
        if gate is not None:
            use = vals.loc[vals[gate].fillna(False).astype(bool), col]
        use = use[np.isfinite(use)]
        if len(use):
            ax.errorbar([0.35], [use.mean()], yerr=[spread(use)], fmt="s", ms=6,
                        color=GROUP_COLOR, ecolor=GROUP_COLOR, capsize=3, zorder=4)
        ax.set_xlim(-0.35, 0.75)
        ax.set_xticks([])
        ax.set_ylabel(label, fontsize=9)
        if i == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower left", fontsize=8)
    return fig


def data_performance_matrix(ds, params):
    c = ds.choice.groupby(["animal", "session_index"])["correct"].mean().reset_index()
    return c.pivot(index="animal", columns="session_index", values="correct")


@panel("performance_matrix", "Performance by animal and session", "compare",
       figsize=(7.6, 2.8),
       caption="Proportion correct per session. Blank cells are sessions that "
               "animal does not have, or that the current filters removed.")
def performance_matrix(fig, ds, params):
    piv = data_performance_matrix(ds, params)
    if piv is None or piv.empty:
        return _empty(fig, "No sessions in selection")
    ax = fig.add_subplot(111)
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5,
                vmin=0.3, vmax=1.0, ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws=dict(label="proportion correct"), annot_kws=dict(fontsize=8))
    ax.set_xlabel("session index"); ax.set_ylabel("")
    return fig


VIEWS = [
    ("design", "Task design"),
    ("analysis", "Analysis"),
    ("compare", "Cross-animal"),
]
SECTIONS = [
    ("design", "design", "Task design"),
    ("analysis", "psychometric", "Psychometric"),
    ("analysis", "learning", "Learning"),
    ("analysis", "duration", "Trial duration"),
    ("analysis", "path", "Path analysis"),
    ("analysis", "within", "Within-trial dynamics"),
    ("compare", "compare", "Cross-animal comparison"),
]
