"""Exploration: trust-focused patterns in the journey data. Saves PNG charts to figures/ and a summary to reports/.

Run from the project root:  python src/03_explore.py
Each chart answers one trust question. Titles state the finding.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/processed/journeys_merged.csv")
events = pd.read_csv(ROOT / "data/raw/intervention_events.csv")

BLUE, RED, GREY = "#1f4e79", "#c0392b", "#7f8c8d"
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})
summary = []


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion (handles small groups)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def rate_table(data, by):
    g = data.groupby(by)["problem_occurred_flag"].agg(["sum", "count"]).rename(
        columns={"sum": "problems", "count": "n"})
    g["rate"] = g.problems / g.n
    ci = [wilson(k, n) for k, n in zip(g.problems, g.n)]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---- 1. Where does it break? Problem rate by weather ----------------------
t = rate_table(df, "weather").sort_values("rate")
summary.append("1. Problem rate by weather\n" + t.round(3).to_string())
fig, ax = plt.subplots(figsize=(9, 5))
colors = [BLUE if r < 0.25 else RED for r in t.rate]
ax.barh(t.index, t.rate * 100, color=colors,
        xerr=[(t.rate - t.lo) * 100, (t.hi - t.rate) * 100],
        error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
for i, (r, n) in enumerate(zip(t.rate, t.n)):
    ax.text(t.hi.iloc[i] * 100 + 1.5, i, f"{r:.0%}  (n={n})", va="center", fontsize=10)
ax.axvline(df.problem_occurred_flag.mean() * 100, color=GREY, ls="--", lw=1)
ax.text(df.problem_occurred_flag.mean() * 100 + 0.5, -0.75, "overall", color=GREY, fontsize=9)
ax.set_xlim(0, 80)
ax.set_xlabel("Journeys with a problem (%)  |  bars show 95% confidence range")
clear, snow = t.loc["Clear", "rate"], t.loc["Snow", "rate"]
ax.set_title(f"Problems rise from {clear:.0%} in clear weather to {snow:.0%} in snow", loc="left")
save(fig, "01_problem_rate_by_weather.png")

# ---- 2. Why? Sensor confidence vs the vehicle's normal level --------------
g = df.groupby("weather")["confidence_gap"].mean().sort_values()
summary.append("2. Mean sensor confidence gap vs nominal (points)\n" + g.round(1).to_string())
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(g.index, g.values, color=[RED if v < -8 else BLUE for v in g.values])
for i, v in enumerate(g.values):
    ax.text(v - 0.4 if v < 0 else v + 0.4, i, f"{v:+.1f}", va="center",
            ha="right" if v < 0 else "left", fontsize=10)
ax.axvline(0, color="#333", lw=1)
ax.set_xlabel("Average sensor confidence minus the vehicle's normal confidence (percentage points)")
ax.set_title(f"Snow cuts sensor confidence by {abs(g['Snow']):.0f} points; fog by {abs(g['Fog']):.0f}", loc="left")
save(fig, "02_confidence_gap_by_weather.png")

# ---- 3. Does better hardware help? Suite x weather ------------------------
order = ["Clear", "Light rain", "Strong wind", "Low sun", "Heavy rain", "Fog", "Snow"]
sub = df[df.weather.isin(order)]
piv = sub.pivot_table(index="weather", columns="sensor_suite",
                      values="problem_occurred_flag", aggfunc="mean").reindex(order)[["Standard", "Enhanced", "Advanced"]]
cnt = sub.pivot_table(index="weather", columns="sensor_suite",
                      values="problem_occurred_flag", aggfunc="count").reindex(order)[["Standard", "Enhanced", "Advanced"]]
summary.append("3. Problem rate by weather x suite\n" + piv.round(3).to_string() + "\n\nCounts\n" + cnt.to_string())
fig, ax = plt.subplots(figsize=(8.5, 5.5))
im = ax.imshow(piv.values * 100, cmap="Reds", vmin=0, vmax=60, aspect="auto")
ax.set_xticks(range(3), piv.columns)
ax.set_yticks(range(len(piv)), piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v, n = piv.iloc[i, j], cnt.iloc[i, j]
        star = "*" if n < 30 else ""
        txt = f"{v:.0%}{star}\nn={int(n)}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                color="white" if v > 0.35 else "black")
ax.spines[:].set_visible(False)
ax.set_title("Standard sensors fare worst in rain and snow; no suite escapes fog", loc="left")
fig.text(0.01, -0.02, "* Fewer than 30 journeys. Treat as directional.", fontsize=9, color=GREY)
save(fig, "03_suite_by_weather_heatmap.png")

# ---- 4. Can it warn us in time? Alerts -------------------------------------
df["alert_band"] = pd.cut(df.sensor_alert_count, [-1, 0, 1, 2, 3, 99],
                          labels=["0", "1", "2", "3", "4+"])
a = rate_table(df, "alert_band")
summary.append("4. Problem rate by sensor alert count\n" + a.round(3).to_string())
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(a.index.astype(str), a.rate * 100, color=[BLUE, BLUE, BLUE, RED, RED],
       yerr=[(a.rate - a.lo) * 100, (a.hi - a.rate) * 100],
       error_kw={"ecolor": "#333", "capsize": 3, "lw": 1})
for i, (r, n) in enumerate(zip(a.rate, a.n)):
    ax.text(i, a.hi.iloc[i] * 100 + 2, f"{r:.0%}\n(n={n})", ha="center", fontsize=10)
ax.set_ylim(0, 115)
ax.set_xlabel("Sensor alerts during the journey")
ax.set_ylabel("Journeys with a problem (%)")
ax.set_title(f"Sensor alerts are an early warning: {a.rate['2']:.0%} problems at 2 alerts, {a.rate['3']:.0%} at 3, {a.rate['4+']:.0%} at 4+", loc="left")
save(fig, "04_problem_rate_by_alerts.png")

# ---- 5. Does it fail safely? Events ---------------------------------------
ev = events.copy()
sev_order = ["Low", "Medium", "High"]
tab = pd.crosstab(ev.event_type, ev.severity).reindex(columns=sev_order, fill_value=0)
tab = tab.loc[tab.sum(axis=1).sort_values().index]
out = ev.groupby("event_type")["outcome"].apply(lambda s: (s == "Journey completed").mean())
done_all = df.journey_completed_flag.mean()
done_ev = (ev.outcome == "Journey completed").mean()
summary.append("5. Events by type and severity\n" + tab.to_string() +
               "\n\nShare of journeys completed after the event\n" + out.round(3).to_string() +
               f"\n\nAll journeys completed: {done_all:.3f} | Journeys with an event completed: {done_ev:.3f}")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
left = np.zeros(len(tab))
for s, c in zip(sev_order, ["#9fc5e8", "#f6b26b", RED]):
    axes[0].barh(tab.index, tab[s], left=left, color=c, label=s)
    left += tab[s].values
axes[0].set_xlabel("Journeys, by their most serious event")
axes[0].legend(title="Severity", frameon=False)
axes[0].set_title("Most events are low severity", loc="left")
o = out.sort_values()
axes[1].barh(o.index, o.values * 100, color=BLUE)
for i, v in enumerate(o.values):
    axes[1].text(v * 100 + 1, i, f"{v:.0%}", va="center")
axes[1].set_xlim(0, 115)
axes[1].set_xlabel("Journeys still completed (%)")
axes[1].set_title("Safety incidents often end the journey", loc="left")
save(fig, "05_events_fail_safely.png")

# ---- 6. What does it cost passengers? Delay and rating --------------------
df["delay_band"] = pd.cut(df.delay_min, [-1, 0, 5, 10, 100], labels=["0 min", "1-5", "6-10", "11+"])
r = df.groupby("delay_band", observed=True)["passenger_rating_1to5"].agg(["mean", "count"])
summary.append("6. Passenger rating by delay band\n" + r.round(2).to_string())
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(r.index.astype(str), r["mean"], color=[BLUE, BLUE, "#e67e22", RED])
for i, (m, n) in enumerate(zip(r["mean"], r["count"])):
    ax.text(i, m + 0.08, f"{m:.2f}\n(n={n})", ha="center", fontsize=10)
ax.set_ylim(0, 5.8)
ax.set_xlabel("Delay versus planned time (minutes)")
ax.set_ylabel("Average passenger rating (1-5)")
ax.set_title(f"Passengers punish delays: rating falls from {r['mean'].iloc[1]:.1f} (1-5 min) to {r['mean'].iloc[3]:.1f} (11+ min)", loc="left")
save(fig, "06_rating_by_delay.png")

HEADLINE = "EXPLORATION SUMMARY: where does the autonomous ride break down, and what warns us?"
(REP / "exploration_summary.txt").write_text(
    HEADLINE + "\n" + "=" * len(HEADLINE) + "\n\n" + "\n\n".join(summary), encoding="utf-8")
print("saved reports/exploration_summary.txt")