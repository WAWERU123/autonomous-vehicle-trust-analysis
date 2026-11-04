"""Events analysis: does the system fail safely when something goes wrong?

Run from the project root:  python src/04_events.py
Saves charts to figures/ and numbers to reports/events_analysis_summary.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG, REP = ROOT / "figures", ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/processed/journeys_merged.csv")
ev = pd.read_csv(ROOT / "data/raw/intervention_events.csv")
m = ev.merge(df[["journey_id", "weather", "sensor_suite"]], on="journey_id", how="left")

BLUE, RED, ORANGE, GREY = "#1f4e79", "#c0392b", "#e67e22", "#7f8c8d"
plt.rcParams.update({
    "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})
summary = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


N = len(df)
n_event = ev.journey_id.nunique()
n_serious = ev.event_type.isin(["Emergency stop", "Safety incident"]).sum()
n_incident = (ev.event_type == "Safety incident").sum()
n_notdone = (ev.outcome == "Journey not completed").sum()

# consistency check: event outcome agrees with the journey's completion flag
chk = ev.merge(df[["journey_id", "journey_completed"]], on="journey_id")
agree = ((chk.outcome == "Journey completed") == (chk.journey_completed == "Yes")).mean()

# ---- 7. The safety funnel ---------------------------------------------------
steps = [
    ("All journeys", N),
    ("Any event (intervention, stop or incident)", n_event),
    ("Emergency stop or safety incident", n_serious),
    ("Journey not completed", n_notdone),
    ("Safety incident", n_incident),
]
summary.append("7. Funnel\n" + "\n".join(f"{a}: {b} ({b / N:.1%})" for a, b in steps) +
               f"\nEvent outcome matches journey_completed flag: {agree:.1%}")
fig, ax = plt.subplots(figsize=(10, 4.8))
labels = [s[0] for s in steps][::-1]
vals = [s[1] for s in steps][::-1]
ax.barh(labels, vals, color=[BLUE, RED, ORANGE, ORANGE, "#9fc5e8"][::-1])
for i, v in enumerate(vals):
    ax.text(v + N * 0.01, i, f"{v:,}  ({v / N:.1%})", va="center")
ax.set_xlim(0, N * 1.22)
ax.set_xlabel("Journeys")
ax.set_title(f"{1 - n_event / N:.0%} of journeys had no event; {n_serious / N:.1%} had a serious one "
             f"and {n_notdone / N:.1%} did not finish", loc="left")
save(fig, "07_safety_funnel.png")

# NOTE: the events file holds ONE row per journey: its most severe event (incident > stop > intervention).
# ---- 8. Event rate by weather, split by type ---------------------------------
order = ["Clear", "Strong wind", "Light rain", "Heat alert", "Low sun", "Heavy rain", "Fog", "Snow"]
n_j = df.groupby("weather").size().reindex(order)
tab = pd.crosstab(m.weather, m.event_type).reindex(order).fillna(0)
rate = tab.div(n_j, axis=0) * 100
rate = rate[["Human intervention", "Emergency stop", "Safety incident"]]
summary.append("8. Events per 100 journeys by weather\n" + rate.round(1).to_string() +
               "\n\nJourneys per weather\n" + n_j.to_string() + "\n\nEvent counts\n" + tab.astype(int).to_string())
fig, ax = plt.subplots(figsize=(10, 5.2))
left = np.zeros(len(rate))
for c, col in zip(rate.columns, ["#9fc5e8", ORANGE, RED]):
    ax.barh(rate.index, rate[c], left=left, color=col, label=c)
    left += rate[c].values
for i, (tot, n) in enumerate(zip(left, n_j)):
    ax.text(tot + 0.6, i, f"{tot:.0f}  (n={n})", va="center", fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, left.max() * 1.2)
ax.set_xlabel("Journeys per 100, grouped by their most serious event")
ax.legend(frameon=False, loc="upper right")
ax.set_title(f"Events are {rate.loc['Snow'].sum() / rate.loc['Clear'].sum():.0f}x more frequent in snow than in clear weather", loc="left")
save(fig, "08_events_per_100_by_weather.png")

# ---- 9. Why do humans step in? Reason x weather ------------------------------
hi = m[m.event_type == "Human intervention"]
reason_order = hi.event_reason.value_counts().index.tolist()
ct = pd.crosstab(hi.event_reason, hi.weather).reindex(index=reason_order, columns=order).fillna(0).astype(int)
summary.append("9. Human intervention reasons by weather\n" + ct.to_string() +
               "\n\nTotals\n" + hi.event_reason.value_counts().to_string())
low = hi[hi.event_reason == "Low sensor confidence"]
bad = ["Fog", "Snow", "Heavy rain"]
share_fs = low.weather.isin(bad).mean() if len(low) else 0
share_all = df.weather.isin(bad).mean()
summary.append(f"Low sensor confidence events in fog, snow or heavy rain: {share_fs:.0%} of {len(low)}")
fig, ax = plt.subplots(figsize=(10, 4.6))
im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(ct.columns)), ct.columns, rotation=30, ha="right")
ax.set_yticks(range(len(ct.index)), ct.index)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.iloc[i, j]
        ax.text(j, i, v, ha="center", va="center",
                color="white" if v > ct.values.max() * 0.55 else "black", fontsize=10)
ax.spines[:].set_visible(False)
ax.set_title(f"It asks for help when its sensors struggle:\n{share_fs:.0%} of low-confidence handovers "
             f"came in fog, snow or heavy rain", loc="left")
fig.text(0.01, -0.03, "Counts of human interventions (n=" + str(len(hi)) +
         f"). Fog, snow and heavy rain are only {share_all:.0%} of all journeys.", fontsize=9, color=GREY)
save(fig, "09_handover_reasons_by_weather.png")

HEADLINE = "EVENTS ANALYSIS SUMMARY: does the system fail safely? (interventions, stops, incidents)"
(REP / "events_analysis_summary.txt").write_text(
    HEADLINE + "\n" + "=" * len(HEADLINE) + "\n\n" + "\n\n".join(summary), encoding="utf-8")
print("saved reports/events_analysis_summary.txt")