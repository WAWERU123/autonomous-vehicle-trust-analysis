"""Red-flag table: a reproducible anomaly report to run BEFORE modelling.

Run from the project root:  python src/02b_red_flags.py
Writes reports/red_flag_table.csv  (one row per journey per issue)

severity:  ANOMALY = a logical contradiction in the data (review before modelling)
           NOTE    = a structural feature worth knowing about (not an error)
Nothing is dropped or changed. This file is an audit trail only.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"
REP.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data/processed/journeys_merged.csv")
ev = pd.read_csv(ROOT / "data/raw/intervention_events.csv")

issues = []


def flag(mask, issue_type, severity, description, frame=df):
    for jid in frame.loc[mask, "journey_id"]:
        issues.append((jid, issue_type, severity, description))


yes = lambda col: df[col] == "Yes"
HI, ES, SI = yes("human_intervention"), yes("emergency_stop"), yes("safety_incident")
any_event = HI | ES | SI

# ---- 1. Malformed rows: values outside the allowed set / unparseable --------
allowed = {
    "human_intervention": {"Yes", "No"}, "emergency_stop": {"Yes", "No"},
    "safety_incident": {"Yes", "No"}, "journey_completed": {"Yes", "No"},
    "problem_occurred": {"Yes", "No"}, "unexpected_obstacle": {"Yes", "No"},
    "risk_level": {"Low", "Medium", "High"}, "traffic_level": {"Low", "Moderate", "High"},
}
bad = pd.Series(False, index=df.index)
for col, ok in allowed.items():
    bad |= ~df[col].isin(ok)
bad |= df.isna().any(axis=1)
bad |= ~df["passenger_rating_1to5"].isin([1, 2, 3, 4, 5])
flag(bad, "Malformed row", "ANOMALY", "Missing value or value outside the allowed set")

# ---- 2. Numeric plausibility -------------------------------------------------
flag(~df.sensor_confidence_pct.between(0, 100), "Confidence out of range", "ANOMALY",
     "sensor_confidence_pct not between 0 and 100")
flag(df.delay_min < 0, "Negative delay", "ANOMALY", "delay_min is negative")
flag(df.delay_min != df.actual_time_min - df.planned_time_min, "Delay arithmetic", "ANOMALY",
     "delay_min is not actual minus planned")

# ---- 3. Flag consistency -----------------------------------------------------
flag(any_event & (df.problem_occurred == "No"), "Problem flag mismatch", "ANOMALY",
     "Intervention, stop or incident recorded but problem_occurred = No")
flag((df.journey_completed == "No") & (df.problem_occurred == "No"), "Not completed, no problem",
     "ANOMALY", "Journey not completed but problem_occurred = No")

# ---- 4. Journeys vs events file ----------------------------------------------
has_event = df.journey_id.isin(ev.journey_id)
flag(any_event & ~has_event, "Missing event record", "ANOMALY",
     "Flagged journey has no row in intervention_events.csv")
flag(~any_event & has_event, "Event without flag", "ANOMALY",
     "Event record exists but no intervention/stop/incident flag on the journey")

m = ev.merge(df[["journey_id", "journey_completed"]], on="journey_id", how="left")
# Contradiction INSIDE the events file: response says terminated, outcome says completed
x = (m.response_action == "Journey terminated") & (m.outcome == "Journey completed")
flag(df.journey_id.isin(m.loc[x, "journey_id"]), "Terminated but completed", "ANOMALY",
     "Event response is 'Journey terminated' but outcome is 'Journey completed'")
y = (m.outcome == "Journey completed") != (m.journey_completed == "Yes")
flag(df.journey_id.isin(m.loc[y, "journey_id"]), "Outcome vs completion flag", "ANOMALY",
     "Event outcome disagrees with journey_completed")

# ---- 5. Structure notes (not errors) -----------------------------------------
n_flags = HI.astype(int) + ES.astype(int) + SI.astype(int)
flag(n_flags > 1, "Multiple flags, one event row", "NOTE",
     "Several flags on this journey but the events file holds one row (its most severe event)")
flag(SI & (df.journey_completed == "Yes"), "Safety incident, journey completed", "NOTE",
     "Safety incident but journey reached its destination (not necessarily an error)")

# ---- Build and save ----------------------------------------------------------
red = pd.DataFrame(issues, columns=["journey_id", "issue_type", "severity", "description"])
red = red.sort_values(["severity", "issue_type", "journey_id"]).reset_index(drop=True)
red.to_csv(REP / "red_flag_table.csv", index=False)

print(f"Journeys checked: {len(df)} | events checked: {len(ev)}\n")
if red.empty:
    print("No red flags found.")
else:
    print(red.groupby(["severity", "issue_type"]).size().rename("journeys").to_string())
    print(f"\nJourneys with at least one ANOMALY: {red.loc[red.severity == 'ANOMALY', 'journey_id'].nunique()}")
print("\nSaved reports/red_flag_table.csv")