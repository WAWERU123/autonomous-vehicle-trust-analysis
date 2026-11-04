"""Data validation: checks the merged journeys data before any analysis.
Prints and saves everything to reports/data_validation_report.txt

Run from the project root:  python src/02_checks.py
"""
from pathlib import Path
import pandas as pd

Path("reports").mkdir(exist_ok=True)
lines = []
HEADLINE = "DATA VALIDATION REPORT: are the autonomous vehicle journey files clean and consistent?"


def out(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    lines.append(text)


df = pd.read_csv("data/processed/journeys_merged.csv")
events = pd.read_csv("data/raw/intervention_events.csv")
roads = pd.read_csv("data/raw/road_environment_factors.csv")

out(HEADLINE)
out("=" * len(HEADLINE) + "\n")
out("=== 1. Shape and duplicate columns ===")
out(df.shape)
for c in ["city", "road_type"]:
    same = df[c] == df[f"{c}_env"]
    out(f"{c} agrees with road file: {same.mean() * 100:.1f}% (mismatches: {(~same).sum()})")

out("\n=== 2. Categories (typos / rare values) ===")
skip = {"journey_id", "journey_date"}
for c in df.select_dtypes("object").columns:
    if c in skip or c.endswith("_env"):
        continue
    out(f"\n{c}:\n{df[c].value_counts(dropna=False).to_string()}")

out("\n=== 3. Number ranges ===")
out(df.describe().T[["min", "mean", "max"]].round(2).to_string())

out("\n=== 4. Logic checks (all should be 0) ===")
flagged = (df.human_intervention == "Yes") | (df.emergency_stop == "Yes") | (df.safety_incident == "Yes")
out("Not completed but no problem:", ((df.journey_completed == "No") & (df.problem_occurred == "No")).sum())
out("Intervention/e-stop/incident but no problem:", (flagged & (df.problem_occurred == "No")).sum())
out("Safety incident but not Yes in problem:", ((df.safety_incident == "Yes") & (df.problem_occurred == "No")).sum())
out("Confidence outside 0-100:", (~df.sensor_confidence_pct.between(0, 100)).sum())
out("Negative delay:", (df.delay_min < 0).sum())
out("Dates:", df.journey_date.min(), "to", df.journey_date.max())

out("\n=== 5. Events vs journey flags ===")
ev = events.groupby("journey_id").size().rename("n_events")
chk = df.merge(ev, on="journey_id", how="left").fillna({"n_events": 0})
out(pd.crosstab(flagged.values, (chk.n_events > 0).values, rownames=["flagged"], colnames=["has_event"]).to_string())
out("Max events per journey:", int(ev.max()))

out("\n=== 6. Group sizes (small groups = cautious conclusions) ===")
for c in ["weather", "sensor_suite", "software_version", "city", "road_type", "time_of_day", "traffic_level"]:
    out(f"\n{c}:\n{df.groupby(c).size().sort_values().to_string()}")

out("\n=== 7. Journeys per road segment ===")
out(df.groupby("road_segment_id").size().describe().round(1).to_string())

out("\n=== 8. Road feature group sizes ===")
for c in ["junction_complexity", "road_markings", "street_lighting", "digital_infrastructure"]:
    out(f"\n{c}:\n{df.groupby(c).size().to_string()}")

out("\n=== 9. Segments per road feature (30 segments in total) ===")
for c in ["junction_complexity", "road_markings", "street_lighting", "digital_infrastructure"]:
    out(f"{c}: {roads[c].value_counts().to_dict()}")

Path("reports/data_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved reports/data_validation_report.txt")