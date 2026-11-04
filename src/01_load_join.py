from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def load(name, required=True):
    path = RAW / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {path}")
        print(f"[skip] {name} not found yet")
        return None
    df = pd.read_csv(path)
    print(f"[ok]   {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


journeys = load("autonomous_vehicle_journeys.csv")
vehicles = load("vehicle_sensor_profiles.csv")
weather = load("weather_risk_factors.csv")
events = load("intervention_events.csv")
dictionary = load("data_dictionary.csv")
roads = load("road_environment_factors.csv", required=False)  # add when you have it

# ---- Basic checks ---------------------------------------------------------
print("\n--- Checks ---")
print("journey_id unique:", journeys["journey_id"].is_unique)
print("missing values in journeys:", int(journeys.isna().sum().sum()))
print("duplicate rows:", int(journeys.duplicated().sum()))
print(
    "delay = actual - planned everywhere:",
    (journeys["delay_min"] == journeys["actual_time_min"] - journeys["planned_time_min"]).all(),
)
print("events link to journeys:", events["journey_id"].isin(journeys["journey_id"]).all())
print("journeys with >=1 event:", events["journey_id"].nunique())

# ---- Joins (left joins keep all 2,500 journeys) ---------------------------
df = journeys.merge(vehicles, on="vehicle_profile_id", how="left", validate="m:1")
df = df.merge(weather, on="weather_code", how="left", validate="m:1")
if roads is not None:
    overlap = [c for c in roads.columns if c in df.columns and c != "road_segment_id"]
    roads = roads.rename(columns={c: f"{c}_env" for c in overlap})
    df = df.merge(roads, on="road_segment_id", how="left", validate="m:1")

assert len(df) == len(journeys), "Join changed the row count - investigate!"
print("\nunmatched vehicle profiles:", int(df["sensor_suite"].isna().sum()))
print("unmatched weather codes:", int(df["weather"].isna().sum()))

# ---- Helpful derived columns ---------------------------------------------
for col in ["problem_occurred", "human_intervention", "emergency_stop",
            "safety_incident", "journey_completed", "unexpected_obstacle"]:
    df[col + "_flag"] = (df[col] == "Yes").astype(int)

df["confidence_gap"] = df["sensor_confidence_pct"] - df["nominal_sensor_confidence_pct"]

df.to_csv(OUT / "journeys_merged.csv", index=False)
print(f"\nSaved {OUT / 'journeys_merged.csv'}  shape={df.shape}")
print("\nProblem rate:", round(df["problem_occurred_flag"].mean(), 3))