"""Modelling: can we predict which journeys will have a problem?

Run from the project root:  python src/05_model.py
Target: problem_occurred (1 = a material performance problem occurred)

Two feature sets are compared on purpose:
  PRE-JOURNEY   what is known at departure (weather, road, vehicle, traffic ...)
  IN-JOURNEY    pre-journey + live signals (sensor confidence, alerts, obstacle flag)
Outcome columns (interventions, stops, incidents, completion, delay, rating, risk_level)
are excluded because they describe what happened, which would be leakage.

Outputs: figures/10-14 (PNG) and reports/model_evaluation_report.txt
Nothing here changes the data. Run 02b_red_flags.py first for the sensitivity check.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
FIG, REP = ROOT / "figures", ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

BLUE, RED, ORANGE, GREY = "#1f4e79", "#c0392b", "#e67e22", "#7f8c8d"
plt.rcParams.update({"savefig.dpi": 200, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13})

df = pd.read_csv(ROOT / "data/processed/journeys_merged.csv")
y = df["problem_occurred_flag"]

# ---- Feature sets -----------------------------------------------------------
pre_cols = ["distance_km", "visibility_km", "traffic_level", "road_surface", "road_type",
            "time_of_day", "weather", "sensor_suite", "software_version",
            "vehicle_age_months", "nominal_sensor_confidence_pct"]
road_cols = [c for c in ["junction_complexity", "road_markings", "street_lighting",
                         "digital_infrastructure", "speed_limit_mph"] if c in df.columns]
pre_cols += road_cols
live_cols = ["sensor_confidence_pct", "sensor_alert_count", "unexpected_obstacle"]
EXCLUDED = ["human_intervention", "emergency_stop", "safety_incident", "journey_completed",
            "passenger_rating_1to5", "delay_min", "actual_time_min", "risk_level",
            "risk_multiplier / visibility_factor / sensor_confidence_adjustment (derived from weather)"]


def design(cols):
    return pd.get_dummies(df[cols], drop_first=True).astype(float)


X_pre, X_live = design(pre_cols), design(pre_cols + live_cols)

# ---- 80/20 split (same rows for every model) --------------------------------
idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=y)
ytr, yte = y.loc[idx_tr], y.loc[idx_te]

lines = []


def out(*a):
    lines.append(" ".join(str(x) for x in a))


def metrics(ytrue, pred, proba=None):
    m = {"accuracy": accuracy_score(ytrue, pred),
         "precision": precision_score(ytrue, pred, zero_division=0),
         "recall": recall_score(ytrue, pred),
         "f1": f1_score(ytrue, pred)}
    m["auc"] = roc_auc_score(ytrue, proba) if proba is not None else np.nan
    return m


def make_models():
    return {
        "Baseline (always 'no problem')": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=SEED),
    }


def proba_of(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None


# ---- 1. Fit all models on both feature sets -----------------------------------
rows, fitted, preds = [], {}, {}
for fs_name, X in [("Pre-journey", X_pre), ("In-journey", X_live)]:
    for name, model in make_models().items():
        model.fit(X.loc[idx_tr], ytr)
        pred = model.predict(X.loc[idx_te])
        pr = proba_of(model, X.loc[idx_te])
        if name.startswith("Baseline"):
            pr = None
        met = metrics(yte, pred, pr)
        rows.append({"features": fs_name, "model": name, **met})
        fitted[(fs_name, name)] = model
        preds[(fs_name, name)] = (pred, pr)
res = pd.DataFrame(rows)

HEADLINE = "MODEL EVALUATION REPORT: can we predict which autonomous journeys will have a problem?"
out(HEADLINE)
out("=" * len(HEADLINE))
out(f"\nJourneys: {len(df)} | problems: {int(y.sum())} ({y.mean():.1%}) | "
    f"train: {len(idx_tr)} | test: {len(idx_te)} | stratified 80/20 split, random_state={SEED}")
out(f"Pre-journey features ({X_pre.shape[1]} after encoding): {', '.join(pre_cols)}")
out(f"In-journey adds: {', '.join(live_cols)}")
out("Excluded as leakage (they describe what happened): " + "; ".join(EXCLUDED))

out("\n=== 1. Test-set results (500 unseen journeys) ===")
out(res.round(3).to_string(index=False))
base_acc = res[res.model.str.startswith("Baseline")].accuracy.iloc[0]
out(f"\nA model that always says 'no problem' is already {base_acc:.1%} accurate, "
    "so accuracy alone is not evidence of skill. Compare recall and precision.")

# ---- 2. Confusion matrices ------------------------------------------------------
out("\n=== 2. Confusion matrices (rows = actual, columns = predicted) ===")
cms = {}
for key in [("Pre-journey", "Logistic regression"), ("In-journey", "Logistic regression"),
            ("In-journey", "Decision tree (depth 4)")]:
    cm = confusion_matrix(yte, preds[key][0])
    cms[key] = cm
    tn, fp, fn, tp = cm.ravel()
    out(f"\n{key[0]} | {key[1]}\n  true negatives {tn} | false alarms {fp} | "
        f"missed problems {fn} | caught problems {tp}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (key, cm) in zip(axes, cms.items()):
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    ax.set_xticks([0, 1], ["No problem", "Problem"])
    ax.set_yticks([0, 1], ["No problem", "Problem"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{key[0]}\n{key[1].split(' (')[0]}", fontsize=11)
    ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(FIG / "10_model_confusion_matrices.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 3. Pre-journey vs in-journey -----------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
sub = res[res.model == "Logistic regression"].set_index("features").loc[["Pre-journey", "In-journey"]]
xs = np.arange(3)
for k, (fs, col) in enumerate(zip(sub.index, [GREY, BLUE])):
    vals = [sub.loc[fs, "precision"], sub.loc[fs, "recall"], sub.loc[fs, "auc"]]
    ax.bar(xs + (k - 0.5) * 0.36, vals, 0.36, color=col, label=fs)
    for x, v in zip(xs + (k - 0.5) * 0.36, vals):
        ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=10)
ax.set_xticks(xs, ["Precision", "Recall", "AUC"])
ax.set_ylim(0, 1.12)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
n_warn = int(cms[("Pre-journey", "Logistic regression")][:, 1].sum())
fig.text(0.01, -0.07, f"Pre-journey precision rests on only {n_warn} warnings out of 500 journeys.", fontsize=9, color=GREY)
ax.set_title(f"Live sensor signals make prediction work: recall {sub.loc['Pre-journey', 'recall']:.0%} "
             f"before departure vs {sub.loc['In-journey', 'recall']:.0%} during the journey", loc="left")
fig.tight_layout()
fig.savefig(FIG / "11_model_pre_vs_in_journey.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 4. Stability: 5-fold cross-validation on the training set -----------------
out("\n=== 3. Stability check: 5-fold cross-validation on the training data ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for fs_name, X in [("Pre-journey", X_pre), ("In-journey", X_live)]:
    for name in ["Logistic regression", "Decision tree (depth 4)"]:
        model = make_models()[name]
        p = cross_val_predict(model, X.loc[idx_tr], ytr, cv=skf)
        fold_f1 = [f1_score(ytr.iloc[te], p[te]) for _, te in skf.split(X.loc[idx_tr], ytr)]
        out(f"{fs_name:12s} | {name:24s} | F1 per fold: mean {np.mean(fold_f1):.3f}, "
            f"std {np.std(fold_f1):.3f}, range {min(fold_f1):.3f}-{max(fold_f1):.3f}")

# ---- 5. Missed problems vs false alarms: threshold trade-off -------------------
lr = fitted[("In-journey", "Logistic regression")]
p_te = preds[("In-journey", "Logistic regression")][1]
prec, rec, thr = precision_recall_curve(yte, p_te)
out("\n=== 4. Operating point: how many problems do we catch vs false alarms we accept? ===")
trade = []
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    pr_ = (p_te >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pr_).ravel()
    trade.append({"threshold": t, "recall": recall_score(yte, pr_), "precision": precision_score(yte, pr_, zero_division=0),
                  "caught": tp, "missed": fn, "false_alarms": fp})
out(pd.DataFrame(trade).round(3).to_string(index=False))
# Honest check: the same trade-off estimated from the TRAINING data only (5-fold CV), so no test-set peeking
oof_tr = cross_val_predict(make_models()["Logistic regression"], X_live.loc[idx_tr], ytr, cv=skf,
                           method="predict_proba")[:, 1]
cv_rows = []
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    tn, fp, fn, tp = confusion_matrix(ytr, (oof_tr >= t).astype(int)).ravel()
    cv_rows.append({"threshold": t, "recall": tp / (tp + fn), "precision": tp / (tp + fp),
                    "false_alarm_rate": fp / (fp + tn)})
out("\nSame trade-off estimated from the training data only (5-fold cross-validation, no test-set peeking):")
out(pd.DataFrame(cv_rows).round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(rec, prec, color=BLUE, lw=2)
for r in trade:
    if r["threshold"] in (0.5, 0.2):
        ax.scatter(r["recall"], r["precision"], color=RED, zorder=5)
        ax.annotate(f"threshold {r['threshold']}\n{r['missed']} missed, {r['false_alarms']} false alarms",
                    (r["recall"], r["precision"]), textcoords="offset points",
                    xytext=(-130 if r["threshold"] == 0.2 else -150, -45), fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": GREY})
ax.set_xlabel("Recall (share of real problems caught)")
ax.set_ylabel("Precision (share of warnings that were real)")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.set_title("Catching more problems means accepting more false alarms", loc="left")
fig.tight_layout()
fig.savefig(FIG / "12_model_missed_vs_false_alarms.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 6. Simple operator rule vs the model ----------------------------------------
out("\n=== 5. Simple rule an operator could use: 'escalate to a human at N or more alerts' ===")
rule_rows = []
for n in [2, 3, 4]:
    pr_ = (df.loc[idx_te, "sensor_alert_count"] >= n).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pr_).ravel()
    rule_rows.append({"rule": f">= {n} alerts", "recall": recall_score(yte, pr_),
                      "precision": precision_score(yte, pr_, zero_division=0),
                      "caught": tp, "missed": fn, "false_alarms": fp})
out(pd.DataFrame(rule_rows).round(3).to_string(index=False))

# ---- 7. What drives the model? ------------------------------------------------------
coefs = pd.Series(lr[-1].coef_[0], index=X_live.columns).sort_values(key=abs, ascending=False)
out("\n=== 6. Logistic regression: strongest drivers (standardised coefficients) ===")
out("Positive = raises the chance of a problem. Features are scaled, so sizes are comparable.")
out(coefs.head(12).round(2).to_string())
top = coefs.head(10)[::-1]
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.barh(top.index, top.values, color=[RED if v > 0 else BLUE for v in top.values])
ax.axvline(0, color="#333", lw=1)
ax.set_xlabel("Effect on the odds of a problem (standardised; right = more likely)")
ax.set_title("Sensor alerts and sensor confidence dominate the model", loc="left")
fig.tight_layout()
fig.savefig(FIG / "13_model_logistic_drivers.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

tree = fitted[("In-journey", "Decision tree (depth 4)")]
out("\n=== 7. Decision tree rules (readable by a non-technical reviewer) ===")
out(export_text(tree, feature_names=list(X_live.columns), max_depth=4))
fig, ax = plt.subplots(figsize=(15, 7))
plot_tree(tree, feature_names=list(X_live.columns), class_names=["No problem", "Problem"],
          filled=True, rounded=True, max_depth=3, fontsize=8, ax=ax, impurity=False)
ax.set_title("Decision tree: a few alert and confidence thresholds do most of the work", loc="left")
fig.savefig(FIG / "14_model_decision_tree.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- 8. Sensitivity: remove the red-flagged journeys -----------------------------
out("\n=== 8. Sensitivity: does the result change without the red-flagged journeys? ===")
rf_path = REP / "red_flag_table.csv"
if rf_path.exists():
    rf = pd.read_csv(rf_path)
    drop_ids = set(rf.loc[rf.severity == "ANOMALY", "journey_id"])
    keep = ~df.journey_id.isin(drop_ids)
    tr2 = [i for i in idx_tr if keep.loc[i]]
    te2 = [i for i in idx_te if keep.loc[i]]
    m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X_live.loc[tr2], y.loc[tr2])
    p2 = m2.predict(X_live.loc[te2])
    a = pd.DataFrame([
        {"run": "All journeys", **metrics(yte, preds[("In-journey", "Logistic regression")][0],
                                          preds[("In-journey", "Logistic regression")][1])},
        {"run": f"Without {len(drop_ids)} flagged journeys",
         **metrics(y.loc[te2], p2, m2.predict_proba(X_live.loc[te2])[:, 1])}])
    out(a.round(3).to_string(index=False))
else:
    out("reports/red_flag_table.csv not found. Run python src/02b_red_flags.py first.")

# ---- 9. Fairness: who does the model miss or over-warn? --------------------------
out("\n=== 9. Error rates by group (out-of-fold predictions on ALL journeys, in-journey logistic) ===")
out("miss_rate = real problems the model failed to flag | false_alarm_rate = trouble-free journeys it flagged")
oof = cross_val_predict(make_models()["Logistic regression"], X_live, y, cv=skf)
out(f"Overall, all {len(df)} journeys, 5-fold cross-validated: accuracy {accuracy_score(y, oof):.3f}, "
    f"precision {precision_score(y, oof):.3f}, recall {recall_score(y, oof):.3f}, F1 {f1_score(y, oof):.3f} "
    "(a steadier estimate than the single 500-journey test split)")
rule3 = (df.sensor_alert_count >= 3).astype(int)
out(f"Rule '3 or more alerts' on all {len(df)} journeys: recall {recall_score(y, rule3):.3f}, "
    f"precision {precision_score(y, rule3):.3f}, false alarms {int(((rule3 == 1) & (y == 0)).sum())}")
fair = df[["weather", "sensor_suite", "city", "road_type"]].copy()
fair["y"], fair["pred"] = y.values, oof
fair_tables = {}
for col in ["weather", "sensor_suite", "city", "road_type"]:
    r = []
    for g, t in fair.groupby(col):
        pos, neg = (t.y == 1).sum(), (t.y == 0).sum()
        r.append({col: g, "n": len(t), "problems": pos,
                  "miss_rate": ((t.y == 1) & (t.pred == 0)).sum() / pos if pos else np.nan,
                  "false_alarm_rate": ((t.y == 0) & (t.pred == 1)).sum() / neg if neg else np.nan})
    ft = pd.DataFrame(r).set_index(col)
    fair_tables[col] = ft
    out(f"\nBy {col}:\n{ft.round(3).to_string()}")
out("\nCaution: some groups have few problems (see the problems column), so rates there are rough.")

out("\n=== 10. Honest limitations ===")
out("- Data is synthetic and observational: patterns are associations, not causes.")
out("- problem_occurred appears strongly tied to alert count and sensor confidence, so the in-journey "
    "model partly recovers how the target was built. It detects trouble as it happens; it does not forecast it.")
out("- Before departure the model barely beats the baseline: conditions alone do not predict problems well.")
out("- Small groups (snow, heat alert, fog) give uncertain error rates. No passenger attributes exist, "
    "so fairness across passenger groups cannot be tested.")

(REP / "model_evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:40]))
print("\nSaved figures 10-14 and reports/model_evaluation_report.txt")
