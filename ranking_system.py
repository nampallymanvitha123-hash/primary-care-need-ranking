"""
Prioritising Unmet Primary Care Need in England - ranking system.
MSc Computer Science Independent Industry Project, Aston University (2026).
Author: Manvitha Nampally

Run from the repository root:  python src/ranking_system.py
"""

# ===== C.1 Setup and synthetic data generation =====

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

N_AREAS = 500
N_EMBEDDED_GAPS = 30


def generate_dataset(seed=42):
    """Creates 500 fictional areas and plants 30 high-need, low-provision areas."""
    rng = np.random.default_rng(seed)
    n, k = N_AREAS, N_EMBEDDED_GAPS
    regions = ["Northshire", "Midlandsvale", "Eastborough", "Westmoor",
               "Southford", "Coastshire", "Londonfield"]
    df = pd.DataFrame({
        "area_id": [f"AREA_{i:03d}" for i in range(1, n + 1)],
        "region": rng.choice(regions, size=n,
                             p=[0.14, 0.16, 0.14, 0.14, 0.15, 0.12, 0.15]),
        "rurality": rng.choice(["Urban", "Suburban", "Rural"], size=n,
                               p=[0.45, 0.35, 0.20]),
    })
    urban = df["rurality"].eq("Urban")
    suburban = df["rurality"].eq("Suburban")
    rural = df["rurality"].eq("Rural")

    # Population and need indicators
    multiplier = df["rurality"].map(
        {"Urban": 1.25, "Suburban": 0.90, "Rural": 0.55}).to_numpy()
    base_population = rng.lognormal(mean=np.log(22000), sigma=0.55, size=n)
    df["population"] = np.clip(
        (base_population * multiplier).round().astype(int), 2000, 80000)
    df["pct_age_65_plus"] = np.clip(
        rng.normal(19, 5.5, n) + np.where(rural, 3.0, 0.0), 8, 35).round(2)
    df["deprivation_score"] = np.clip(
        rng.beta(2.2, 2.2, n) * 100
        + np.where(urban, 4.0, 0.0) - np.where(rural, 2.0, 0.0), 0, 100).round(2)
    df["chronic_illness_rate"] = np.clip(
        5 + 0.18 * df["pct_age_65_plus"] + 0.16 * df["deprivation_score"]
        + rng.normal(0, 3.0, n), 5, 35).round(2)
    df["population_growth_rate"] = np.clip(
        rng.normal(1.4, 1.6, n) + np.where(urban, 0.45, 0.0), -2, 6).round(2)

    # Provision indicators
    expected_gp_fte = (df["population"] / 6500 + np.where(urban, 0.9, 0.0)
                       - np.where(rural, 0.5, 0.0) + rng.normal(0, 1.1, n))
    df["gp_fte"] = np.clip(expected_gp_fte, 0.5, 25).round(2)
    df["patients_per_gp"] = (df["population"] / df["gp_fte"]).round(2)
    base_distance = rng.gamma(2.0, 1.4, n)
    df["nearest_gp_distance_km"] = np.clip(
        base_distance + np.where(rural, 4.5, 0.0)
        + np.where(suburban, 1.2, 0.0) - np.where(urban, 0.6, 0.0), 0.2, 20).round(2)

    # Plant 30 high-need, low-provision areas (hidden label)
    idx = rng.choice(df.index, size=k, replace=False)
    df["is_embedded_gap"] = False
    df.loc[idx, "is_embedded_gap"] = True

    def boost(col, low, high, lo_clip, hi_clip):
        df.loc[idx, col] = np.clip(df.loc[idx, col] + rng.uniform(low, high, k),
                                   lo_clip, hi_clip).round(2)

    boost("pct_age_65_plus", 6, 12, 8, 35)
    boost("deprivation_score", 20, 40, 0, 100)
    boost("chronic_illness_rate", 6, 13, 5, 35)
    boost("population_growth_rate", 1.0, 3.0, -2, 6)
    df.loc[idx, "gp_fte"] = np.clip(
        df.loc[idx, "gp_fte"] * rng.uniform(0.35, 0.65, k), 0.5, 25).round(2)
    df.loc[idx, "patients_per_gp"] = (
        df.loc[idx, "population"] / df.loc[idx, "gp_fte"]).round(2)
    boost("nearest_gp_distance_km", 4, 10, 0.2, 20)
    return df


df = generate_dataset(42)
print(df.shape, "embedded areas:", df["is_embedded_gap"].sum())

# ===== C.2 Min-max normalisation =====

def min_max_scale(series):
    """
    Scales a numeric pandas Series to a 0-100 range.
    Higher original values produce higher scaled values.
    """
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(50.0, index=series.index)

    return ((series - minimum) / (maximum - minimum)) * 100


INDICATORS = {
    "age_65_plus_norm": "pct_age_65_plus",
    "deprivation_norm": "deprivation_score",
    "chronic_illness_norm": "chronic_illness_rate",
    "population_growth_norm": "population_growth_rate",
    "patients_per_gp_norm": "patients_per_gp",
    "distance_to_gp_norm": "nearest_gp_distance_km",
}

scored_df = df.copy()
for new_column, raw_column in INDICATORS.items():
    scored_df[new_column] = min_max_scale(scored_df[raw_column])

# ===== C.3 Baseline weights =====

NEED_WEIGHTS = {
    "age_65_plus_norm": 0.25,
    "deprivation_norm": 0.30,
    "chronic_illness_norm": 0.30,
    "population_growth_norm": 0.15
}

PROVISION_WEIGHTS = {
    "patients_per_gp_norm": 0.60,
    "distance_to_gp_norm": 0.40
}

# ===== C.4 Need Score and Provision Gap Score =====

scored_df["need_score"] = sum(
    scored_df[indicator] * weight
    for indicator, weight in NEED_WEIGHTS.items()
)

scored_df["provision_gap_score"] = sum(
    scored_df[indicator] * weight
    for indicator, weight in PROVISION_WEIGHTS.items()
)

scored_df["need_score"] = scored_df["need_score"].round(2)
scored_df["provision_gap_score"] = scored_df["provision_gap_score"].round(2)

# ===== C.5 Priority Score and ranking =====

scored_df["priority_score"] = (
    scored_df["need_score"] * scored_df["provision_gap_score"]
) / 100

scored_df["priority_score"] = scored_df["priority_score"].round(2)

ranked_df = scored_df.sort_values(
    by="priority_score",
    ascending=False
).reset_index(drop=True)

ranked_df["priority_rank"] = ranked_df.index + 1

columns = ["priority_rank", "area_id", "rurality", "need_score",
           "provision_gap_score", "priority_score"]
print(ranked_df[columns].head(10).to_string(index=False))

# ===== C.6 Evaluation against the hidden labels =====

def evaluate_ranking(input_df, score_column, top_k=30):
    """
    Evaluates a completed score/ranking against the hidden ground-truth label.
    The label is used only after scoring.
    """
    evaluation_df = input_df.sort_values(
        by=score_column,
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True).copy()

    evaluation_df["rank"] = evaluation_df.index + 1
    evaluation_df["flagged"] = evaluation_df["rank"] <= top_k

    tp = int((evaluation_df["flagged"] & evaluation_df["is_embedded_gap"]).sum())
    fp = int((evaluation_df["flagged"] & ~evaluation_df["is_embedded_gap"]).sum())
    fn = int((~evaluation_df["flagged"] & evaluation_df["is_embedded_gap"]).sum())
    tn = int((~evaluation_df["flagged"] & ~evaluation_df["is_embedded_gap"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "True positives": tp,
        "False positives": fp,
        "False negatives": fn,
        "True negatives": tn,
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 score": round(f1 * 100, 2),
        "False-positive rate": round(fp / (fp + tn) * 100, 2)
    }


print(evaluate_ranking(ranked_df, "priority_score", top_k=30))
# Expected: 23 true positives, 7 false positives, F1 76.67, false-positive rate 1.49

# ===== C.7 Comparison of scoring methods and the dual-threshold rule =====

need, prov = scored_df["need_score"], scored_df["provision_gap_score"]
scored_df["equal_sum"] = 0.5 * need + 0.5 * prov
scored_df["need_emphasis"] = 0.6 * need + 0.4 * prov
scored_df["provision_emphasis"] = 0.4 * need + 0.6 * prov

for method in ["priority_score", "equal_sum", "need_emphasis", "provision_emphasis"]:
    result = evaluate_ranking(scored_df, method, top_k=30)
    print(f"{method:20s} TP {result['True positives']:2d}  F1 {result['F1 score']}")

# Dual threshold: scored only on the areas it actually flags (not a forced top 30)
flagged = (need >= 60) & (prov >= 60)
tp = int((flagged & scored_df["is_embedded_gap"]).sum())
n_flagged = int(flagged.sum())  # guard against zero flagged areas
precision = tp / n_flagged if n_flagged else 0.0
recall = tp / N_EMBEDDED_GAPS
print("Dual threshold: flagged", int(flagged.sum()), "TP", tp,
      "F1", round(2 * precision * recall / (precision + recall) * 100, 2) if tp else 0.0)
# Expected: 23, 23, 25, 19; dual threshold flags 5 areas, 3 embedded, F1 17.14

# ===== C.8 Repeating the experiment over 100 random seeds =====

def score_areas(data, log_ppg=False, by_rurality=False):
    """Recomputes all scores; options apply the two normalisation fixes."""
    d = data.copy()
    for new_column, raw_column in list(INDICATORS.items())[:4]:
        d[new_column] = min_max_scale(d[raw_column])
    ppg = np.log(d["patients_per_gp"]) if log_ppg else d["patients_per_gp"]
    d["_ppg"], d["_dist"] = ppg, d["nearest_gp_distance_km"]
    if by_rurality:
        groups = d.groupby("rurality")
        d["patients_per_gp_norm"] = groups["_ppg"].transform(min_max_scale)
        d["distance_to_gp_norm"] = groups["_dist"].transform(min_max_scale)
    else:
        d["patients_per_gp_norm"] = min_max_scale(d["_ppg"])
        d["distance_to_gp_norm"] = min_max_scale(d["_dist"])
    n = sum(d[c] * w for c, w in NEED_WEIGHTS.items()).round(2)
    p = sum(d[c] * w for c, w in PROVISION_WEIGHTS.items()).round(2)
    d["multiplicative"] = (n * p / 100).round(2)
    d["equal_sum"] = 0.5 * n + 0.5 * p
    d["need_emphasis"] = 0.6 * n + 0.4 * p
    d["provision_emphasis"] = 0.4 * n + 0.6 * p
    return d


METHODS = ["multiplicative", "equal_sum", "need_emphasis", "provision_emphasis"]
SINGLE = ["patients_per_gp", "deprivation_score", "nearest_gp_distance_km"]
rows = []
for seed in range(1, 101):
    data = generate_dataset(seed)
    for setting, options in [("Min-max", {}),
                             ("Log + within rurality",
                              {"log_ppg": True, "by_rurality": True})]:
        d = score_areas(data, **options)
        columns = METHODS + (SINGLE if setting == "Min-max" else [])
        for method in columns:
            rows.append({"seed": seed, "setting": setting, "method": method,
                         "f1": evaluate_ranking(d, method)["F1 score"]})

results = pd.DataFrame(rows)
print(results.groupby(["setting", "method"])["f1"].agg(["mean", "std"]).round(2))

table = results.pivot_table(index="seed", columns=["setting", "method"], values="f1")
for setting in ["Min-max", "Log + within rurality"]:
    a = table[(setting, "need_emphasis")]
    b = table[(setting, "multiplicative")]
    print(setting, "need-emphasis vs multiplicative:",
          "better", int((a > b).sum()), "tied", int((a == b).sum()),
          "worse", int((a < b).sum()), "Wilcoxon p = %.2g" % wilcoxon(a, b).pvalue)
# Expected: Min-max means 71.80, 76.53, 79.60, 70.67; 89 better, 7 tied, 4 worse.
# Log + within rurality means 87.37, 87.57, 87.70, 85.70; p = 0.22 (a tie).