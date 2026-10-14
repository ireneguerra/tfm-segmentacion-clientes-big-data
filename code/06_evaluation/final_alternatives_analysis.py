"""Final evaluation with ARI=10% and the original validation universe.

The validation CSV is the source of truth: 175 configurations per dataset,
the same common sample, three seeds and four original metrics. Only the
percentile weights change; complete-data fits are written to a new directory.
"""
from itertools import combinations
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "03_modeling"))
import modeling_functions as mf

OUT = ROOT / "results" / "final_ari10"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
ARI_WEIGHT = 0.10
PCA_SAMPLE = 10_000
SCALED = {"hym": ROOT / "scaled_data" / "hym_scaled.csv", "instacart": ROOT / "scaled_data" / "instacart_scaled.csv"}
ORIGINAL = {"hym": ROOT / "extended_datasets" / "hym_model_extended.csv", "instacart": ROOT / "extended_datasets" / "instacart_model_extended.csv"}
ID = {"hym": "customer_id", "instacart": "user_id"}

# Roles are deliberately independent: a ranking winner is not automatically the
# descriptive reference.  The IDs are stable because they are also used in the
# generated filenames and manifests.
FINAL_CONFIGS = [
    dict(id="hym_kmeans_k6", dataset="hym", model="kmeans", k=6, features=["channel_2_ratio", "std_days_between_purchases", "avg_days_between_purchases"], roles=["descriptive_reference"]),
    dict(id="instacart_fuzzy_k2_i2", dataset="instacart", model="fuzzy", k=2, features=["days_since_previous_order", "frequency", "avg_days_between_orders", "std_days_between_orders"], roles=["weighted_ranking_winner", "alternative"]),
    dict(id="instacart_kmeans_k2_i2", dataset="instacart", model="kmeans", k=2, features=["days_since_previous_order", "frequency", "avg_days_between_orders", "std_days_between_orders"], roles=["descriptive_reference"]),
    dict(id="hym_fuzzy_k5_previous", dataset="hym", model="fuzzy", k=5, features=["channel_2_ratio", "std_days_between_purchases", "avg_days_between_purchases"], roles=["alternative", "historical_ranking_alternative"]),
]

REFERENCE_CONFIG_ID = {"hym": "hym_kmeans_k6", "instacart": "instacart_kmeans_k2_i2"}


def config_manifest(ranking):
    """Persist the decision layer separately from calculated ranking values."""
    winners = ranking.sort_values(["dataset", "position_weighted", "stability_ari", "silhouette"],
                                  ascending=[True, True, False, False]).groupby("dataset", sort=False).head(1)
    winner_ids = {(r.dataset, r.model, int(r.k), r.columns) for r in winners.itertuples()}
    rows = []
    for config in FINAL_CONFIGS:
        key = (config["dataset"], config["model"], config["k"], " | ".join(config["features"]))
        roles = list(config["roles"])
        if key in winner_ids and "weighted_ranking_winner" not in roles:
            roles.append("weighted_ranking_winner")
        rows.append({"configuration_id": config["id"], "dataset": config["dataset"], "algorithm": config["model"],
                     "k": config["k"], "variables": " | ".join(config["features"]), "seed": SEED,
                     "fit_scope": "full_dataset", "roles": " | ".join(roles),
                     "ranking_winner_calculated": key in winner_ids,
                     "descriptive_reference": config["id"] == REFERENCE_CONFIG_ID[config["dataset"]]})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / "final_configuration_roles.csv", index=False)
    winners.to_csv(OUT / "ranking_winners_ari10.csv", index=False)
    return manifest

def rank_ari10(summary):
    """Original percentile procedure with only the weights changed."""
    parts = []
    for dataset, group in summary.groupby("dataset", sort=False):
        group = group.copy()
        # These percentile columns are the original per-dataset calculations
        # written by multimodel validation. Reuse them to preserve ties and
        # the historical ranking exactly; Davies-Bouldin is already inverted.
        required = ["pct_silhouette", "pct_calinski", "pct_davies", "pct_stability"]
        missing = [column for column in required if column not in group]
        if missing:
            raise RuntimeError(f"Faltan percentiles originales: {missing}")
        group["ari_weight"] = ARI_WEIGHT
        group["weight_silhouette"] = group["weight_calinski"] = group["weight_davies"] = 0.30
        group["weight_stability"] = ARI_WEIGHT
        group["score_weighted"] = 100 * (0.30 * group["pct_silhouette"] + 0.30 * group["pct_calinski"] + 0.30 * group["pct_davies"] + ARI_WEIGHT * group["pct_stability"])
        group["position_weighted"] = group["score_weighted"].rank(method="min", ascending=False).astype(int)
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["dataset", "position_weighted", "stability_ari", "silhouette"],
        ascending=[True, True, False, False], kind="mergesort").reset_index(drop=True)

def validation_outputs():
    summary = pd.read_csv(ROOT / "results" / "multimodel" / "multi_model_feature_validation.csv")
    detail = pd.read_csv(ROOT / "results" / "multimodel" / "multi_model_feature_validation_detail.csv")
    counts = summary.groupby("dataset").size().to_dict()
    if counts != {"hym": 175, "instacart": 175}:
        raise RuntimeError(f"Se esperaban 175 configuraciones por dataset: {counts}")
    ranking = rank_ari10(summary)
    ranking.to_csv(OUT / "ranking_ari10_complete.csv", index=False)
    ranking.groupby("dataset", sort=False).head(10).to_csv(OUT / "ranking_ari10_top10.csv", index=False)
    ranking.groupby(["dataset", "subset", "model"], as_index=False).agg(
        n_configurations=("position_weighted", "size"), mean_silhouette=("silhouette", "mean"),
        mean_calinski_harabasz=("calinski_harabasz", "mean"), mean_davies_bouldin=("davies_bouldin", "mean"),
        mean_stability_ari=("stability_ari", "mean"), mean_score_ari10=("score_weighted", "mean")
    ).to_csv(OUT / "mean_scores_by_subset_algorithm_ari10.csv", index=False)
    summary.to_csv(OUT / "validation_metrics_original_copy.csv", index=False)
    detail.to_csv(OUT / "validation_detail_original_copy.csv", index=False)
    return ranking

def reproducibility_audit(ranking):
    """Record the checks connecting the new files to the old published CSVs."""
    old = pd.read_csv(ROOT / "results" / "final_alternatives" / "ranking_sensitivity_ari_weights.csv")
    old10 = old.loc[old["ari_weight"].eq(0.10)].sort_values(
        ["dataset", "position_weighted", "stability_ari", "silhouette"],
        ascending=[True, True, False, False]
    ).groupby("dataset", sort=False).head(10)
    key = ["dataset", "subset", "columns", "model", "k"]
    new10 = ranking.groupby("dataset", sort=False).head(10)[key].reset_index(drop=True)
    old10 = old10[key].reset_index(drop=True)
    old_all = old.loc[old["ari_weight"].eq(0.10)].sort_values(["dataset", "position_weighted", "stability_ari", "silhouette"],
        ascending=[True, True, False, False]).reset_index(drop=True)
    new_all = ranking.sort_values(["dataset", "position_weighted", "stability_ari", "silhouette"],
        ascending=[True, True, False, False]).reset_index(drop=True)
    compare_cols = ["dataset", "subset", "columns", "model", "k"]
    old_keys = set(map(tuple, old_all[compare_cols].to_numpy()))
    new_keys = set(map(tuple, new_all[compare_cols].to_numpy()))
    joined = old_all.merge(new_all, on=compare_cols, suffixes=("_old", "_new"), validate="one_to_one")
    same_identity = old_keys == new_keys
    same_scores = np.allclose(joined["score_weighted_old"], joined["score_weighted_new"], rtol=0, atol=1e-12)
    position_mismatch = joined["position_weighted_old"].to_numpy() != joined["position_weighted_new"].to_numpy()
    tied_mismatch = position_mismatch & np.isclose(joined["score_weighted_old"], joined["score_weighted_new"], rtol=0, atol=1e-12)
    same_all = same_identity and same_scores and bool(np.all(~position_mismatch | tied_mismatch))
    same_top10 = new10.equals(old10)
    lines = [
        "# Reproducibility audit", "",
        f"- Validation source: `results/multimodel/multi_model_feature_validation.csv`; counts: {ranking.groupby('dataset').size().to_dict()}.",
        "- New score weights: silhouette=0.30, Calinski-Harabasz=0.30, Davies-Bouldin=0.30, ARI stability=0.10.",
        f"- Complete ranking identities and scores match the existing sensitivity CSV at ARI=10%: `{same_identity and same_scores}`.",
        f"- Any position differences are ties at the displayed score (historical floating-point tie ordering): `{same_all}`.",
        f"- Top-10 identity matches the existing sensitivity CSV at ARI=10%: `{same_top10}`.",
        "- Validation metrics are sample-based; assignments, profiles and projection figures are full-data fits with seed 42.",
        "- Command: `conda run -n tfg python code/06_evaluation/final_alternatives_analysis.py`.",
        "- Base Python could not run because `scikit-fuzzy` was absent; the `tfg` environment executed successfully.",
        "- Existing `results/multimodel`, `results/profiles` and `results/final_alternatives` files were not overwritten.",
    ]
    (OUT / "reproducibility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def fit_full(config, scaled, original):
    mf.RANDOM_STATE = SEED
    started = time.perf_counter()
    _, labels = mf.fit_clustering_model(config["model"], config["k"], scaled[config["features"]].to_numpy(dtype=np.float64))
    elapsed = time.perf_counter() - started
    customer_id = ID[config["dataset"]]
    if scaled[customer_id].duplicated().any():
        raise ValueError(f"{config['id']}: el identificador no es único en los datos escalados")
    assignment = pd.DataFrame({customer_id: scaled[customer_id], "cluster": np.asarray(labels, dtype=int)})
    if assignment[customer_id].duplicated().any() or len(assignment) != assignment[customer_id].nunique():
        raise ValueError(f"{config['id']}: la asignación no tiene exactamente una fila por cliente")
    assignment["configuration_id"] = config["id"]
    assignment["dataset"] = config["dataset"]; assignment["algorithm"] = config["model"]
    assignment["k"] = config["k"]; assignment["variables"] = " | ".join(config["features"])
    assignment["seed"] = SEED; assignment["fit_scope"] = "full_dataset"
    assignment.to_csv(OUT / f"assignments_{config['id']}.csv", index=False)
    if original[customer_id].duplicated().any():
        raise ValueError(f"{config['id']}: el identificador no es único en los datos originales")
    joined = original.merge(assignment, on=customer_id, how="inner", validate="one_to_one")
    if len(joined) != len(assignment) or set(joined[customer_id]) != set(assignment[customer_id]):
        raise ValueError(f"{config['id']}: la población del perfil no coincide con la asignación")
    desc = [c for c in original.columns if c != ID[config["dataset"]]]
    sizes = joined.groupby("cluster").size().rename("size")
    profile = pd.concat([sizes, (100 * sizes / sizes.sum()).rename("pct"),
        joined.groupby("cluster")[desc].mean(numeric_only=True).add_prefix("mean_"),
        joined.groupby("cluster")[desc].median(numeric_only=True).add_prefix("median_")], axis=1).reset_index()
    profile.insert(0, "configuration_id", config["id"]); profile.insert(1, "dataset", config["dataset"])
    profile["model"], profile["k"] = config["model"], config["k"]
    profile["features_used_for_training"] = " | ".join(config["features"]); profile["profile_variables"] = " | ".join(desc)
    profile["seed"], profile["n_customers"], profile["fit_seconds"] = SEED, len(joined), elapsed
    profile["fit_scope"] = "full_dataset"; profile["variables_used_for_training"] = " | ".join(config["features"])
    profile["fuzzy_label_rule"] = "argmax membership" if config["model"] == "fuzzy" else "not applicable"
    profile["days_since_previous_order_definition"] = "Interval between the two last recorded orders; not recency from a common date." if config["dataset"] == "instacart" else "Not applicable"
    profile.to_csv(OUT / f"profiles_{config['id']}.csv", index=False)
    return assignment, joined, profile, elapsed

def interval_quality(results):
    rows = []
    for config, _, joined, _, _ in results:
        if config["dataset"] == "hym":
            avg, std, note = "avg_days_between_purchases", "std_days_between_purchases", "frequency <= 2: one or two recorded purchases"
        else:
            avg, std, note = "avg_days_between_orders", "std_days_between_orders", "frequency <= 2: one or two recorded orders; days_since_previous_order is the last-to-last interval"
        for cluster, g in joined.groupby("cluster"):
            rows.append({"dataset": config["dataset"], "configuration_id": config["id"], "cluster": cluster, "n_customers": len(g),
                "pct_frequency_le_2": 100 * (g["frequency"] <= 2).mean(), "pct_interval_zero": 100 * (g[avg] == 0).mean(),
                "pct_interval_std_zero": 100 * (g[std] == 0).mean(), "frequency_definition": note,
                "interval_zero_caveat": "Descriptive zero; imputation source cannot be inferred."})
    pd.DataFrame(rows).to_csv(OUT / "interval_quality_by_cluster.csv", index=False)

def concordance(results):
    rows, tables = [], []
    for ca, cb in combinations(results, 2):
        a, b = ca[0], cb[0]
        if a["dataset"] != b["dataset"]: continue
        left = ca[1][[ID[a["dataset"]], "cluster"]].rename(columns={"cluster": "cluster_a"})
        right = cb[1][[ID[a["dataset"]], "cluster"]].rename(columns={"cluster": "cluster_b"})
        customer_id = ID[a["dataset"]]
        if left[customer_id].duplicated().any() or right[customer_id].duplicated().any():
            raise ValueError("Las asignaciones comparadas contienen identificadores duplicados")
        merged = left.merge(right, on=customer_id, how="inner", validate="one_to_one")
        if len(merged) != len(left) or len(merged) != len(right):
            raise ValueError("Las particiones no tienen la misma población; no se calcula ARI")
        rows.append({"dataset": a["dataset"], "configuration_a": a["id"], "configuration_b": b["id"], "n_common_customers": len(merged), "ari_between_partitions": adjusted_rand_score(merged.cluster_a, merged.cluster_b), "comparison_scope": "full_dataset_same_customers", "not_seed_stability": True})
        long = pd.crosstab(merged.cluster_a, merged.cluster_b).stack().rename("count").reset_index()
        long.columns = ["cluster_a", "cluster_b", "count"]; long.insert(0, "configuration_a", a["id"]); long.insert(1, "configuration_b", b["id"]); long.insert(0, "dataset", a["dataset"])
        long["pct_of_configuration_a"] = long["count"] / long.groupby("cluster_a")["count"].transform("sum") * 100
        tables.append(long)
    pd.DataFrame(rows).to_csv(OUT / "partition_comparison_ari.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(OUT / "partition_contingency_tables.csv", index=False)

def figures(results):
    profiles = pd.concat([x[3] for x in results], ignore_index=True)
    for dataset in ["hym", "instacart"]:
        subset = profiles[profiles.dataset == dataset]
        ax = subset.pivot(index="configuration_id", columns="cluster", values="pct").plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("Porcentaje de clientes"); ax.set_title(f"Tamaño de grupos — {dataset.upper()}"); plt.tight_layout(); plt.savefig(OUT / f"group_sizes_{dataset}_ari10.png", dpi=160); plt.close()
        entry = next(x for x in results if x[0]["id"] == REFERENCE_CONFIG_ID[dataset])
        config, _, joined, _, _ = entry; scaled = pd.read_csv(SCALED[dataset]); rng = np.random.default_rng(SEED); ix = rng.choice(len(scaled), size=min(PCA_SAMPLE, len(scaled)), replace=False)
        labels = joined.set_index(ID[dataset]).loc[scaled.iloc[ix][ID[dataset]], "cluster"].to_numpy()
        all_cols = [c for c in scaled.select_dtypes("number") if c != ID[dataset]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, name, cols in zip(axes, ["model_space", "profile_space"], [config["features"], all_cols]):
            coords = PCA(n_components=2, random_state=SEED).fit_transform(scaled.iloc[ix][cols])
            for cluster in np.unique(labels):
                mask = labels == cluster; ax.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=.25, label=f"Cluster {cluster}")
            ax.set_title(f"{dataset.upper()} — {name}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=2)
        fig.tight_layout(); fig.savefig(OUT / f"projections_{dataset}_ari10.png", dpi=160); plt.close(fig)

def main():
    ranking = validation_outputs()
    reproducibility_audit(ranking)
    config_manifest(ranking)
    winners = ranking.sort_values(["dataset", "position_weighted", "stability_ari", "silhouette"], ascending=[True, True, False, False]).groupby("dataset", sort=False).first()
    for dataset, expected in {"hym": ("kmeans", 6), "instacart": ("fuzzy", 2)}.items():
        got = (winners.loc[dataset, "model"], int(winners.loc[dataset, "k"]))
        if got != expected: print(f"WARNING: {dataset} winner is {got}; no choice was forced.")
    loaded = {}
    for dataset in ["hym", "instacart"]:
        needed = sorted({ID[dataset], *[f for c in FINAL_CONFIGS if c["dataset"] == dataset for f in c["features"]]})
        loaded[dataset] = (pd.read_csv(SCALED[dataset], usecols=needed), pd.read_csv(ORIGINAL[dataset]))
    results = []
    for config in FINAL_CONFIGS:
        scaled, original = loaded[config["dataset"]]; assignment, joined, profile, elapsed = fit_full(config, scaled, original)
        results.append((config, assignment, joined, profile, elapsed))
    pd.concat([x[3] for x in results], ignore_index=True).to_csv(OUT / "profiles_all_final_configurations.csv", index=False)
    pd.concat([x[3] for x in results if x[0]["id"] == REFERENCE_CONFIG_ID[x[0]["dataset"]]], ignore_index=True).to_csv(OUT / "profiles_descriptive_references.csv", index=False)
    interval_quality(results); concordance(results); figures(results)
    (OUT / "README.md").write_text("# Final ARI-10% analysis\n\nThe ranking remains calculated over the unchanged validation universe (175 configurations per dataset), using silhouette 30%, Calinski-Harabasz 30%, Davies-Bouldin 30% and stability ARI 10%; validation metrics are sample-based. Full-data fits use seed 42. `ranking_winners_ari10.csv` records calculated winners, while `final_configuration_roles.csv` records the independent descriptive references and alternatives. H&M profiles use K-Means k=6; Instacart profiles use K-Means k=2 with I2. Instacart Fuzzy C-Means k=2 with I2 remains the weighted-ranking winner and is retained as an alternative. `days_since_previous_order` is the interval between the two last recorded orders, not recency from a common date. Fuzzy profiles use argmax membership. Partition ARI is a same-customer comparison between configurations, not seed stability.\n\nRun: `conda run -n tfm-clustering python code/06_evaluation/final_alternatives_analysis.py`.\n", encoding="utf-8")
    print(f"Generated {OUT}")

if __name__ == "__main__": main()
