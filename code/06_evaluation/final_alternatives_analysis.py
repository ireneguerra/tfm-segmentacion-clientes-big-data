"""Reproducible comparison of the alternative final clustering solutions.

The script deliberately writes to results/final_alternatives and never modifies
the existing multimodel or profiles results.  Validation metrics are read from
the existing common-sample study; only the full-data partitions are fitted
again because those assignments were not persisted by the original study.
"""
from pathlib import Path
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "03_modeling"))
import modeling_functions as mf

OUT = ROOT / "results" / "final_alternatives"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

CONFIGS = [
    dict(id="hym_fuzzy_k5", dataset="hym", model="fuzzy", k=5,
         features=["channel_2_ratio", "std_days_between_purchases", "avg_days_between_purchases"]),
    dict(id="hym_kmeans_k5", dataset="hym", model="kmeans", k=5,
         features=["channel_2_ratio", "std_days_between_purchases", "avg_days_between_purchases"]),
    dict(id="hym_kmeans_k6", dataset="hym", model="kmeans", k=6,
         features=["channel_2_ratio", "std_days_between_purchases", "avg_days_between_purchases"]),
    dict(id="instacart_fuzzy_k2_std", dataset="instacart", model="fuzzy", k=2,
         features=["days_since_previous_order", "frequency", "avg_days_between_orders", "std_days_between_orders"]),
    dict(id="instacart_fuzzy_k2_simple", dataset="instacart", model="fuzzy", k=2,
         features=["days_since_previous_order", "frequency", "avg_days_between_orders"]),
    dict(id="instacart_kmeans_k2_std", dataset="instacart", model="kmeans", k=2,
         features=["days_since_previous_order", "frequency", "avg_days_between_orders", "std_days_between_orders"]),
]

SCALED = {"hym": ROOT / "scaled_data" / "hym_scaled.csv",
          "instacart": ROOT / "scaled_data" / "instacart_scaled.csv"}
ORIGINAL = {"hym": ROOT / "extended_datasets" / "hym_model_extended.csv",
            "instacart": ROOT / "extended_datasets" / "instacart_model_extended.csv"}
ID = {"hym": "customer_id", "instacart": "user_id"}


def verify_and_filter_validation():
    """Keep a traceable copy of only the requested rows from the old detail CSV."""
    detail_path = ROOT / "results" / "multimodel" / "multi_model_feature_validation_detail.csv"
    summary_path = ROOT / "results" / "multimodel" / "multi_model_feature_validation.csv"
    detail = pd.read_csv(detail_path)
    summary = pd.read_csv(summary_path)
    requested = pd.DataFrame([{**c, "columns": " | ".join(c["features"]), "n_features": len(c["features"])} for c in CONFIGS])
    key = ["dataset", "model", "k", "columns"]
    filtered_detail = detail.merge(requested[key], on=key, how="inner")
    filtered_summary = summary.merge(requested[key], on=key, how="inner")
    filtered_detail.to_csv(OUT / "validation_detail_requested.csv", index=False)
    filtered_summary.to_csv(OUT / "validation_metrics_requested.csv", index=False)
    # Explicit audit of missing requested configurations.
    found = set(map(tuple, filtered_summary[key].astype(str).to_numpy()))
    audit = requested[key].copy()
    audit["found_in_existing_summary"] = [tuple(map(str, row)) in found for row in audit[key].to_numpy()]
    audit.to_csv(OUT / "requested_configuration_audit.csv", index=False)
    return filtered_summary, audit


def sensitivity(summary):
    """Recompute the original percentile score with ARI weights 0, .10 and .25."""
    rows = []
    for dataset, group in summary.groupby("dataset", sort=False):
        group = group.copy()
        group["pct_silhouette"] = group["silhouette"].rank(pct=True, ascending=True)
        group["pct_calinski"] = group["calinski_harabasz"].rank(pct=True, ascending=True)
        group["pct_davies"] = group["davies_bouldin"].rank(pct=True, ascending=False)
        group["pct_stability"] = group["stability_ari"].rank(pct=True, ascending=True)
        for w in [0.0, 0.10, 0.25]:
            group["ari_weight"] = w
            group["score_weighted"] = 100 * ((1-w) * (group.pct_silhouette + group.pct_calinski + group.pct_davies) / 3 + w * group.pct_stability)
            group["position_weighted"] = group["score_weighted"].rank(method="min", ascending=False).astype(int)
            rows.append(group.copy())
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["dataset", "ari_weight", "position_weighted", "stability_ari", "silhouette"], ascending=[True, True, True, False, False])
    result.to_csv(OUT / "ranking_sensitivity_ari_weights.csv", index=False)
    requested = result.merge(pd.DataFrame([{**c, "columns": " | ".join(c["features"])} for c in CONFIGS]), on=["dataset", "model", "k", "columns"], how="inner")
    requested.to_csv(OUT / "ranking_sensitivity_requested.csv", index=False)
    result.groupby(["dataset", "ari_weight"], as_index=False).head(10).to_csv(OUT / "ranking_sensitivity_top10.csv", index=False)


def fit_full_config(config, scaled, original):
    features = config["features"]
    X = scaled[features].to_numpy(dtype=np.float64)
    start = time.perf_counter()
    mf.RANDOM_STATE = SEED
    model, labels = mf.fit_clustering_model(config["model"], config["k"], X)
    elapsed = time.perf_counter() - start
    labels = np.asarray(labels, dtype=int)
    assign = pd.DataFrame({ID[config["dataset"]]: scaled[ID[config["dataset"]]].to_numpy(), "cluster": labels})
    assign["configuration_id"] = config["id"]
    assign.to_csv(OUT / f"assignments_{config['id']}.csv", index=False)
    joined = original.merge(assign, on=ID[config["dataset"]], how="inner", validate="one_to_one")
    desc_cols = [c for c in original.columns if c != ID[config["dataset"]]]
    means = joined.groupby("cluster")[desc_cols].mean(numeric_only=True).add_prefix("mean_")
    medians = joined.groupby("cluster")[desc_cols].median(numeric_only=True).add_prefix("median_")
    sizes = joined.groupby("cluster").size().rename("size")
    profile = pd.concat([sizes, (100*sizes/sizes.sum()).rename("pct"), means, medians], axis=1).reset_index()
    profile.insert(0, "configuration_id", config["id"])
    profile.insert(1, "dataset", config["dataset"])
    profile["model"] = config["model"]
    profile["k"] = config["k"]
    profile["features_used_for_training"] = " | ".join(features)
    profile["profile_variables"] = " | ".join(desc_cols)
    profile["seed"] = SEED
    profile["n_customers"] = len(joined)
    profile["fit_seconds"] = elapsed
    profile["fuzzy_label_rule"] = "argmax membership" if config["model"] == "fuzzy" else "not applicable"
    profile.to_csv(OUT / f"profiles_{config['id']}.csv", index=False)
    return config, assign, profile, joined, elapsed


def concordance(assignments):
    records = []
    tables = []
    for dataset in ["hym", "instacart"]:
        dataset_assignments = [x for x in assignments if x.attrs.get("dataset") == dataset]
        for a, b in combinations(dataset_assignments, 2):
            ca, cb = a["configuration_id"].iloc[0], b["configuration_id"].iloc[0]
            left = a.drop(columns="configuration_id").rename(columns={"cluster": "cluster_a"})
            right = b.drop(columns="configuration_id").rename(columns={"cluster": "cluster_b"})
            id_col = ID[dataset]
            merged = left.merge(right, on=id_col, validate="one_to_one")
            records.append({"dataset": dataset, "configuration_a": ca, "configuration_b": cb,
                            "n_customers": len(merged), "ari_between_configurations": adjusted_rand_score(merged.cluster_a, merged.cluster_b)})
            ct = pd.crosstab(merged.cluster_a, merged.cluster_b)
            long = ct.stack().rename("count").reset_index()
            long.columns = ["cluster_a", "cluster_b", "count"]
            long["pct_of_configuration_a"] = long["count"] / long.groupby("cluster_a")["count"].transform("sum") * 100
            long.insert(0, "configuration_a", ca); long.insert(1, "configuration_b", cb); long.insert(0, "dataset", dataset)
            tables.append(long)
    pd.DataFrame(records).to_csv(OUT / "partition_concordance_ari.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(OUT / "partition_contingency_tables.csv", index=False)


def interval_quality(joined_results):
    rows = []
    for config, _, _, joined, _ in joined_results:
        if config["dataset"] == "hym":
            freq, avg, std = "frequency", "avg_days_between_purchases", "std_days_between_purchases"
        else:
            freq, avg, std = "frequency", "avg_days_between_orders", "std_days_between_orders"
        for cluster, g in joined.groupby("cluster"):
            rows.append({"dataset": config["dataset"], "configuration_id": config["id"], "cluster": cluster,
                         "n_customers": len(g), "pct_frequency_le_2": 100*(g[freq] <= 2).mean(),
                         "pct_interval_zero": 100*(g[avg] == 0).mean(),
                         "pct_interval_std_zero": 100*(g[std] == 0).mean(),
                         "note": "Zeros are reported descriptively; imputation source cannot be inferred from zero alone."})
    pd.DataFrame(rows).to_csv(OUT / "interval_quality_by_cluster.csv", index=False)


def figures(profiles, concordance_df):
    for dataset in ["hym", "instacart"]:
        subset = profiles[profiles.dataset == dataset]
        plt.figure(figsize=(10, 5)); subset.pivot(index="configuration_id", columns="cluster", values="pct").plot(kind="bar", ax=plt.gca())
        plt.ylabel("Porcentaje de clientes"); plt.title(f"Tamaño de grupos — {dataset.upper()}"); plt.tight_layout(); plt.savefig(OUT / f"group_sizes_{dataset}.png", dpi=160); plt.close()
        plt.figure(figsize=(10, 5)); d = concordance_df[concordance_df.dataset == dataset]
        if not d.empty:
            d.pivot(index="configuration_a", columns="configuration_b", values="ari_between_configurations").plot(kind="bar", ax=plt.gca(), legend=False)
        plt.ylabel("ARI entre configuraciones"); plt.title(f"Concordancia — {dataset.upper()}"); plt.tight_layout(); plt.savefig(OUT / f"concordance_{dataset}.png", dpi=160); plt.close()


def main():
    summary, audit = verify_and_filter_validation()
    # La sensibilidad debe conservar el universo del ranking original:
    # 175 configuraciones por dataset. El CSV filtrado se usa únicamente
    # para exportar la vista de las seis alternativas solicitadas.
    full_summary = pd.read_csv(ROOT / "results" / "multimodel" / "multi_model_feature_validation.csv")
    sensitivity(full_summary)
    if not audit.found_in_existing_summary.all():
        raise RuntimeError("Faltan configuraciones solicitadas en los resultados existentes; revise requested_configuration_audit.csv")
    results, assignments = [], []
    loaded = {}
    for dataset in ["hym", "instacart"]:
        needed = sorted({ID[dataset], *[f for c in CONFIGS if c["dataset"] == dataset for f in c["features"]]})
        scaled = pd.read_csv(SCALED[dataset], usecols=needed)
        original = pd.read_csv(ORIGINAL[dataset])
        loaded[dataset] = (scaled, original)
    for config in CONFIGS:
        scaled, original = loaded[config["dataset"]]
        config, assign, profile, joined, elapsed = fit_full_config(config, scaled, original)
        assign.attrs["dataset"] = config["dataset"]
        assignments.append(assign)
        results.append((config, assign, profile, joined, elapsed))
    all_profiles = pd.concat([r[2] for r in results], ignore_index=True)
    all_profiles.to_csv(OUT / "profiles_all_alternatives.csv", index=False)
    concordance(assignments)
    ari = pd.read_csv(OUT / "partition_concordance_ari.csv")
    interval_quality(results)
    figures(all_profiles, ari)
    with open(OUT / "README.md", "w", encoding="utf-8") as f:
        f.write("# Comparación de alternativas finales\n\n")
        f.write("Métricas de validación: muestra común y resultados existentes. Perfiles y asignaciones: ajuste completo, semilla 42. Fuzzy usa argmax de pertenencias. ARI de concordancia compara configuraciones, no semillas.\n")
    print(f"Resultados generados en {OUT}")


if __name__ == "__main__":
    main()
