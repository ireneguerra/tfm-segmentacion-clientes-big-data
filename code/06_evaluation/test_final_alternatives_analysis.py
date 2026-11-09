import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).with_name("final_alternatives_analysis.py")
spec = importlib.util.spec_from_file_location("final_alternatives_analysis", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _tables():
    key = dict(dataset="hym", subset="s", columns="a | b | c", model="kmeans", k=3)
    detail = pd.DataFrame([
        {**key, "seed": seed, "error": "", "silhouette": 0.1 * seed / 42,
         "calinski_harabasz": float(seed), "davies_bouldin": 2.0}
        for seed in (42, 123, 2026)
    ])
    summary = pd.DataFrame([{**key, "silhouette": detail.silhouette.mean(),
                             "calinski_harabasz": detail.calinski_harabasz.mean(),
                             "davies_bouldin": 2.0, "stability_ari": 0.5,
                             "pct_silhouette": 1.0, "pct_calinski": 1.0,
                             "pct_davies": 1.0, "pct_stability": 1.0}])
    return summary, detail


def test_summary_detail_means_are_checked():
    summary, detail = _tables()
    module.validate_summary_detail(summary, detail)
    summary.loc[0, "silhouette"] += 0.01
    with pytest.raises(RuntimeError, match="Summary/detail"):
        module.validate_summary_detail(summary, detail)


def test_nonfinite_and_seed_duplicates_are_rejected():
    summary, detail = _tables()
    detail.loc[0, "silhouette"] = np.inf
    with pytest.raises(RuntimeError, match="no certificado"):
        module.validate_summary_detail(summary, detail)
    summary, detail = _tables()
    detail = detail.loc[detail.seed.ne(123)].copy()
    with pytest.raises(RuntimeError, match="no certificado"):
        module.validate_summary_detail(summary, detail)
    summary, detail = _tables()
    detail.loc[1, "seed"] = 42
    with pytest.raises(RuntimeError, match="no certificado"):
        module.validate_summary_detail(summary, detail)


def test_percentile_universe_and_order_are_checked():
    summary, _ = _tables()
    module.validate_percentiles(summary)
    summary.loc[0, "pct_davies"] = 0.0
    with pytest.raises(RuntimeError, match="percentiles"):
        module.validate_percentiles(summary)
