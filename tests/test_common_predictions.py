import pandas as pd

from funmirbench.common_predictions import build_common_prediction_summary


def test_common_prediction_summary_excludes_controls_and_reports_common_sets():
    joined = pd.DataFrame(
        {
            "gene_id": ["g1", "g2", "g3", "g4"],
            "score_targetscan": [1.0, 0.8, None, None],
            "score_mirdb_mirtarget": [0.9, None, 0.4, None],
            "score_tec-mitarget": [None, None, None, 0.7],
            "score_random": [0.1, 0.2, 0.3, 0.4],
            "score_perfect": [1.0, 1.0, 0.0, 0.0],
        }
    )

    summary = build_common_prediction_summary(
        joined,
        dataset_id="D001",
        tool_ids=["targetscan", "mirdb_mirtarget", "tec-mitarget", "random", "perfect"],
        publication_min_common_coverage=0.25,
    )

    assert "random" not in set(summary["tools"])
    assert "perfect" not in set(summary["tools"])

    singles = summary[summary["summary_type"] == "single_predictor"]
    assert set(singles["tools"]) == {"targetscan", "mirdb_mirtarget", "tec-mitarget"}

    publication_common = summary[summary["summary_type"] == "publication_common_set"].iloc[0]
    assert publication_common["tools"] == "targetscan,mirdb_mirtarget,tec-mitarget"
    assert publication_common["rows_common"] == 0
    assert publication_common["percent_common"] == 0.0

    pair = summary[
        (summary["summary_type"] == "pairwise_common_set")
        & (summary["tools"] == "targetscan,mirdb_mirtarget")
    ].iloc[0]
    assert pair["rows_common"] == 1
    assert pair["percent_common"] == 0.25
