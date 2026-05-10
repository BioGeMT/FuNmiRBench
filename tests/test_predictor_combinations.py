import pandas as pd

from funmirbench.predictor_combinations import compute_predictor_combination_summary


def test_combination_summary_excludes_oracles_and_scores_rank_mean():
    joined = pd.DataFrame(
        {
            "gene_id": ["g1", "g2", "g3", "g4"],
            "logFC": [-2.0, -1.5, 0.2, 0.1],
            "FDR": [0.01, 0.02, 0.5, 0.8],
            "score_targetscan": [0.9, 0.8, 0.2, 0.1],
            "score_mirdb_mirtarget": [0.85, 0.7, 0.3, 0.2],
            "score_cheating": [1.0, 1.0, 0.0, 0.0],
            "score_random": [0.1, 0.8, 0.4, 0.6],
        }
    )

    summary = compute_predictor_combination_summary(
        [joined],
        tool_ids=["targetscan", "mirdb_mirtarget", "cheating", "random"],
        fdr_threshold=0.05,
        abs_logfc_threshold=1.0,
        max_combination_size=2,
    )

    assert set(summary["combination_id"]) == {
        "targetscan",
        "mirdb_mirtarget",
        "targetscan+mirdb_mirtarget",
    }
    combo = summary[summary["combination_id"] == "targetscan+mirdb_mirtarget"].iloc[0]
    assert combo["combination_size"] == 2
    assert combo["dataset_count"] == 1
    assert combo["aps_mean"] == 1.0
    assert combo["positive_coverage_mean"] == 1.0
