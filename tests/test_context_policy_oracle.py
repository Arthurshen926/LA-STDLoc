from map_learning.context_policy_oracle import (
    pose_policy_loss,
    summarize_policy_oracle,
)


def _row(query_index, seed, a1_te, context_te):
    return {
        "query_index": query_index,
        "image_name": f"q{query_index}",
        "direction": "fold",
        "seed": seed,
        "a1_te_cm": a1_te,
        "a1_ae_deg": a1_te / 2,
        "a1_hypotheses": 1000,
        "a1_failed": False,
        "context_lambda_0p01_te_cm": context_te,
        "context_lambda_0p01_ae_deg": context_te / 2,
        "context_lambda_0p01_hypotheses": 900,
        "context_lambda_0p01_failed": False,
    }


def test_pose_policy_loss_penalizes_pose_tail_and_solver_work():
    clean = pose_policy_loss(te_cm=1, ae_deg=1, hypotheses=100)
    expensive = pose_policy_loss(te_cm=1, ae_deg=1, hypotheses=10000)
    catastrophic = pose_policy_loss(te_cm=101, ae_deg=1, hypotheses=100)

    assert clean < expensive < catastrophic


def test_policy_oracle_chooses_per_query_strategy_and_has_headroom():
    rows = []
    for seed in (1, 2):
        rows.append(_row(0, seed, a1_te=1, context_te=3))
        rows.append(_row(1, seed, a1_te=4, context_te=1))

    report = summarize_policy_oracle(
        rows,
        context_protocol="context_lambda_0p01",
        seeds=(1, 2),
        bootstrap_samples=100,
    )

    assert report["oracle_policy_counts"] == {
        "a1": 1,
        "context_lambda_0p01": 1,
    }
    assert report["oracle_risk"] < min(report["fixed_policy_risk"].values())
    assert report["oracle_headroom_bootstrap_95ci"][0] >= 0.0
    assert report["pose"]["oracle"]["mean_te_cm"] == 1.0
