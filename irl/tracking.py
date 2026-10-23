"""
MLflow tracking, model registry and a champion/challenger promotion rule (local SQLite store: mlflow.db).

Promotion rule: the challenger replaces the champion only if its record-level B-cubed F1 is at
least 0.005 higher AND it does not create more over-merged clusters. Over-merging (two people
treated as one) is the costlier error for an identity graph used in marketing activation.
"""
import mlflow

EXPERIMENT = "identity-resolution"
MIN_F1_GAIN = 0.005
TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "identity-matcher"


def promotion_decision(champion, challenger):
    c, g = champion["constrained_components"], challenger["constrained_components"]
    gain = g["bcubed_f1"] - c["bcubed_f1"]
    promote = gain >= MIN_F1_GAIN and g["overmerged_clusters"] <= c["overmerged_clusters"]
    return {"promote_challenger": bool(promote), "bcubed_f1_gain": gain,
            "overmerge_change": g["overmerged_clusters"] - c["overmerged_clusters"]}


def log_seed(res, lgbm_model):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for name in ("champion_fellegi_sunter", "challenger_lightgbm"):
        r = res[name]
        with mlflow.start_run(run_name=f"{name}_seed{res['seed']}"):
            mlflow.set_tags({"role": name.split("_")[0], "seed": res["seed"], "blocking": "production_union"})
            mlflow.log_params({"threshold": round(r["threshold"], 4), "review_pairs": res["review_pairs"],
                               "candidate_pairs": res["candidate_pairs"]})
            mlflow.log_metrics({"pair_f1": r["pair_f1"], "pair_ap": r["pair_ap"],
                                "bcubed_f1": r["constrained_components"]["bcubed_f1"],
                                "overmerged_clusters": r["constrained_components"]["overmerged_clusters"]})
            if name == "challenger_lightgbm":
                try:
                    info = mlflow.lightgbm.log_model(lgbm_model.booster_, name="matcher",
                                                     registered_model_name=MODEL_NAME)
                except TypeError:
                    info = mlflow.lightgbm.log_model(lgbm_model.booster_, artifact_path="matcher",
                                                     registered_model_name=MODEL_NAME)
                version = getattr(info, "registered_model_version", None)
    # alias the newest challenger version; promote it to "champion" only if the rule says so
    client = mlflow.MlflowClient()
    if version is None:
        version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    client.set_registered_model_alias(MODEL_NAME, "challenger", str(version))
    if res["promotion"]["promote_challenger"]:
        client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
