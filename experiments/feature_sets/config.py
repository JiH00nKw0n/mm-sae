"""Validate the approved experiment contract before any work starts."""

from pathlib import Path


def validate(config):
    o = config.experiment.options
    required = {"source_run", "suites", "source_policy", "splits", "feature_counts", "selection",
                "linear_models", "removal_readout", "controls", "original_rq2", "refined_rq2",
                "multi_feature_rq1", "uncertainty", "runtime", "report"}
    unknown = set(o)-required-{"smoke_categories"}
    if required-set(o) or unknown:
        raise ValueError(f"Missing options {required-set(o)}; unknown options {unknown}")
    if not set(o["suites"]) <= {"original_rq2", "refined_rq2", "multi_feature_rq1"}:
        raise ValueError("Unknown suite")
    if o["source_policy"] != "require_complete_cache":
        raise ValueError("Only cached frozen-model analyses are supported")
    counts = o["feature_counts"]
    if not counts or 1 not in counts or counts != sorted(set(counts)) or any(n < 1 for n in counts):
        raise ValueError("Feature counts must be unique, increasing, positive and contain 1")
    if not 0 < o["splits"]["tune_image_fraction"] < 1:
        raise ValueError("Invalid tuning fraction")
    if o["selection"]["metric"] not in {"pooled_auroc", "paired_mean_drop",
                                        "single_logistic", "probe_attribution"}:
        raise ValueError("Unknown feature selection metric")
    if o["selection"]["metric"] != "pooled_auroc" and not o["selection"].get("prepared_directory"):
        raise ValueError("Alternative selectors require verified prepared rankings")
    fixed = [(o["selection"], "split", "fit"), (o["selection"], "minimum_auroc", None),
             (o["removal_readout"], "coefficient_constraint", "nonnegative"),
             (o["removal_readout"], "output_for_correlation", "linear_score"),
             (o["multi_feature_rq1"], "decorrelation", "pairwise_positive_only_preserve_marginals")]
    for section, key, expected in fixed:
        if section[key] != expected:
            raise ValueError(f"Unsupported option {key}={section[key]}")
    if min(o["linear_models"]["penalties"]) <= 0:
        raise ValueError("Ridge/logistic penalties must be positive")
    if not Path(o["source_run"]).is_dir():
        raise FileNotFoundError(o["source_run"])
    if config.output.resolve() == Path(o["source_run"]).resolve():
        raise ValueError("Never overwrite the source experiment")
    return o
