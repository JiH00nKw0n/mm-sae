import pytest
from mm_sae.config import Config
from experiments.rq1.config import RQ1Config


def test_unknown_options_and_partial_smoke_limits_are_rejected():
    with pytest.raises(ValueError):
        Config.model_validate({"training": {"learing_rate": 0.1}})
    with pytest.raises(ValueError):
        Config.model_validate({"data": {"image_limits": {"train2017": 4}}})
    with pytest.raises(ValueError):
        RQ1Config.model_validate({"bin_width": 0.3})


def test_validation_is_disjoint_from_selection():
    cfg = Config()
    with pytest.raises(ValueError):
        RQ1Config(validation_split="train2017").validate_splits(cfg)
