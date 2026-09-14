"""The RQ1 entry point. Other research questions register their own experiment, not another trainer."""

from mm_sae.data.index import prepare
from mm_sae.features import embed
from mm_sae.models.encoder import make_encoder
from mm_sae.training import train, assert_frozen, validate_training_arguments
from .config import RQ1Config
from .selection import select
from .correspondence import build_panel, experiment1, experiment2
from .interventions import experiment3
from .plotting import report

STAGES = [
    "prepare",
    "embed",
    "train",
    "select",
    "correlate",
    "experiment1",
    "experiment2",
    "experiment3",
    "report",
]


def validate(config):
    options = RQ1Config.model_validate(config.experiment.options)
    options.validate_splits(config)
    validate_training_arguments(config)
    return options


def run(config, store, stage="all"):
    import logging

    options = validate(config)
    encoder = None

    def get_encoder():
        nonlocal encoder
        if encoder is None:
            encoder = make_encoder(config.encoder)
        return encoder

    actions = {
        "prepare": lambda: prepare(config, get_encoder()),
        "embed": lambda: embed(config, get_encoder()),
        "train": lambda: train(config),
        "select": lambda: select(config, options, get_encoder()),
        "correlate": lambda: build_panel(config, options),
        "experiment1": lambda: experiment1(config, options),
        "experiment2": lambda: experiment2(config, options),
        "experiment3": lambda: experiment3(config, options),
        "report": lambda: report(config, options),
    }
    dependencies = {
        "prepare": [],
        "embed": ["prepare"],
        "train": ["embed"],
        "select": ["train"],
        "correlate": ["train"],
        "experiment1": ["select", "correlate"],
        "experiment2": ["select", "correlate"],
        "experiment3": ["select", "experiment2"],
        "report": ["experiment1", "experiment2", "experiment3"],
    }
    if stage != "all" and stage not in STAGES:
        raise ValueError(f"Unknown RQ1 stage {stage!r}; choose from {STAGES}")
    for name in STAGES if stage == "all" else [stage]:
        if store.done(name):
            logging.info("Reuse completed stage %s", name)
            continue
        store.require(*dependencies[name])
        logging.info("Start stage %s", name)
        if name not in {"prepare", "embed", "train"}:
            assert_frozen(config.output)
        actions[name]()
        store.complete(name)
