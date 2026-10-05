"""Separate privileged evaluation-label reference for concept information content."""

import argparse
from pathlib import Path

import numpy as np

from experiments.oracle_sets.run import load_config
from mm_sae.analysis.data import save_json
from mm_sae.analysis.label_retrieval_reference import label_only_retrieval
from mm_sae.io import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    index = Path(cfg['source_run'])/'index/val2017'
    result = label_only_retrieval(np.load(index/'presence.npy'), np.load(index/'parents.npy'))
    result['sources'] = {str(index/name): sha256(index/name) for name in ('presence.npy', 'parents.npy')}
    save_json(Path(cfg['output'])/'label_reference.json', result)


if __name__ == '__main__':
    main()
