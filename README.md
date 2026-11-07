# MM-SAE experiment suite

Train separate sparse autoencoders for image and text embeddings, learn correspondences from paired training data, and evaluate the frozen coordinates on concept prediction and cross-modal retrieval. The suite implements the November 4 meeting protocols on slides 11–16.

The [COCO val2017 results with at most 16 SAE features per concept axis](docs/baselines-2026-11-08/README.md) include the comparison diagram, measured tables and reproduction commands. The [earlier uncapped evaluation](docs/baselines-2026-11-07/README.md) is also available.

## Run the complete offline example

From a clean checkout with Docker and Make installed:

```sh
make build
make smoke
```

The CPU image trains two independent Top-K SAEs with Hugging Face Trainer. It then fits a regularized CCA mapping, rotates it and runs the full set of evaluation adapters on a generated color fixture. No dataset download, GPU or model hub account is required for this example. Results are saved under `runs/smoke/mappings/<id>/evaluation/<id>/results.json`. Fixture scores only test the software. The image is built from source using fixed package versions.

## Use frozen weights

```sh
make run SAE="$PWD/runs/smoke/sae/models" ARGS="run configs/smoke.yaml --output runs/reuse"
```

Set `SAE` to an existing MM-SAE run or its models directory containing `image/` and `text/`, each with config.json and model.safetensors. Existing TopKSAE parameter names and the original encoder/decoder convention are preserved. When provided, the suite loads the saved pair without taking an optimizer step. Evaluation and mapping use the checkpoint's actual latent width and top-k setting. Use the same CLIP model, revision and preprocessing that produced the training embeddings.

## Run on research data

Mount the native datasets described in [benchmark inputs](docs/benchmarks.md). Use `configs/paper.yaml` with a completed CC3M paired embedding cache under `CACHE/cc3m`. Use `configs/cc3m.yaml` to extract that cache from the pinned CC3M WebDataset shards. Extraction resumes at completed shards. Put the Hugging Face cache under `CACHE/huggingface`. The data mount is read-only; caches and results mount separately. The GPU target requires an NVIDIA Docker runtime.

```sh
make build TARGET=gpu IMAGE=mm-sae:gpu
make run TARGET=gpu IMAGE=mm-sae:gpu DATA=/datasets CACHE=/cache RUNS=/results \
  ARGS="run configs/cc3m.yaml"
make run TARGET=gpu IMAGE=mm-sae:gpu DATA=/datasets CACHE=/cache RUNS=/results \
  SAE=/checkpoints/cc3m/models ARGS="run configs/paper.yaml --mapping configs/mapping/hungarian.yaml"
```

All commands use the same installed `mm-sae` entry point. `prepare` extracts embeddings and records IDs/splits; `train-sae` trains or resumes; `fit-mapping` saves the mapping; `evaluate` loads an existing fit using `--fit /workspace/runs/.../mappings/<id>`. `run` executes all stages. A separate recipe can select a subset of benchmarks. An edited benchmark also requires its original probe in the same recipe. Set experiment seeds in the recipe. Stored inputs are checked before reuse; use a fresh output/cache when the data or encoder changes.

## Methods

The training set alone defines both modality means, scales and covariance matrices. Constant SAE features have zero inverse scale and do not enter the fit. Method YAML files in configs/mapping select:

- Hungarian one-to-one assignments. Rank truncation takes the strongest pairs of the full maximum-sum assignment.
- PLS-SVD and ridge CCA. The latter applies the within-modality correction to the projection coefficients.
- Cardinality-constrained sparse PLS and sparse ridge CCA using the archived research solvers. These are support-constrained implementations, not reproductions of a particular L1 paper.
- Post-fit column pruning, or a common rotation followed by pruning. The covariance rotation maximizes the sum of within-column variances of squared, row-normalized feature–coordinate covariances. The sample rotation starts there and maximizes covariance of paired squared image and text scores on sampled training pairs. Both use orthogonal updates with objective-based line search and report stopping status.

Mapping files preserve base coefficients, shared rotation, final coefficients, original feature order and standardization for both modalities. This retains axis semantics even when two rotations have the same overall product. Results record the model fingerprints and resolved method settings.

## Development

```sh
uv sync --frozen --extra dev
uv run pytest
docker build --target test -f docker/Dockerfile -t mm-sae:test .
docker run --rm mm-sae:test
```

The historical command-line interface, experimental scripts and earlier configuration files are preserved on the [archive branch](https://github.com/JiH00nKw0n/mm-sae/tree/archive/pre-rewrite-2026-11-07). Data, cached embeddings and checkpoints live outside version control.
