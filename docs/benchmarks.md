# Benchmark inputs and outputs

The paths in `configs/paper.yaml` are relative to the mounted data directory. Unpack the datasets there, or point the recipe at the extracted files. Real evaluations use local annotations and images. The offline fixture writes miniature files following the same schemas.

| Dataset | Required upstream files | Population and output |
| --- | --- | --- |
| [COCO](https://cocodataset.org/#download) | Train/val 2017 images and instance annotations | One label per annotated category present anywhere in an image, across all instances. Train and validation probes use images from train2017. Test uses val2017. |
| [COCO-Stuff 164K](https://github.com/nightrome/cocostuff) | Same image splits, official indexed PNG masks and labels.txt | Use unique IDs in each PNG; ignore 255. PNG index 0 corresponds to category 1 in labels.txt. Measures object and stuff concepts jointly. |
| [Visual Genome](https://visualgenome.org/api/v0/api_readme) | Extract region_graphs.json and image_data.json; use COCO validation IDs for overlapping images | One text sample per region phrase. Labels come from objects, relationships and explicit attributes in that region. Image-wide annotations are never broadcast to all its phrases. Attribute coverage depends on the distributed graph release. Every region from one image stays in one split. COCO-linked groups share the same partition as the image evaluation. Non-COCO VG images are split by VG image ID. |
| [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) | images/, images.txt, train_test_split.txt, attributes.txt, image_attribute_labels.txt | Binary attribute presence from the original CUB annotations. The official training partition supplies calibration images and the test partition is held out. These original-data probes serve synCUB. |
| [synCOCO](https://huggingface.co/datasets/jokl/syncoco/tree/v1.0) and [synCUB](https://huggingface.co/datasets/jokl/syncub/tree/v1.0) | Unpack v1.0 images and metadata.csv; synCUB also supplies attributes.txt | Hold selected axes and classifier weights fixed. Score the removed, added and retained concepts separately. Pair CSV records signed changes. For retained concepts use mean absolute change, since opposite changes could cancel. Unavailable probes are counted. |
| COCO retrieval | captions_val2017.json and val2017/ | All images and all captions, grouped by original image ID. |
| [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) | Karpathy dataset_flickr30k.json and images/ | Only rows explicitly marked test; every caption associated with its image. |
| [DOCCI](https://google.github.io/docci/) | docci_descriptions.jsonlines and the directory of images named after example_id | Official test rows, human description. |
| [ImageInWords](https://github.com/google/imageinwords) | datasets/IIW-400/data.jsonl and corresponding DOCCI-AAR images | Use the IIW human text and image/key. Evaluate the published IIW-400 file separately from DOCCI. |

## Evaluation protocols

The SAE, normalization and mapping are frozen. For each concept, fit a two-parameter binary logistic classifier on each **single** combination score. Minimize unweighted mean binary cross-entropy on train. A vectorized Newton solver standardizes predictors using training means and scales for conditioning; stored slopes and intercepts refer to original score units. The validation set selects the lowest-loss coordinate. Report the test loss of that coordinate and the unweighted mean over eligible concepts, separately for each dataset and modality. Concepts need the configured minimum positive and negative training count and at least one of each on validation. Store exclusions, per-concept support and the selected coordinate. Equal validation losses resolve to the first coordinate.

For optional DCI-inspired diagnostics use the nonnegative validation loss reduction over a constant classifier using the training prevalence. Normalize these gains across concepts per axis for mixing, and across axes per concept for dispersion. Average axis entropies weighted by total gain; average concept entropies over concepts with positive gain. Report zero-gain concepts separately. This definition and the logistic probes do not fit an all-coordinate DCI predictor.

For retrieval use exact cosine ranks against all candidates of the opposite modality in the same benchmark. A match to any positive counts; ties follow stable candidate order. A zero vector gets zero similarity. Report both directions and the fraction of captions exceeding the encoder's shared token limit. No per-batch candidate restriction.

Run metadata, ordered sample/group IDs and splits are saved beside the embedding cache. Each result directory has a long-form summary.csv for plotting across runs. It also includes the resolved recipe, per-benchmark JSON, per-axis validation losses and importance matrices, edited pair records, and per-query ranks. Synthetic runs carry an explicit flag.
