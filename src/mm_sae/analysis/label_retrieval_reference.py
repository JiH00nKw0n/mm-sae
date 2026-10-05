"""Information diagnostic using evaluation annotations, never a learned result."""

from __future__ import annotations

import numpy as np


def label_only_retrieval(image_labels, caption_parents, *, ks=(1, 5, 10)):
    """Expected recall when every caption receives its image's exact binary labels.

    For nonempty binary vectors cosine is one exactly for equal label sets.
    Uniform random ordering within that tied candidate group gives analytical
    Recall@K. Empty-label observations are retained as failed zero-norm queries.
    This privileged test-label diagnostic is not a general performance bound.
    """
    y = np.asarray(image_labels)
    parents = np.asarray(caption_parents)
    if y.ndim != 2 or not len(y) or not np.all((y == 0) | (y == 1)):
        raise ValueError('A nonempty binary image-label matrix is required')
    if parents.ndim != 1 or not len(parents) or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError('Caption parent indices must be a nonempty integer vector')
    if np.any((parents < 0) | (parents >= len(y))):
        raise ValueError('Caption parent indices are outside the image-label matrix')
    if not ks or any(int(k) != k or k < 1 for k in ks):
        raise ValueError('Recall ranks must be positive integers')
    _, inverse, counts = np.unique(np.packbits(y.astype(bool), axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    captions_per_image = np.bincount(parents, minlength=len(y))
    captions_per_group = np.bincount(inverse[parents], minlength=len(counts))
    active = y.any(1)
    recalls = {'image_to_text': {}, 'text_to_image': {}}
    for k in ks:
        image_hits = []
        for i in range(len(y)):
            total, positive = int(captions_per_group[inverse[i]]), int(captions_per_image[i])
            if not active[i] or positive == 0:
                hit = 0.0
            elif k > total-positive:
                hit = 1.0
            else:
                miss = np.prod([(total-positive-j)/(total-j) for j in range(k)])
                hit = float(1-miss)
            image_hits.append(hit)
        text_hits = np.minimum(float(k)/counts[inverse[parents]], 1) * active[parents]
        recalls['image_to_text'][int(k)] = float(np.mean(image_hits))
        recalls['text_to_image'][int(k)] = float(np.mean(text_hits))
    return dict(
        kind='privileged_test_annotation_reference',
        explanation='Exact test image labels copied to its captions; not activation predictions or an upper bound',
        tie_handling='analytical expectation over uniform random ordering of identical annotation sets',
        uses_test_annotations=True, learned_model=False, dimensions=y.shape[1],
        image_count=len(y), caption_count=len(parents), distinct_label_sets=len(counts),
        images_with_unique_label_set=int(np.sum(counts == 1)), largest_identical_set_group=int(counts.max()),
        empty_label_images=int((~active).sum()),
        retrieval={direction: dict(recall=values) for direction, values in recalls.items()})
