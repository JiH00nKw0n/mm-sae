"""Signed row-wise argmax and maximum-weight one-to-one assignment."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match(c, alive_image, alive_text, valid_image, valid_text):
    # Keep the original candidate universe, but undefined correlations are not edges.
    # Constant image rows stay in the output as -1 and in the assessment denominator.
    ii = np.flatnonzero(alive_image & valid_image)
    jj = np.flatnonzero(alive_text & valid_text)
    greedy = np.full(c.shape[0], -1, dtype=np.int64)
    hungarian = greedy.copy()
    ties = np.zeros(c.shape[0], np.int64)
    if len(ii) and len(jj):
        sub = c[np.ix_(ii, jj)]
        if not np.isfinite(sub).all():
            raise ValueError("A correlation marked valid is not finite")
        greedy[ii] = jj[np.argmax(sub, axis=1)]
        ties[ii] = (sub == sub.max(axis=1, keepdims=True)).sum(axis=1)
        a, b = linear_sum_assignment(-sub)
        hungarian[ii[a]] = jj[b]
    return {"greedy": greedy, "hungarian": hungarian, "greedy_ties": ties}
