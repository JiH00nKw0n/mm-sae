"""Shared cached state and atomic per-job artifacts for this experiment suite."""

from pathlib import Path
import json

import numpy as np
from scipy import sparse

from mm_sae.analysis.data import StudyData, SIDES, save_json
from mm_sae.analysis.linear import LinearScore, scale_statistics


class Context:
    def __init__(self, output, options):
        self.out, self.o = Path(output), options
        self.data = StudyData(options["source_run"], options["splits"])
        limit = options.get("smoke_categories")
        self.ids = self.data.train.ids if limit is None else [int(v) for v in limit]
        self.counts = options["feature_counts"]
        self.penalties = options["linear_models"]["penalties"]
        self.repeats = options["uncertainty"]["resamples"]
        self.seed = options["uncertainty"]["seed"]
        self.mean, self.scale, self.alive, self.firing = {}, {}, {}, {}
        for side in SIDES:
            x = self.data.train.activations[side][self.data.fit[side]]
            self.mean[side], self.scale[side] = scale_statistics(x)
            self.alive[side] = np.flatnonzero(np.asarray(x.getnnz(0)).ravel())
            self.firing[side] = np.asarray((x > 0).mean(0)).ravel()

    def file(self, group, key):
        path = self.out / group / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def selection(self, side, cid):
        return json.loads(self.file("selection", f"{side}_{cid}").read_text())

    def features(self, side, cid, n):
        return np.array(self.selection(side, cid)["ranking"][:n], int)

    def readout(self, side, cid, n, condition="learned"):
        r = json.loads(self.file("concepts", f"{side}_{cid}").read_text())
        model = r["models"].get(f"{condition}_{n}")
        return LinearScore.from_record(model) if model else None

    def metadata(self, side, cid):
        return dict(side=side, category_id=cid, name=self.data.names[cid],
                    kind="object" if cid < 91 else "background")

    def removal_data(self, side, cid):
        result = []
        for split, keep in self.data.samples(side):
            rows, changed = split.removed(side, cid)
            selected = keep[rows]
            rows, changed = rows[selected], changed[selected]
            x = sparse.vstack([split.activations[side][rows], changed], format="csr")
            y = np.r_[np.ones(len(rows), bool), np.zeros(len(rows), bool)]
            groups = np.tile(split.groups[side][rows], 2)
            result.append((x, y, groups))
        return result

    def save(self, group, key, record):
        save_json(self.file(group, key), record)


CTX = None


def initialize_worker(output, options):
    global CTX
    CTX = Context(output, options)


def get_context():
    if CTX is None:
        raise RuntimeError("Worker context has not been initialized")
    return CTX
