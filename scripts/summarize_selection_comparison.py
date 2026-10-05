"""Compare selection rules using immutable outputs and explicit common cohorts.

No model fitting or inference. Writes full distribution statistics, not only means.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys

import numpy as np

METHODS = ("pooled_auroc", "paired_mean_drop", "single_logistic", "probe_attribution")
SOURCES: dict[str, str] = {}


def read(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        return list(csv.DictReader(f))


def read_json(path):
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def num(row, key):
    v = row.get(key)
    return float(v) if v not in (None, "", "None") else np.nan


def stats(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0, mean=None, std=None, min=None, q1=None, median=None, q3=None, max=None)
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std()), min=float(x.min()),
                q1=float(np.quantile(x, .25)), median=float(np.median(x)),
                q3=float(np.quantile(x, .75)), max=float(x.max()))


def write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def common_values(tables, key, value):
    maps = {tag: {key(r): r for r in rows if np.isfinite(num(r, value))}
            for tag, rows in tables.items()}
    ids = sorted(set.intersection(*(set(d) for d in maps.values())))
    return maps, ids


def concepts(runs):
    data = {m: read(p / "report/concepts.csv") for m, p in runs.items()}
    rows, details = [], []
    for side in ("image", "text"):
        for task in ("removal", "removal_score_presence", "concept_presence"):
            condition = "top_n" if task == "concept_presence" else "learned"
            tables = {(m, n): [r for r in rs if r["side"] == side and r["task"] == task
                       and r["condition"] == condition and int(r["n"]) == n]
                       for m, rs in data.items() for n in (1, 5)}
            maps, ids = common_values(tables, lambda r: int(r["category_id"]), "auroc")
            for kind in ("all", "object", "background"):
                subset = [c for c in ids if kind == "all" or ("object" if c < 91 else "background") == kind]
                for (m, n), values in maps.items():
                    vals = [num(values[c], "auroc") for c in subset]
                    available = [c for c in values if kind == "all" or ("object" if c < 91 else "background") == kind]
                    rows.append(dict(method=m, side=side, task=task, features=n, kind=kind,
                        available_categories=len(available), excluded_category_ids=sorted(set(available)-set(subset)),
                        ids=subset, **stats(vals)))
                    if kind == "all":
                        details.extend(dict(method=m, side=side, task=task, features=n,
                            category_id=c, name=values[c]["name"], auroc=num(values[c], "auroc")) for c in subset)
    return rows, details


def raw_prediction(runs):
    data = {m: read(p / "report/raw_prediction.csv") for m, p in runs.items()}
    rows = []
    for direction in ("image_to_text", "text_to_image"):
        tables = {(m, n): [r for r in rs if r["direction"] == direction
                   and r["condition"] == "top_n" and int(r["n"]) == n]
                   for m, rs in data.items() for n in (1, 5)}
        maps, ids = common_values(tables, lambda r: int(r["category_id"]), "r2_mean")
        for (m, n), values in maps.items():
            rows.append(dict(method=m, direction=direction, features=n, ids=ids,
                             **stats([num(values[c], "r2_mean") for c in ids])))
    return rows


def shared_pairs(runs):
    rows = []
    for side in ("image", "text"):
        tables = {}
        for m, p in runs.items():
            data = read(p / "report/shared_pairs.csv")
            for condition, n in [("shared_one", 1), ("best_all", 1), ("union", 5)]:
                tables[m, condition] = [r for r in data if r["side"] == side
                    and r["condition"] == condition and int(r["n"]) == n]
        def pair_key(r):
            return (int(r["category_a"]), int(r["category_b"]))
        all_maps, common = common_values(tables, pair_key, "auroc")
        for m in METHODS:
            own_maps, own = common_values({k: v for k, v in tables.items() if k[0] == m}, pair_key, "auroc")
            for cohort, ids, maps in [("within_method", own, own_maps), ("all_methods", common, all_maps)]:
                for condition in ("shared_one", "best_all", "union"):
                    values = maps[m, condition]
                    overlap = ([10-int(values[c]["coordinates"]) for c in ids]
                               if condition == "union" else [])
                    rows.append(dict(method=m, side=side, cohort=cohort, condition=condition, ids=ids,
                        mean_top_five_overlap=float(np.mean(overlap)) if overlap else None,
                        identical_top_five_pairs=sum(v == 5 for v in overlap) if overlap else None,
                        **stats([num(values[c], "auroc") for c in ids])))
    return rows


def connections(runs):
    result = []
    for m, p in runs.items():
        rows = read(p / "report/intervention_connections.csv")
        for direction in ("image_to_text", "text_to_image"):
            for condition in ("aligned", "shuffled"):
                selected = {r["model"]: r for r in rows if r["direction"] == direction and r["condition"] == condition}
                base = num(selected["one_to_one"], "normalized_mse")
                for model, row in selected.items():
                    result.append(dict(method=m, direction=direction, condition=condition, model=model,
                        normalized_mse=num(row, "normalized_mse"), targets=int(row["targets"]),
                        relative_improvement_percent=100*(base-num(row, "normalized_mse"))/base))
    return result


def correspondence(runs, root, split):
    counts = (1,) if split == "train" else (1, 5)
    tables = {}
    for m, run in runs.items():
        folder = root / m / "correspondence_train" if split == "train" else run / "correspondence"
        for n in counts:
            tables[m, n] = read(folder / f"learned_{n}_pairs.csv")
    def key(r):
        return (r["direction"], int(r["anchor_id"]), int(r["rival_id"]))
    maps = {tag: {key(r): r for r in rows} for tag, rows in tables.items()}
    ranks = set.intersection(*(set(k for k, r in d.items()
        if np.isfinite(num(r, "original_other")) and np.isfinite(num(r, "original_same"))) for d in maps.values()))
    distributions, rankings, effects = [], [], []
    for cohort, required in [("original_available", ["original_other"]),
                             ("control_available", ["original_other", "controlled_other"])]:
        keys = set.intersection(*(set(k for k, r in d.items() if all(np.isfinite(num(r, v)) for v in required))
                                  for d in maps.values()))
        reference = maps["pooled_auroc", 1]
        for direction in ("image_to_text", "text_to_image"):
            for kinds in ("all", "object_object", "object_background", "background_object", "background_background"):
                selected = sorted(k for k in keys if k[0] == direction and (kinds == "all" or reference[k]["types"] == kinds))
                for lower, upper in zip((0., .2, .4, .6, .8), (.2, .4, .6, .8, 1.), strict=True):
                    ids = [k for k in selected if lower <= num(reference[k], "annotation_correlation")
                           < (upper if upper < .999 else 1.00001)]
                    for (m, n), data in maps.items():
                        meta = dict(split=split, method=m, features=n, direction=direction, types=kinds,
                                    cohort=cohort, lower=round(float(lower), 1), upper=round(float(upper), 1))
                        before = np.array([num(data[k], "original_other") for k in ids])
                        for condition in ("original", "controlled") if cohort == "control_available" else ("original",):
                            values = [num(data[k], f"{condition}_other") for k in ids]
                            distributions.append(dict(**meta, condition=condition, **stats(values)))
                        if cohort == "control_available":
                            after = np.array([num(data[k], "controlled_other") for k in ids])
                            effects.append(dict(**meta, reduced=int((after < before).sum()),
                                increased=int((after > before).sum()), **stats(after-before)))
                rank_keys = sorted(k for k in ranks if k[0] == direction and (kinds == "all" or reference[k]["types"] == kinds))
                if cohort == "original_available":
                    for (m, n), data in maps.items():
                        for lower, upper in [(None, None)] + [(round(float(a),1), round(float(a+.2),1)) for a in np.arange(0,1,.2)]:
                            ids = rank_keys if lower is None or upper is None else [k for k in rank_keys if lower <= num(reference[k], "annotation_correlation") < (upper if upper < .999 else 1.00001)]
                            high = [k for k in ids if num(data[k], "original_other") > num(data[k], "original_same")]
                            margin = np.array([num(data[k], "original_other")-num(data[k], "original_same") for k in ids])
                            anchors = {k[1] for k in ids}
                            rankings.append(dict(split=split, method=m, features=n, direction=direction, types=kinds,
                                lower=lower, upper=upper, pairs=len(ids), higher_pairs=len(high),
                                equal_pairs=int(np.sum(margin == 0)),
                                positive_margins_at_most_1e_12=int(np.sum((margin > 0) & (margin <= 1e-12))),
                                pair_percent=100*len(high)/len(ids) if ids else None,
                                anchors=len(anchors), higher_anchors=len({k[1] for k in high})))
    return distributions, rankings, effects


def diagnosis(runs, baseline):
    data = {m: read((baseline if m == "pooled_auroc" else p / "diagnosis") / "report/category_metrics.csv")
            for m, p in runs.items()}
    result = []
    for train, test in [("presence", "presence"), ("presence", "removal"), ("removal", "removal")]:
        tables = {(m, rep): [r for r in rows if r["trained_on"] == train and r["evaluated_on"] == test and r["representation"] == rep]
                  for m, rows in data.items() for rep in ("embedding", "reconstruction", "all_sae", "selected_five")}
        maps, ids = common_values(tables, lambda r: int(r["category_id"]), "auroc")
        for (m, rep), values in maps.items():
            for metric in ("auroc", "paired_decrease", "mean_paired_change"):
                result.append(dict(method=m, representation=rep, trained_on=train, evaluated_on=test,
                                   measure=metric, ids=ids, **stats([num(values[c], metric) for c in ids])))
    return result


def selection(runs):
    result = []
    for side in ("image", "text"):
        baseline = {p.stem: read_json(p)["ranking"] for p in (runs["pooled_auroc"] / "selection").glob(f"{side}_*.json")}
        all_records = {m: {q.stem: read_json(q) for q in (p / "selection").glob(f"{side}_*.json")}
                       for m, p in runs.items()}
        common = set.intersection(*(set(k for k, v in records.items() if v["ranking"])
                                    for records in all_records.values()))
        for m, p in runs.items():
            records = all_records[m]
            selected = {k: v["ranking"] for k, v in records.items()}
            keys = [k for k in baseline.keys() & selected.keys() if baseline[k] and selected[k]]
            c = Counter(selected[k][0] for k in keys)
            common_counts = Counter(selected[k][0] for k in common)
            result.append(dict(method=m, side=side, categories=len(keys), distinct_top_one=len(c),
                shared_category_pairs=sum(n*(n-1)//2 for n in c.values()),
                common_categories=len(common), common_distinct_top_one=len(common_counts),
                common_shared_category_pairs=sum(n*(n-1)//2 for n in common_counts.values()),
                unselected_categories=[{key: v.get(key) for key in
                    ("category_id", "name", "fit_positives", "tune_positives", "status")}
                    for v in records.values() if not v["ranking"]],
                negative_presence_coefficient_count=(sum(records[k]["coefficient"][0] < 0 for k in keys)
                    if m == "single_logistic" else None),
                same_top_one_as_auroc=sum(baseline[k][0] == selected[k][0] for k in keys),
                mean_top_five_overlap=float(np.mean([len(set(baseline[k][:5]) & set(selected[k][:5])) for k in keys]))))
    return result


def reader_guide(out, result):
    labels = dict(zip(METHODS, ["원본·가림 AUROC", "가림 전후 평균 활성값 감소", "단일 좌표 분류 손실", "개념 분류기 기여도"], strict=True))
    lines = [
        "같은 데이터 분할과 고정된 이미지·텍스트 SAE에서 특징 선택 기준만 바꾸어 비교했다. 아래 수치는 네 기준 모두에서 평가할 수 있는 같은 범주 집합의 평균이다. 분류 성능의 평가 지표는 AUROC로 유지했다.",
        "",
        "| 선택 기준 | 이미지 존재 구별, 특징 1개 | 이미지 존재 구별, 특징 5개 | 텍스트 언급 구별, 특징 1개 | 텍스트 언급 구별, 특징 5개 |",
        "|---|---|---|---|---|",
    ]
    for method in METHODS:
        values = []
        for side in ["image", "text"]:
            for n in [1, 5]:
                row = next(r for r in result["concepts"] if r["method"] == method and r["side"] == side
                           and r["task"] == "concept_presence" and r["features"] == n and r["kind"] == "all")
                values.append(f"{row['mean']:.4f} ({row['n']}개 범주)")
        lines.append("| " + " | ".join([labels[method]] + values) + " |")
    for row in result["selection"]:
        for category in row["unselected_categories"]:
            side = "이미지" if row["side"] == "image" else "텍스트"
            lines += ["", f"{labels[row['method']]} 기준의 {side} {category['name']} 범주는 특징을 선정하지 못했다. "
                      f"학습용 양성 표본은 {category['fit_positives']}개, 조정용 양성 표본은 {category['tune_positives']}개였다."]
    lines += [
        "",
        "원본·가림 AUROC는 개념을 포함한 원본 집단과 해당 개념을 가린 집단을 비교한다. 평균 감소량은 같은 입력의 가림 전후 활성값 차이를 평균한다. 단일 좌표 분류 손실은 원본에서 개념 존재를 예측하는 좌표별 분류기의 조정용 교차엔트로피 손실이다. 분류기 기여도는 개념 유무에 따른 평균 활성값 차이와 해당 복원 방향이 원래 임베딩의 분류기 점수에 기여하는 정도를 곱한다.",
        "",
        "앞의 두 기준은 개념 제거 반응을, 뒤의 두 기준은 원본의 개념 존재 예측을 기준으로 특징을 고른다. 점수 수식뿐 아니라 선택 목표도 달라졌으므로, 성능 차이를 AUROC와 교차엔트로피라는 수식의 차이만으로 설명할 수 없다.",
        "",
        "단일 좌표 분류는 개념 부재를 예측하는 좌표를 고를 수도 있다. 선택한 좌표의 활성값 방향을 상관 평가에서 임의로 뒤집지 않았다. 부재 예측 좌표 수는 선택 통계의 negative_presence_coefficient_count에 기록했다.",
        "",
        "train 상관 분석은 지난 미팅의 질문을 전체 train 이미지·캡션 쌍에서 다시 평가했다. 다만 이번에는 네 기준 모두 같은 80% 학습용, 20% 조정용 분할로 특징을 골랐다. 최초 발표의 train 전체 선택 결과와는 구분해야 한다. val 분석은 이번 미팅과 같은 val 5,000장으로 평가했다.",
        "",
        "상관 표의 direction은 기준 모달리티와 비교할 모달리티를 뜻한다. image_to_text는 이미지 범주 c와 텍스트 범주 d의 비교이며, text_to_image는 텍스트 범주 c와 이미지 범주 d의 비교다. types의 첫 유형은 기준 범주, 두 번째 유형은 상대 범주다. object는 물체, background는 배경이다.",
        "",
        "상자의 중앙선은 중앙값, 상자는 25%와 75% 분위수, 수염은 전체 최솟값과 최댓값이다. 표준편차는 전체 평가 범주 또는 범주 쌍의 분포를 기준으로 계산했다. 이 분포는 평균의 신뢰구간이 아니다. 각 범주 평가의 이미지 단위 1,000회 재표집 구간은 개별 실험 결과에 저장했다.",
        "",
        "같은 대표를 공유하는 범주 쌍과 상대 활성값 예측의 목표 좌표는 선택 기준에 따라 달라진다. 전자는 기준별 집단과 네 기준 공통 집단을 구분했다. 후자는 각 기준 안에서 목표를 고정한 특징 1개와 5개 비교로 해석해야 한다.",
        "",
    ]
    descriptions = {
        "concepts": "개념 존재와 원본·가림 구별의 범주별 분포, 평가 가능한 범주 수와 제외 범주",
        "shared_pairs": "같은 1위 특징을 공유한 범주 쌍의 직접 구별 성능과 상위 5개 중복",
        "raw_prediction": "상대 모달리티 활성값의 예측 성능",
        "connections": "실제 짝과 짝을 섞은 대조에서 연결 제약을 완화한 효과",
        "diagnosis": "원래 임베딩, SAE 전체, 선택한 5개 특징의 이미지 표현 진단",
        "selection": "기준에 따른 1위·상위 5개 변화와 대표 공유 수",
    }
    for split in ["train", "val"]:
        descriptions.update({f"{split}_distributions": f"{split}의 주석 상관 0.2 구간별 전체 분포",
                             f"{split}_rankings": f"{split}의 동일 범주 점수 초과 비율",
                             f"{split}_effects": f"{split}의 양의 동시 등장 통제 전후 차이"})
    for name, description in descriptions.items():
        lines.append(f"- [{description}]({(out / (name + '.csv')).resolve()})에 전체 수치를 저장했다.")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--diagnosis-baseline", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    runs = {m: (args.baseline.resolve() if m == "pooled_auroc" else root / m) for m in METHODS}
    out = root / "comparison"
    out.mkdir(exist_ok=True)
    concept, detail = concepts(runs)
    result = dict(concepts=concept, raw_prediction=raw_prediction(runs), shared_pairs=shared_pairs(runs),
                  connections=connections(runs), diagnosis=diagnosis(runs, args.diagnosis_baseline.resolve()),
                  selection=selection(runs))
    for split in ("train", "val"):
        d, r, e = correspondence(runs, root, split)
        result[f"{split}_distributions"], result[f"{split}_rankings"], result[f"{split}_effects"] = d, r, e
    for name, rows in result.items():
        write(out / f"{name}.csv", rows)
    write(out / "category_details.csv", detail)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    (out / "sources.json").write_text(json.dumps(SOURCES, indent=2))
    reader_guide(out, result)
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_selection_comparison.py")),
                    str(out / "summary.json")], check=True)
    print(out)


if __name__ == "__main__":
    main()
