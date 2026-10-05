"""Present completed mapping comparisons as one evidence table per message."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from experiments.mapping_ablation.report import (
    DIRECTIONS,
    METHODS,
    METHOD_KO,
    RECALL_KS,
    _load_records,
    _score,
)
from experiments.mapping_suite.report import retrieval_table


@dataclass(frozen=True)
class EvidenceRow:
    key: str
    label: str
    note: str = ""
    divider: bool = False


@dataclass(frozen=True)
class Message:
    anchor: str
    title: str
    navigation: str
    comparison: str
    caption: str
    rows: list[EvidenceRow]
    observation: str
    interpretation: str
    limit: str = ""


IMAGE_SPACE = "text_projected_to_image"
TEXT_SPACE = "image_projected_to_text"


def _key(method: str, preprocessing: str, space: str = IMAGE_SPACE) -> str:
    return f"{method}__{preprocessing}__{space}"


def _messages(records: dict[str, dict]) -> list[Message]:
    def score(key: str, direction: str, k: int = 10) -> str:
        return f"{100 * _score(records[key], direction, k):.2f}%"

    def pair(key: str) -> str:
        return "와 ".join(score(key, d) for d in DIRECTIONS)

    hu = _key("hungarian", "raw")
    hc = _key("hungarian", "centered")
    hs = _key("hungarian", "standardized")
    mr = _key("same_budget_many", "raw")
    mc = _key("same_budget_many", "centered")
    gi = _key("sparse_factorization", "standardized")
    gt = _key("sparse_factorization", "standardized", TEXT_SPACE)
    ga = "groups__weighted_average"
    gv = "groups__unit_variance"
    return [
        Message(
            "mean", "평균을 빼는 것만으로 헝가리안의 검색 성능이 크게 좋아졌습니다.",
            "같은 일대일 대응에서 입력 처리만 바꿔 비교합니다.",
            "헝가리안으로 구한 437개 연결과 가중치를 고정하고, 이미지와 텍스트 activation의 처리만 바꿨습니다. "
            "평균과 표준편차는 학습 자료에서 계산했습니다. 텍스트를 이미지 특징 공간으로 옮겨 비교했습니다.",
            "표 1. 같은 헝가리안 대응에서 입력 처리만 바꾼 검색 결과",
            [EvidenceRow(hu, "헝가리안 · 원래 activation 사용"),
             EvidenceRow(hc, "헝가리안 · 학습 평균 제거"),
             EvidenceRow(hs, "헝가리안 · 학습 평균 제거 후 표준편차로 나눔")],
            f"이미지로 캡션을 찾는 Recall@10은 {score(hu, 'image_to_text')}에서 "
            f"{score(hc, 'image_to_text')}로, 캡션으로 이미지를 찾는 Recall@10은 "
            f"{score(hu, 'text_to_image')}에서 {score(hc, 'text_to_image')}로 높아졌습니다. "
            "Recall@1과 Recall@5도 양방향 모두 높아졌습니다.",
            "기존 헝가리안의 낮은 검색 성능에는 입력의 공통 평균 성분이 영향을 주고 있었습니다. "
            "일대일 연결의 제약만으로 기존의 낮은 성능을 설명할 수 없습니다.",
            "이 조건에서는 표준편차까지 적용하면 평균만 제거했을 때보다 성능이 낮았습니다. "
            "입력 처리를 더 많이 한다고 항상 좋아지는 것은 아닙니다.",
        ),
        Message(
            "many-to-many", "같은 연결 수에서 보였던 다대다의 우위는 평균 제거 후 뒤집혔습니다.",
            "연결을 437개로 맞춘 일대일·다대다 대응의 순위 변화를 확인합니다.",
            "헝가리안과 다대다 대응의 전체 연결 수를 각각 437개로 맞췄습니다. "
            "다대다 대응은 한 특징에 여러 연결을 허용합니다. 각 방법의 연결과 가중치는 고정하고 "
            "입력 평균 제거 여부만 바꿨습니다. 텍스트를 이미지 특징 공간으로 옮겨 비교했습니다.",
            "표 2. 같은 437개 연결을 사용한 일대일·다대다 대응",
            [EvidenceRow(hu, "헝가리안 일대일 · 원래 activation 사용", "비교 기준 · 연결 437개"),
             EvidenceRow(mr, "연결 수 제한 다대다 · 원래 activation 사용", "시도한 방법 · 연결 437개"),
             EvidenceRow(hc, "헝가리안 일대일 · 학습 평균 제거", "비교 기준 · 연결 437개", True),
             EvidenceRow(mc, "연결 수 제한 다대다 · 학습 평균 제거", "시도한 방법 · 연결 437개")],
            f"원래 activation에서는 다대다 대응의 양방향 Recall@10이 {pair(mr)}로 "
            f"헝가리안의 {pair(hu)}보다 높았습니다. 평균을 제거하면 다대다는 {pair(mc)}, "
            f"헝가리안은 {pair(hc)}로 순위가 뒤집혔습니다. Recall@1과 Recall@5에서도 동일했습니다. "
            "두 값은 각각 이미지로 캡션 검색과 캡션으로 이미지 검색을 뜻합니다.",
            "앞서 관찰한 다대다의 우위를 입력 처리와 무관한 대응 구조의 이점으로 해석할 수 없습니다.",
            "이는 이번에 고정한 두 대응 방법에 대한 결과입니다. 모든 다대다 방법이 일대일보다 "
            "나쁘다거나, 다대다 대응이 필요 없다는 결론은 아닙니다.",
        ),
        Message(
            "whitening", "같은 256차원에서는 CCA의 내부 상관 보정이 큰 성능 격차를 만들지 않았습니다.",
            "공통 차원을 256개로 맞추고 모달리티 내부 상관 보정의 효과를 비교합니다.",
            "양쪽 activation에 같은 학습 평균과 표준편차를 적용했습니다. 한 조건은 이미지와 텍스트가 "
            "함께 변하는 공통 방향을 특이값분해(SVD)로 구했습니다. 다른 조건은 각 모달리티 내부에서 "
            "특징들이 함께 변하는 정도도 보정하는 정준상관분석(CCA)을 사용했습니다. "
            "두 조건 모두 공통 방향 256개를 남겼습니다.",
            "표 3. 같은 256차원에서 내부 상관 보정 유무를 비교한 검색 결과",
            [EvidenceRow("cross_svd_256", "내부 상관 보정 없이 공통 방향 256개 사용", "특이값분해로 방향 계산"),
             EvidenceRow("cca_256", "내부 상관 보정 후 공통 방향 256개 사용", "CCA로 방향 계산")],
            f"내부 상관 보정이 없는 조건의 양방향 Recall@10은 {pair('cross_svd_256')}, "
            f"CCA는 {pair('cca_256')}였습니다. CCA는 이미지로 캡션을 찾는 Recall@1·5·10을 "
            "각각 0.68·1.50·0.40%포인트 높였지만, 캡션으로 이미지를 찾는 Recall@5·10은 낮췄습니다.",
            "CCA가 기존 희소 대응보다 크게 앞섰던 이유를 내부 상관 보정만으로 설명하기는 어렵습니다. "
            "보정이 없는 공통 표현도 비슷한 검색 성능을 냈습니다.",
            "보정 효과가 전혀 없다는 뜻은 아닙니다. 이미지로 캡션을 찾는 Recall@5에서는 1.50%포인트 "
            "차이가 있습니다. 여기서는 공통 표현의 차원을 맞춘 상태에서 추가 보정의 효과를 확인했습니다.",
        ),
        Message(
            "dimensions", "공통 방향을 437개 모두 쓰는 것보다 256개 남겼을 때 검색이 잘됐습니다.",
            "같은 계산 방법에서 남기는 공통 방향의 수를 64·256·437개로 바꿉니다.",
            "입력 표준화와 공통 방향을 구하는 방법을 고정하고, 함께 변하는 정도가 큰 방향을 "
            "64개, 256개 또는 437개 남겼습니다. 내부 상관 보정이 없는 방법과 CCA를 각각 비교했습니다. "
            "437은 이번 데이터에서 사용할 수 있는 전체 공통 방향의 수입니다.",
            "표 4. 공통 방향을 남기는 수에 따른 검색 결과",
            [EvidenceRow(f"cross_svd_{d}", f"내부 상관 보정 없이 공통 방향 {d}개 사용") for d in (64, 256, 437)]
            + [EvidenceRow(f"cca_{d}", f"CCA로 공통 방향 {d}개 사용", divider=d == 64)
               for d in (64, 256, 437)],
            f"내부 상관 보정 없이 437개를 쓰면 양방향 Recall@10이 {pair('cross_svd_437')}, "
            f"256개를 쓰면 {pair('cross_svd_256')}였습니다. CCA에서도 437개의 "
            f"{pair('cca_437')}보다 256개의 {pair('cca_256')}가 높았습니다. "
            "두 방법 모두 Recall@1·5·10에서 256개가 437개보다 높았습니다.",
            "양쪽에서 함께 변하는 일부 방향을 남기는 선택이 검색에 도움이 됐습니다. "
            "이는 내부 상관 보정 없이도 나타났습니다.",
            "64개로 더 줄이면 다시 성능이 낮아졌습니다. 따라서 차원을 줄일수록 좋다는 결과가 "
            "아니며, 256개가 모든 데이터에 최적이라는 뜻도 아닙니다.",
        ),
        Message(
            "groups", "같은 특징 집합을 직접 비교하자 기존 대응 행렬 방식보다 양방향 검색이 좋아졌습니다.",
            "같은 256개 집합에서 대응 행렬을 쓰는 방식과 집합 값을 직접 비교하는 방식을 비교합니다.",
            "집합은 가중치가 0이 아닌 SAE 특징들을 묶은 것입니다. 기존에 구한 이미지·텍스트 "
            "집합 256개와 소속 특징을 고정했습니다. 기존 방식은 대응 행렬의 가중치를 행 또는 열의 합으로 "
            "나눠 한쪽 activation을 상대 특징 공간으로 옮겼습니다. 새 방식은 각 집합의 activation을 "
            "가중 평균하여 양쪽을 각각 256개 값으로 만들고 직접 비교했습니다. 입력은 모두 표준화했습니다.",
            "표 5. 동일한 256개 특징 집합을 서로 다른 방식으로 사용한 검색 결과",
            [EvidenceRow(gi, "기존 대응 행렬 · 이미지 특징 공간에서 비교", "대응 가중치를 합으로 나눈 방식"),
             EvidenceRow(gt, "기존 대응 행렬 · 텍스트 특징 공간에서 비교", "대응 가중치를 합으로 나눈 방식"),
             EvidenceRow(ga, "256개 집합의 가중 평균끼리 직접 비교", "집합의 구성은 동일", True),
             EvidenceRow("groups__unnormalized_mapping__text_projected_to_image",
                         "가중치를 합으로 나누지 않은 대응 행렬 · 이미지 특징 공간", "비교 방식의 영향을 확인한 중간 조건", True),
             EvidenceRow("groups__unnormalized_mapping__image_projected_to_text",
                         "가중치를 합으로 나누지 않은 대응 행렬 · 텍스트 특징 공간", "비교 방식의 영향을 확인한 중간 조건")],
            f"집합의 가중 평균을 직접 비교하면 양방향 Recall@10은 {pair(ga)}였습니다. "
            f"기존 대응 행렬을 이미지 특징 공간에서 비교한 {pair(gi)}, "
            f"텍스트 특징 공간에서 비교한 {pair(gt)}보다 높았습니다. Recall@1과 Recall@5도 모두 높았습니다.",
            "새 집합을 학습하지 않아도, 같은 집합을 공통 좌표로 직접 사용하는 것만으로 기존 방식의 "
            "검색 성능을 개선할 수 있었습니다.",
            "다만 표의 마지막 두 중간 조건은 각기 한 검색 방향에서 직접 집합 비교보다 높습니다. "
            "따라서 모든 대응 행렬 사용법보다 집합 비교가 좋다는 뜻은 아닙니다. 가중치 처리와 "
            "벡터 길이로 나누는 방식도 달라졌으므로, 집합의 의미가 더 정확해진 효과로 해석할 수 없습니다.",
        ),
        Message(
            "gap", "집합을 직접 비교해도 CCA와의 성능 차이는 남았습니다.",
            "개선한 집합 비교를 기존 가중 대응과 256차원 공통 표현에 비교합니다.",
            "모든 입력에 같은 학습 평균과 표준편차를 적용했습니다. 집합의 가중 평균을 직접 "
            "비교하는 조건과, 집합마다 학습 표준편차로도 나누는 조건을 함께 평가했습니다. "
            "기존 Sinkhorn 가중 대응과, 내부 상관 보정 유무만 다른 두 256차원 공통 표현을 기준으로 두었습니다. "
            "Sinkhorn은 대응 행렬의 행·열별 총 가중치를 맞추는 방법입니다.",
            "표 6. 직접 집합 비교와 기존 비교 방법 사이에 남은 성능 차이",
            [EvidenceRow(_key("sinkhorn", "standardized"), "Sinkhorn · 이미지 특징 공간에서 비교", "기존 가중 대응"),
             EvidenceRow(_key("sinkhorn", "standardized", TEXT_SPACE), "Sinkhorn · 텍스트 특징 공간에서 비교", "기존 가중 대응"),
             EvidenceRow("cross_svd_256", "내부 상관 보정 없이 공통 방향 256개 사용", "기존 선형 변환을 바탕으로 한 비교 조건", True),
             EvidenceRow("cca_256", "CCA로 공통 방향 256개 사용", "기존 비교 방법"),
             EvidenceRow(ga, "256개 집합의 가중 평균끼리 직접 비교", "이번에 확인한 집합 비교", True),
             EvidenceRow(gv, "256개 집합의 가중 평균을 집합별 표준편차로도 나눠 비교", "이번에 확인한 집합 비교")],
            f"직접 집합 비교의 양방향 Recall@10은 {pair(ga)}, 집합별 표준편차까지 적용하면 "
            f"{pair(gv)}였습니다. 두 조건 모두 CCA의 {pair('cca_256')}보다 낮았습니다. "
            "집합별 표준편차를 추가로 적용한 효과는 작았고, Recall@1·5·10 전체에서 일관되게 좋아지지는 않았습니다.",
            "작은 특징 집합을 유지하면서 기존 집합 사용 방식보다 검색을 개선한 것은 확인했습니다. "
            "그러나 CCA 수준의 검색 성능이나 집합의 의미적 정확성을 달성했다고 말할 단계는 아닙니다.",
            "집합에 포함할 특징 수의 제한과 음수 가중치를 허용하지 않는 제약이 남은 성능 차이에 "
            "얼마나 기여했는지는 이번 비교에서 분리하지 않았습니다. 직접 집합 비교가 모든 검색 방향에서 "
            "기존 가중 대응보다 높은 것도 아닙니다.",
        ),
    ]


def _table(rows: list[EvidenceRow], records: dict[str, dict], caption: str) -> str:
    maxima = {(direction, k): max(_score(records[row.key], direction, k) for row in rows)
              for direction in DIRECTIONS for k in RECALL_KS}
    head = (
        f"<div class='table-scroll' role='region' aria-label='{html.escape(caption)}' tabindex='0'>"
        f"<table><caption>{html.escape(caption)}</caption><thead><tr><th rowspan='2' scope='col'>비교 조건</th>"
        "<th colspan='3' scope='colgroup'>이미지로 캡션 검색</th>"
        "<th colspan='3' scope='colgroup'>캡션으로 이미지 검색</th></tr><tr>"
        + "".join(f"<th scope='col'>Recall@{k}</th>" for _ in DIRECTIONS for k in RECALL_KS)
        + "</tr></thead><tbody>"
    )
    body = []
    for row in rows:
        record = records[row.key]
        cls = " class='divider'" if row.divider else ""
        text = f"<tr{cls} data-result-key='{html.escape(row.key)}'><th scope='row'>{html.escape(row.label)}"
        if row.note:
            text += f"<span class='row-note'>{html.escape(row.note)}</span>"
        text += "</th>"
        for direction in DIRECTIONS:
            for k in RECALL_KS:
                score = _score(record, direction, k)
                best = score == maxima[direction, k]
                cls = " class='best'" if best else ""
                value = f"<strong>{score:.2%}</strong>" if best else f"{score:.2%}"
                text += f"<td{cls} data-direction='{direction}' data-k='{k}'>{value}</td>"
        body.append(text + "</tr>")
    return (head + "".join(body) + "</tbody></table></div>"
            "<p class='table-legend'><strong>굵은 값은 각 열의 최댓값</strong>입니다. "
            "반올림 전 수치로 비교했으며, 정확히 같으면 모두 강조했습니다.</p>")


def _readout(records: dict[str, dict]) -> dict[str, list[str]]:
    def s(key: str, direction: str) -> str:
        return f"{_score(records[key], direction):.2%}"

    image, text = "image_to_text", "text_to_image"
    hu, hc = _key("hungarian", "raw"), _key("hungarian", "centered")
    mr, mc = _key("same_budget_many", "raw"), _key("same_budget_many", "centered")
    gi = _key("sparse_factorization", "standardized")
    gt = _key("sparse_factorization", "standardized", TEXT_SPACE)
    ga, gv = "groups__weighted_average", "groups__unit_variance"
    return {
        "mean": [
            f"첫째 행과 둘째 행을 비교하면, 캡션으로 이미지 검색의 Recall@10이 {s(hu, text)}에서 "
            f"{s(hc, text)}로 높아집니다. 연결은 같고 입력 평균만 제거했으므로, 입력 처리만으로 생긴 개선입니다.",
            f"이미지로 캡션 검색도 {s(hu, image)}에서 {s(hc, image)}로 좋아집니다. "
            "따라서 기존 헝가리안의 낮은 성능을 일대일 대응 자체의 한계로만 볼 수 없습니다.",
        ],
        "many-to-many": [
            f"원래 activation을 쓴 위 두 행에서는, 이미지로 캡션 검색의 Recall@10이 헝가리안 "
            f"{s(hu, image)}, 다대다 {s(mr, image)}입니다. 이 조건만 보면 다대다가 유리합니다.",
            f"평균을 제거한 아래 두 행에서는 헝가리안 {s(hc, image)}, 다대다 {s(mc, image)}로 "
            f"뒤집힙니다. 캡션으로 이미지 검색도 헝가리안 {s(hc, text)}, 다대다 {s(mc, text)}입니다. "
            "따라서 원래 입력에서의 우위를 다대다 구조의 일반적인 이점으로 해석할 수 없습니다.",
        ],
        "whitening": [
            f"이미지로 캡션 검색의 Recall@10은 내부 상관 보정 없이 {s('cross_svd_256', image)}, "
            f"CCA로 보정하면 {s('cca_256', image)}입니다. 같은 256차원에서는 보정 없이도 CCA에 가까운 성능을 냈습니다.",
            f"캡션으로 이미지 검색의 Recall@10은 보정 없이 {s('cross_svd_256', text)}, "
            f"CCA는 {s('cca_256', text)}입니다. CCA가 양방향에서 일관되게 유리하지는 않았으므로, "
            "기존의 큰 성능 격차를 내부 상관 보정만으로 설명하기 어렵습니다.",
        ],
        "dimensions": [
            f"내부 상관 보정이 없는 위 세 행에서, 이미지로 캡션 검색의 Recall@10은 437개 방향을 "
            f"쓰면 {s('cross_svd_437', image)}, 256개를 쓰면 {s('cross_svd_256', image)}입니다. "
            f"캡션으로 이미지 검색도 {s('cross_svd_437', text)}에서 {s('cross_svd_256', text)}로 높아집니다.",
            f"CCA를 쓴 아래 세 행에서도 437개보다 256개가 높습니다. 그러나 64개로 줄이면 이미지로 "
            f"캡션 검색은 보정 없는 조건 {s('cross_svd_64', image)}, CCA {s('cca_64', image)}로 낮아집니다. "
            "일부 공통 방향을 남기는 선택은 도움이 됐지만, 적게 남길수록 좋은 것은 아닙니다.",
        ],
        "groups": [
            f"첫 두 행과 세 번째 행을 비교하면, 이미지로 캡션 검색의 Recall@10은 기존 이미지 공간 "
            f"{s(gi, image)}, 기존 텍스트 공간 {s(gt, image)}, 집합 직접 비교 {s(ga, image)}입니다. "
            f"캡션으로 이미지 검색도 각각 {s(gi, text)}, {s(gt, text)}, {s(ga, text)}로 집합 직접 비교가 높습니다.",
            f"하지만 마지막 두 행에서 가중치를 합으로 나누는 처리를 없애면, 이미지 공간에서 "
            f"이미지로 캡션 검색은 {s('groups__unnormalized_mapping__text_projected_to_image', image)}, "
            f"텍스트 공간에서 캡션으로 이미지 검색은 {s('groups__unnormalized_mapping__image_projected_to_text', text)}입니다. "
            "따라서 집합 직접 비교는 기존 두 사용법을 개선했지만, 모든 비교 방식에서 최고는 아닙니다.",
        ],
        "gap": [
            f"CCA 행과 마지막 행을 비교하면, 이미지로 캡션 검색의 Recall@10은 CCA "
            f"{s('cca_256', image)}, 집합별 표준편차까지 적용한 직접 비교 {s(gv, image)}입니다. "
            f"캡션으로 이미지 검색도 CCA {s('cca_256', text)}, 집합 직접 비교 {s(gv, text)}로 차이가 남습니다.",
            f"마지막 두 행끼리 비교하면, 집합별 표준편차를 적용한 전후의 Recall@10은 이미지로 "
            f"캡션 검색에서 {s(ga, image)}와 {s(gv, image)}, 캡션으로 이미지 검색에서 "
            f"{s(ga, text)}와 {s(gv, text)}입니다. 이 추가 처리만으로 CCA와의 차이가 해소되지는 않았습니다.",
        ],
    }


def _highlight_numbers(text: str) -> str:
    return re.sub(r"(\d+(?:\.\d+)?%)", r"<strong>\1</strong>", html.escape(text))


CSS = """
:root{color-scheme:light;--ink:#233342;--muted:#586776;--line:#d7e0e5;--teal:#176a69}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f7f8;color:var(--ink);
font:17px/1.75 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Noto Sans KR',sans-serif}
main{max-width:1280px;margin:0 auto;padding:52px 34px 72px}header{padding-bottom:26px}
.eyebrow{color:var(--teal);font-weight:700;font-size:14px;letter-spacing:.06em;margin:0 0 12px}
h1{font-size:34px;line-height:1.4;letter-spacing:-.035em;margin:0 0 16px}p{margin:12px 0;word-break:keep-all}
.lead{font-size:20px;max-width:1000px}h2{font-size:26px;line-height:1.5;margin:0 0 18px;letter-spacing:-.025em}
h3{font-size:18px;margin:18px 0 6px}a{color:var(--teal);text-underline-offset:3px}nav{margin:14px 0 34px;
padding:22px 28px;background:#eaf1f1;border-left:4px solid var(--teal)}nav ol{margin:0;padding-left:24px}
nav li{padding:7px 0}nav a{font-weight:650;text-decoration:none}nav small{display:block;color:var(--muted);font-size:14px}
.setup{margin-bottom:32px;padding:22px 28px;background:white;border:1px solid var(--line)}
.setup p{font-size:15px}.message{padding:30px 30px 24px;margin:28px 0;background:white;
border:1px solid var(--line);border-radius:6px;scroll-margin-top:24px}.number{display:block;
font-size:13px;letter-spacing:.05em;color:var(--teal);margin-bottom:10px;font-weight:700}
.comparison{color:var(--muted);font-size:16px}.table-scroll{overflow-x:auto;margin:23px 0 22px}
table{border-collapse:collapse;width:100%;min-width:830px;font-size:14px;line-height:1.55;font-variant-numeric:tabular-nums}
caption{text-align:left;font-size:15px;font-weight:650;padding:0 0 12px}td,th{padding:11px 10px;
border-bottom:1px solid var(--line);text-align:right;vertical-align:middle}thead{background:#edf2f4}
thead th[colspan]{text-align:center}th[scope=row]{text-align:left;font-weight:500;width:40%;min-width:290px}
thead th:first-child[rowspan]{text-align:left}.row-note,.method-note{display:block;font-size:12px;color:var(--muted);font-weight:400}
.divider>*{border-top:2px solid #a9bac4}.best{background:#edf7f2;color:#124f45}.best strong{font-weight:850}
.table-legend{color:var(--muted);font-size:13px;margin:-12px 0 24px}
.takeaway{background:#eaf3f1;border-left:5px solid var(--teal);padding:19px 22px;margin-bottom:24px}
.takeaway-label{display:block;color:var(--teal);font-size:14px;font-weight:700;margin-bottom:7px}
.takeaway p{font-size:23px;font-weight:750;line-height:1.65;margin:0;letter-spacing:-.025em}
.reading h3{font-size:18px;color:var(--teal);margin:0 0 12px}.reading ul{margin:0;padding-left:23px}
.reading li{margin:10px 0;font-size:17px}.reading strong{color:var(--ink)}
.limit{color:var(--muted);font-size:14px;margin-top:18px;padding-top:13px;border-top:1px solid #e6ebee}
details{background:white;border:1px solid var(--line);padding:18px 24px;margin:18px 0}summary{cursor:pointer;font-weight:650}
.group-divider th{text-align:left;background:#edf2f4;border-top:2px solid #a9bac4}
.reference .group-divider th{background:#eaf0f7}.explored .group-divider th{background:#e8f4ee}
.supplement img{display:block;width:100%;height:auto;margin:20px 0}.supplement p{font-size:15px;color:var(--muted)}
footer{color:var(--muted);font-size:14px;border-top:1px solid var(--line);padding-top:20px;margin-top:30px}
@media(max-width:700px){main{padding:26px 14px}h1{font-size:27px}h2{font-size:22px}.lead{font-size:18px}
.message{padding:23px 16px}nav,.setup{padding:18px}table{font-size:13px}.reading{padding-left:12px}}
@media print{body{background:white}main{max-width:none;padding:0}.message{break-inside:avoid;margin:18px 0;
padding:20px}nav,.supplement{display:none}table{min-width:0;font-size:11px}h2{font-size:21px}}
"""


def report(suite: Path, ablation: Path) -> Path:
    suite, ablation = suite.resolve(), ablation.resolve()
    records = {r["key"]: r for r in _load_records(ablation)}
    messages = _messages(records)
    readout = _readout(records)
    population = json.loads((suite / "population.json").read_text())
    for record in records.values():
        for direction, count in (("image_to_text", population["test_images"]),
                                 ("text_to_image", population["test_captions"])):
            if record["retrieval"][direction]["query_count"] != count:
                raise ValueError("Follow-up and original report use different query populations")

    body = ["<header><p class='eyebrow'>2026-10-04 · 고정한 SAE에서 대응과 검색 방식을 비교한 결과</p>"
            "<h1>다대다 특징 대응 실험에서 확인한 여섯 가지</h1>"
            "<p class='lead'>입력 처리에 따라 방법 간 순위가 달라졌습니다. CCA의 내부 상관 보정 효과는 "
            "제한적이었고, 같은 특징 집합을 공통 좌표로 직접 비교하면 기존 집합 사용 방식보다 검색이 좋아졌습니다.</p>"
            "<p>각 메시지의 결론을 크게 강조하고, 근거 표의 열별 최댓값을 굵게 표시했습니다. "
            "표 아래에는 어떤 값을 비교하면 그 해석을 얻을 수 있는지 적었습니다.</p></header>",
            "<nav aria-label='목차'><ol>"]
    for msg in messages:
        body.append(f"<li><a href='#{msg.anchor}'>{html.escape(msg.title)}</a>"
                    f"<small>{html.escape(msg.navigation)}</small></li>")
    body += ["</ol></nav>", "<aside class='setup'><h2>비교를 읽는 기준</h2>",
             "<p>SAE는 모델의 내부 표현을 소수의 활성 특징으로 분해하는 모델입니다. "
             "activation은 각 특징의 활성값이고, coactivation correlation은 이미지·텍스트 쌍에서 "
             "두 특징의 활성값이 함께 변하는 정도입니다.</p>",
             f"<p>동일한 COCO 검증 이미지 {population['test_images']:,}장과 캡션 "
             f"{population['test_captions']:,}개로 평가했습니다. SAE와 임베딩 모델은 고정했습니다. "
             "대응은 COCO 학습 이미지의 80%로 구하고, 나머지 20%에서 양방향 activation 예측 오차로 "
             "설정을 선택했습니다. 이번 후속 비교에서는 검색 성능에 맞춰 설정을 다시 선택하지 않았습니다.</p>",
             "<p><strong>Recall@1·5·10은 정답이 검색 상위 1개·5개·10개 안에 들어간 질의의 비율입니다.</strong> "
             "이미지로 캡션을 찾을 때는 해당 이미지의 정답 캡션 중 하나 이상을 찾으면 성공입니다. "
             "캡션으로 이미지를 찾을 때는 해당 캡션의 이미지를 찾으면 성공입니다. 값이 높을수록 좋습니다. "
             "이는 특징 사이의 의미 대응 정확도를 직접 측정한 값은 아닙니다.</p>",
             "<p>모든 표에 두 검색 방향의 세 지표를 표시했습니다. 본문은 Recall@10을 중심으로 설명하되 "
             "나머지 지표에서 다른 경향이 나타나면 함께 적었습니다. 입력 평균·표준편차와 특징 간 "
             "상관은 학습 자료에서만 계산했습니다.</p></aside>"]
    evidence = []
    for number, msg in enumerate(messages, 1):
        body.extend([f"<section class='message' id='{msg.anchor}'><span class='number'>메시지 {number:02d}</span>",
                     f"<h2>{html.escape(msg.title)}</h2>",
                     "<div class='takeaway'><span class='takeaway-label'>핵심 메시지</span>",
                     f"<p><strong>{html.escape(msg.interpretation)}</strong></p></div>",
                     f"<p class='comparison'>{html.escape(msg.comparison)}</p>",
                     _table(msg.rows, records, msg.caption),
                     "<div class='reading'><h3>어떤 결과를 보면 이 해석을 얻을 수 있나?</h3><ul>",
                     "".join(f"<li>{_highlight_numbers(line)}</li>" for line in readout[msg.anchor]),
                     "</ul></div>",
                     f"<p class='limit'>{html.escape(msg.limit)}</p></section>"])
        for row in msg.rows:
            for direction in DIRECTIONS:
                evidence.append({"message": number, "message_title": msg.title, "condition": row.label,
                                 "result_key": row.key, "direction": direction,
                                 **{f"recall_at_{k}": _score(records[row.key], direction, k) for k in RECALL_KS}})

    body.append("<div class='supplement'><h2>전체 비교표와 기존 보조 결과</h2>"
                "<p>본문의 메시지를 확인하는 데 필요한 표는 위에 모두 표시했습니다. "
                "전체 방법과 중간 조건을 확인하려면 아래 항목을 펼칠 수 있습니다.</p>")
    selected = json.loads((suite / "selected.json").read_text())
    original = [json.loads((suite / "evaluation" / f"{r['family']}.json").read_text()) for r in selected]
    dense = json.loads((suite / "dense_references.json").read_text())["results"]
    body.append("<details><summary>기존 결과 전체를 확인합니다.</summary>"
                "<p>아래 기존 표는 대응 행렬 방법에는 원래 activation을 사용하고, CCA와 Procrustes에는 "
                "평균과 표준편차를 적용한 입력을 사용했습니다. 따라서 입력 처리가 다른 상태의 참고 결과입니다. "
                "위 여섯 메시지는 후속 통제 비교까지 반영했습니다.</p>"
                "<p>Procrustes는 벡터의 길이와 각도를 보존하는 선형 변환입니다. CCA와 함께 별도의 "
                "변환된 공간에서 검색했으므로 특징 사이 연결 수는 해당하지 않습니다.</p>")
    for space, label in ((IMAGE_SPACE, "텍스트를 이미지 특징 공간으로 옮긴 기존 결과"),
                         (TEXT_SPACE, "이미지를 텍스트 특징 공간으로 옮긴 기존 결과")):
        body.extend([f"<h3>{label}</h3>", retrieval_table(original, dense, space, highlight_best=True)])
    body.append("</details><details><summary>10개 대응 방법의 입력 처리별 결과를 모두 확인합니다.</summary>"
                "<p>각 방법의 연결과 가중치는 고정했습니다. 같은 방법 안에서 입력 처리의 효과를 비교합니다.</p>")
    for space, label in ((IMAGE_SPACE, "텍스트를 이미지 특징 공간으로 옮긴 후속 비교"),
                         (TEXT_SPACE, "이미지를 텍스트 특징 공간으로 옮긴 후속 비교")):
        rows = [EvidenceRow(_key(method, condition, space), METHOD_KO[method], preprocessing, condition == "raw")
                for method in METHODS
                for condition, preprocessing in (("raw", "원래 activation 사용"), ("centered", "학습 평균 제거"),
                                                  ("standardized", "학습 평균 제거 후 표준편차로 나눔"))]
        body.append(_table(rows, records, label))
    body.append("</details><details><summary>기존 예측·객체 제거·연결 수 분석을 확인합니다.</summary>"
                "<p>이 보조 결과는 이번 입력 처리·공통 표현 비교로 다시 계산하지 않았습니다. "
                "위 여섯 메시지의 근거는 각 메시지에 붙인 검색 결과 표입니다.</p>")
    for name, note in (
        ("prediction_distribution", "상대 모달리티 activation을 예측한 결과입니다. 위쪽은 대응 비율을 유지하고 "
         "특징별 배율만 학습했으며, 아래쪽은 선택한 연결 안에서 회귀 계수도 다시 학습했습니다."),
        ("removal_distribution", "객체를 가리기 전후의 activation 차이를 예측한 결과입니다. "
         "범주 안에서 이미지·텍스트 짝을 섞은 조건과도 비교했습니다."),
        ("complexity", "대응 연결 수와 연결된 특징의 비율입니다. 연결의 의미가 정확한지를 측정한 값은 아닙니다."),
    ):
        body.append(f"<p>{note}</p><img src='figures/{name}.svg' alt='{note}'>")
    body.append("</details></div><footer><p>본문 표의 모든 수치는 완료된 결과 파일에서 읽었습니다. "
                "이번 문서 수정에서는 학습이나 실험을 다시 실행하지 않았습니다. "
                "<a href='message_evidence.csv'>메시지별 근거 수치 CSV</a>와 "
                "<a href='retrieval_summary.csv'>기존 전체 검색 결과 CSV</a>를 제공합니다.</p></footer>")

    with (suite / "message_evidence.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)
    document = ("<!doctype html>\n<html lang='ko'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<title>다대다 특징 대응 실험에서 확인한 여섯 가지</title>"
                f"<style>{CSS}</style></head><body><main>" + "\n".join(body) + "</main></body></html>")
    output = suite / "report.html"
    output.write_text(document, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    arguments = parser.parse_args()
    print(report(arguments.suite, arguments.ablation))
