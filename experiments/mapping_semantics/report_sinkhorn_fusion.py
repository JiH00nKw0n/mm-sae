"""Write complete retrieval tables for fixed Sinkhorn cosine-score averaging."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    methods = {'text_transform': '텍스트를 이미지 좌표로 변환',
               'image_transform': '이미지를 텍스트 좌표로 변환',
               'mean_cosine': '두 코사인 유사도를 50%씩 평균'}
    conditions = {'coco-coco': 'SAE와 대응을 모두 COCO로 학습',
                  'cc3m-cc3m': 'SAE와 대응을 모두 CC3M으로 학습',
                  'cc3m-coco': 'SAE는 CC3M, 대응은 COCO로 학습'}
    lines = ['# Sinkhorn의 두 비교 공간에서 계산한 유사도의 평균', '',
             '**16개 연결을 남긴 Sinkhorn에서는 유사도 평균으로 이미지 질의 Recall@5가 세 조건 모두 높아졌지만, 텍스트 질의에서는 한 조건만 높아졌다. 일반 Sinkhorn에서는 두 검색 방향 모두 각 방향에서 더 좋은 한쪽 변환보다 낮았다.**', '',
             '기존 Sinkhorn 연결 행렬과 학습 자료의 평균·표준편차를 고정하고 검색 점수만 바꾸었다. '
             '텍스트를 이미지 좌표로 변환한 코사인 유사도와 이미지를 텍스트 좌표로 변환한 코사인 유사도를 '
             '각각 계산한 뒤 50%씩 평균했다. 검색 성능 수치를 평균한 것이 아니라 평균한 유사도로 검색 순위를 다시 계산했다.', '',
             '일반 Sinkhorn과 학습 후 행·열마다 최대 16개 연결을 남긴 희소 Sinkhorn을 비교했다. '
             '추가 학습이나 평균 가중치 조정은 하지 않았다. 정규화한 두 표현을 이어 붙이는 계산이 '
             '코사인 유사도 평균과 같음을 단위 검사와 실제 평가 표본에서 확인했다.', '',
             '평가는 COCO val2017 이미지 5,000장과 캡션 25,014개를 사용했다. '
             'Recall@K는 정답이 상위 K개 안에 하나 이상 있는 질의의 비율이다. '
             '이미지 질의에서는 해당 이미지에 속한 모든 캡션을 정답으로 인정한다. '
             '아래 수치는 %이며 각 표·열의 최댓값을 굵게 표시했다.', '']
    records = []
    differences = []
    for condition, title in conditions.items():
        for support in ('full', '16'):
            path = args.root / condition / f'{support}.json'
            result = json.loads(path.read_text())
            data = result['retrieval']
            vals = {m: [100 * data[m][d]['recall'][str(k)]
                        for d in ('image_to_text', 'text_to_image') for k in (1, 5, 10)] for m in methods}
            for method in methods:
                for direction in ('image_to_text', 'text_to_image'):
                    metrics = data[method][direction]
                    assert {metrics['query_count'], metrics['candidate_count']} == {5000, 25014}
                    assert metrics['zero_norm_query_count'] == metrics['zero_norm_candidate_count'] == 0
                    records.append(dict(condition=condition, support=support, method=method,
                                        direction=direction, **{f'recall_at_{k}': metrics['recall'][str(k)]
                                                               for k in (1, 5, 10)}))
            label = '일반 Sinkhorn' if support == 'full' else '최대 16개 연결을 남긴 Sinkhorn'
            lines += [f'## {title}, {label}', '',
                      '| 비교 방법 | 이미지 질의 R@1 | 이미지 질의 R@5 | 이미지 질의 R@10 | 텍스트 질의 R@1 | 텍스트 질의 R@5 | 텍스트 질의 R@10 |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            best = [max(round(v[c], 2) for v in vals.values()) for c in range(6)]
            for method, name in methods.items():
                cells = [f'**{v:.2f}**' if round(v, 2) == best[c] else f'{v:.2f}'
                         for c, v in enumerate(vals[method])]
                lines.append('| ' + ' | '.join([name, *cells]) + ' |')
            lines.append('')
            differences.append((condition, support, *[
                round(vals['mean_cosine'][c] - max(vals['image_transform'][c], vals['text_transform'][c]), 4)
                for c in (1, 4)]))
    lines += ['기존 한쪽 변환도 이번 평균 방식과 같은 서버·계산 정밀도로 다시 평가했다. 이전 저장 결과와 일부 텍스트 질의 지표에서 소수 질의 차이가 있었다. 표는 모두 이번 재계산 결과를 사용한다. 이전 COCO 희소 Sinkhorn은 CPU에서 64비트로 평가했고 이번 평가는 GPU에서 32비트 유사도를 계산했다.', '', '## Recall@5에서 더 높은 한쪽 변환과 비교한 변화', '',
              '각 검색 방향에서 기존 두 변환 중 더 높은 성능을 기준으로 평균 방식의 차이를 계산했다. '
              '이는 결과를 설명하기 위한 비교이며 평가 결과를 이용해 평균 가중치를 선택하지 않았다.', '',
              '| SAE·대응 학습 조건 | 연결 제한 | 이미지 질의 차이(%p) | 텍스트 질의 차이(%p) |',
              '|---|---|---:|---:|']
    for condition, support, image, text in differences:
        lines.append(f'| {conditions[condition]} | {"제한 없음" if support == "full" else "행·열마다 최대 16개"} '
                     f'| {image:+.2f} | {text:+.2f} |')
    (args.root / 'report.md').write_text('\n'.join(lines)+'\n')
    with (args.root / 'retrieval_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
