"""Render measured localization and paired-masking results, including failures."""
from pathlib import Path
from collections import defaultdict
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results'
FIGS = ROOT / 'figures'
FIGS.mkdir(exist_ok=True)
NAMES = {
    'patch_token': '원래 패치와 문장 안 토큰을 비교',
    'patch_phrase': '원래 패치와 별도 구문을 비교',
    'local_token': '마지막 층의 패치 혼합을 생략하고 문장 안 토큰과 비교',
    'local_phrase': '마지막 층의 패치 혼합을 생략하고 별도 구문과 비교',
    'random': '같은 면적을 무작위로 선택',
}
METHODS = list(NAMES)
s = json.loads((DATA / 'summary.json').read_text())
rows = json.loads((DATA / 'records.json').read_text())
audit = json.loads((ROOT / 'audit.json').read_text())


def percent(x):
    return f'{100*x:.2f}%'


def number(x):
    return f'{x:.4f}' if x is not None else '해당 없음'


def table(values, metrics, formats, methods=None):
    methods = methods or METHODS
    best = {metric: sorted({round(values[m][metric]['mean'], 8) for m in METHODS
                            if values[m][metric]['mean'] is not None}, reverse=True) for metric,_ in metrics}
    head = '<tr><th>비교한 방법</th>'+''.join(f'<th>{label}</th>' for _,label in metrics)+'</tr>'
    body = []
    for method in methods:
        cells = []
        for (metric,_), fmt in zip(metrics,formats):
            val = values[method][metric]['mean']
            cell = fmt(val)
            ranking = best[metric]
            if val is not None:
                if round(val,8) == ranking[0]: cell = f'<b>{cell}</b>'
                elif len(ranking)>1 and round(val,8) == ranking[1]: cell = f'<u>{cell}</u>'
            cells.append('<td>'+cell+'</td>')
        body.append('<tr><td>'+NAMES[method]+'</td>'+''.join(cells)+'</tr>')
    return '<div class="table"><table>'+head+''.join(body)+'</table></div>'


def bootstrap_difference(method, control, metric):
    byimage = defaultdict(list)
    for row in rows:
        a = row['methods'][method][metric]
        b = row['random'][metric] if control=='random' else row['methods'][control][metric]
        if a is not None and b is not None:
            byimage[row['image_id']].append(a-b)
    pairs = np.array([(sum(v),len(v)) for v in byimage.values()])
    rng = np.random.default_rng(31)
    samples = rng.integers(len(pairs), size=(2000,len(pairs)))
    boots = pairs[samples,0].sum(1)/pairs[samples,1].sum(1)
    return {'difference': float(pairs[:,0].sum()/pairs[:,1].sum()),
            'ci95': np.quantile(boots,[.025,.975]).tolist(), 'cluster': 'image', 'resamples': 2000}


bootstrap = {metric: bootstrap_difference('local_phrase','patch_token',metric)
             for metric in ['point_hit','precision','recall','delta_cosine','delta_correct_minus_other']}
(ROOT/'uncertainty.json').write_text(json.dumps(bootstrap,indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
sorted_rows = sorted(rows,key=lambda r:(r['methods']['local_phrase']['precision'],r['image_id'],r['category_id']))
examples = [sorted_rows[round(q*(len(sorted_rows)-1))] for q in [0,.125,.25,.375,.5,.625,.75,.875,1]]
cards=[]
gallery_selection=[]
for case_no,row in enumerate(examples):
    case_dir=DATA/'cases'/str(row['image_id'])
    arr=np.load(case_dir/f"{row['category_id']}.npz")
    image=np.asarray(Image.open(case_dir/'input.png'))
    target=arr['target']
    fig,axes=plt.subplots(1,4,figsize=(12.8,3.8))
    axes[0].imshow(image)
    overlay=np.zeros((224,224,4)); overlay[target]=[.12,.65,.28,.48]
    axes[0].imshow(overlay)
    axes[0].set_title('Annotation used only for evaluation',fontsize=10)
    for ax,method,title in [(axes[1],'patch_token','Original patches + caption tokens'),
                            (axes[2],'local_phrase','Local patches + separate phrase')]:
        heat=arr[method].reshape(7,7)
        ax.imshow(image)
        ax.imshow(heat,extent=(-.5,223.5,223.5,-.5),cmap='magma',alpha=.46,
                  interpolation='nearest',vmin=heat.min(),vmax=heat.max())
        for patch in row['methods'][method]['selected_patches']:
            ax.add_patch(Rectangle(((patch%7)*32,(patch//7)*32),32,32,fill=False,lw=1.8,color='cyan'))
        ax.set_title(title,fontsize=10)
        v=row['methods'][method]
        ax.set_xlabel(f"Mask precision {100*v['precision']:.1f}% | coverage {100*v['recall']:.1f}%",fontsize=9)
    masked=image.copy()
    for patch in row['methods']['local_phrase']['selected_patches']:
        r,c=divmod(patch,7); masked[r*32:(r+1)*32,c*32:(c+1)*32]=[123,117,104]
    axes[3].imshow(masked); axes[3].set_title('Five selected patches masked',fontsize=10)
    for ax in axes:
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f"Image {row['image_id']} | phrase: {row['phrase']} | sorted example {case_no+1}/9",x=.03,ha='left',fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.91])
    name=f'case-{case_no+1}.png';fig.savefig(FIGS/name,dpi=150,bbox_inches='tight');plt.close(fig)
    parts=[];pos=0
    for a,b in sorted(row['intervals']):
        parts.extend([html.escape(row['caption'][pos:a]),'<mark>'+html.escape(row['caption'][a:b])+'</mark>']);pos=b
    parts.append(html.escape(row['caption'][pos:]))
    cards.append(f'<article><img loading="lazy" src="figures/{name}" alt="원본과 정답 영역, 두 방법의 유사도, 가림 결과"><p>{"".join(parts)}</p><p class="muted">선택한 표현을 삭제한 문장 · {html.escape(row["deleted_caption"])}</p></article>')
    gallery_selection.append({'image_id':row['image_id'],'category_id':row['category_id'],'precision_quantile':case_no/8,'file':name})

# A complete patch-by-token matrix for the median example, not just the target word.
mid=examples[4]
tokens=np.load(DATA/'cases'/str(mid['image_id'])/'all_tokens.npz')
token_labels=[str(t).replace('</w>','') for t in tokens['tokens']]
matrix=tokens['scores']
fig,ax=plt.subplots(figsize=(max(7,.48*len(token_labels)),8.5))
im=ax.imshow(matrix,cmap='viridis',aspect='auto')
ax.set_xticks(range(len(token_labels)),token_labels,rotation=55,ha='right')
ax.set_yticks(range(0,49,7),[f'row {r+1}' for r in range(7)])
ax.set_xlabel('Caption tokens');ax.set_ylabel('49 image patches in row-major order')
ax.set_title(f"All patches compared with all caption tokens | image {mid['image_id']}",loc='left',fontweight='bold')
fig.colorbar(im,ax=ax,label='Cosine similarity',pad=.02)
fig.tight_layout();fig.savefig(FIGS/'all-tokens.png',dpi=140);plt.close(fig)

(ROOT/'gallery-selection.json').write_text(json.dumps(gallery_selection,indent=2))
locmetrics=[('point_hit','最大 유사도 패치 중심의 정답률'.replace('最大','최대')),
            ('precision','가린 픽셀 중 목표 물체의 비율'),('recall','목표 물체에서 가려진 비율'),('iou','가림 영역과 정답 영역의 IoU')]
NAMES['center']='이미지 중앙의 패치 5개를 고정 선택'
METHODS.append('center')
loc_table=table(s['methods'] | {'center':audit['center']},locmetrics,[percent]*4)
METHODS.remove('center')
deltametrics=[('target_similarity_drop','목표 구문과의 유사도 감소'),('delta_cosine','이미지 변화와 텍스트 변화의 유사도'),
              ('delta_correct_minus_other','정답 변화 유사도에서 다른 구문 변화 유사도를 뺀 값')]
delta_table=table(s['methods'],deltametrics,[number]*3)
category_rows=[]
for name,entry in sorted(s['by_category'].items(),key=lambda t:-t[1]['n']):
    category_rows.append('<tr><td>'+html.escape(name)+'</td><td>'+str(entry['n'])+'</td>'+''.join(
        '<td>'+percent(entry['methods'][m]['precision']['mean'])+'</td>' for m in ['patch_token','local_phrase','random'])+'</tr>')
area_rows=[]
for name,entry in s['by_area'].items():
    if entry['n']:
        area_rows.append('<tr><td>'+{'under_5pct':'물체가 입력 면적의 5% 미만','5_to_20pct':'물체가 입력 면적의 5% 이상 20% 미만','over_20pct':'물체가 입력 면적의 20% 이상'}[name]+'</td><td>'+str(entry['n'])+'</td>'+''.join('<td>'+percent(entry['methods'][m]['precision']['mean'])+'</td>' for m in ['patch_token','local_phrase','random'])+'</tr>')

original=s['methods']['patch_token'];local=s['methods']['local_phrase'];random=s['methods']['random']
pair=audit['within_image_comparison']
pair_table='<table><tr><th>같은 이미지에서 영역을 선택한 기준</th><th>가린 픽셀 중 목표 물체의 비율</th><th>목표 물체에서 가려진 비율</th></tr>'+''.join(
    '<tr><td>'+name+'</td><td>'+percent(pair['methods'][key]['precision'])+'</td><td>'+percent(pair['methods'][key]['recall'])+'</td></tr>'
    for key,name in [('correct','목표 물체를 가리키는 구문으로 선택'),('wrong_phrase','같은 캡션의 다른 물체 구문으로 선택'),('center','중앙의 패치 5개를 선택'),('random','같은 면적을 무작위로 선택')])+'</table>'
content=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CLIP 패치·단어 대응과 공동 가림</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;background:#faf9f6;color:#20242a;margin:0;line-height:1.7}}main{{max-width:1240px;margin:auto;padding:48px 32px}}h1{{font-size:32px;line-height:1.35}}h2{{margin-top:42px;font-size:23px}}p{{max-width:1050px}}.lead{{font-size:21px;border-left:4px solid #ae4545;padding-left:20px}}.muted,small{{color:#606872}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{padding:13px 12px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left;max-width:320px}}th{{background:#efeeeb}}b{{font-weight:750}}u{{text-underline-offset:4px}}.table{{overflow:auto}}article{{background:white;margin:24px 0;padding:16px;border:1px solid #e5e2db;border-radius:8px}}article img{{width:100%}}article p{{margin:6px 12px}}mark{{background:#ffdcad}}a{{color:#365c9b}}code{{font-size:13px;overflow-wrap:anywhere}}.note{{background:#f0ede6;padding:14px 20px}}li{{margin:9px 0}}</style><main>
<p class="muted">2026-10-09 · 고정한 CLIP ViT-B/32 · 학습 없는 사전 확인</p>
<h1>패치와 단어의 유사도로 같은 물체를 찾아 함께 가릴 수 있는가</h1>
<p class="lead"><b>단어에 따라 선택 영역이 달라지는 신호는 확인했습니다. 그러나 물체 위치를 찾는 전체 정확도는 중앙을 고르는 기준과 비슷해, 이 상태로 같은 의미만 가리는 학습 자료를 만들기에는 부족합니다.</b></p>
<p>원래 패치와 문장 안 토큰을 비교했을 때 최대 유사도 패치의 중심이 정답 물체 안에 있는 비율은 <b>{percent(original['point_hit']['mean'])}</b>였습니다. 마지막 층에서 패치 간 혼합을 생략하고 캡션 속 표현을 따로 인코딩하면 <b>{percent(local['point_hit']['mean'])}</b>였습니다. 같은 면적을 무작위로 고를 때의 기준값은 {percent(random['point_hit']['mean'])}였습니다.</p>
<p>가린 영역이 목표 물체로만 이루어진 것은 아닙니다. 마지막 층의 혼합을 생략한 구문 비교에서도 가린 픽셀 중 목표 물체의 비율은 {percent(local['precision']['mean'])}이고, 목표 물체 전체에서 가려진 비율은 {percent(local['recall']['mean'])}였습니다. 따라서 이 결과만으로 개념 하나만 제거하는 학습 자료를 만들었다고 볼 수 없습니다.</p>
<p>중앙의 패치 5개를 고정 선택한 기준은 위치 정답률 {percent(audit['center']['point_hit']['mean'])}, 가림 정밀도 {percent(audit['center']['precision']['mean'])}였습니다. 구문으로 영역을 찾은 방법과의 차이가 작습니다. 이미지 단위로 재표집해 구한 가림 정밀도 차이의 95% 구간도 0을 포함했습니다. 중앙에 물체가 자주 있는 자료의 특성을 고려하면, 무작위보다 높다는 결과만으로 위치 추정이 충분히 좋다고 해석할 수 없습니다.</p>
<h2>무엇을 비교했는가</h2>
<p>COCO val2017의 이미지 {s['sampled_images']}장을 무작위로 선택하고 이미지마다 캡션 하나를 선택했습니다. 그중 입력에 보이는 물체와 캡션의 명시적 언급이 함께 있는 {s['evaluated_images']}장, {s['evaluated_mentions']}개 물체 언급, {s['categories']}개 범주를 평가했습니다. 같은 이미지에서 여러 물체를 평가할 수 있습니다. 픽셀 주석은 위치 정확도 평가에만 사용했고, 패치 순위나 가릴 면적을 결정할 때 사용하지 않았습니다. 캡션에서 평가할 단어를 찾을 때는 기존 80개 물체의 동의어 사전을 사용했습니다. 완전한 자유 어휘 평가는 아닙니다.</p>
<p>입력은 224×224 픽셀이며 패치는 7×7개입니다. 모든 방법에서 유사도가 높은 패치 5개를 선택해 입력의 10.20%를 가렸습니다. 별도 구문 비교에서는 캡션에 실제로 등장한 표현을 “a photo of [표현].”에 넣어 인코딩했습니다. 모든 토큰 비교에서 문장 시작·끝과 채움 토큰은 제외했습니다.</p>
<p>마지막 층의 혼합을 생략하는 조건은 MaskCLIP의 공개 구현에 있는 패치별 value 변환, 잔차 연결, 개별 패치 변환을 Hugging Face CLIP에 적용했습니다. 앞선 층의 문맥 혼합은 남아 있습니다. MaskCLIP의 전체 분할 시스템을 재현한 결과는 아니며, 주석을 사용한 추가 학습이나 정답 범주명 입력은 하지 않았습니다.</p>
<h2>물체 위치를 얼마나 정확히 찾았는가</h2>
{loc_table}
<p>최대 유사도 패치 중심의 정답률은 유사도가 가장 높은 패치의 중심 픽셀이 목표 물체에 속한 비율입니다. 가림 정밀도는 선택한 5개 패치의 픽셀 중 목표 물체에 속한 비율입니다. 물체의 가림 비율은 목표 물체 픽셀 중 선택한 패치로 덮인 비율입니다. IoU는 가림 영역과 정답 영역의 교집합 면적을 합집합 면적으로 나눈 값입니다. 모든 값은 물체 언급별 값을 평균했습니다. 각 열의 1등은 굵게, 2등은 밑줄로 표시했습니다.</p>
<p class="note">최대 유사도 위치를 맞추는 것과 물체 전체만 정확히 가리는 것은 다릅니다. 하나의 32×32 패치에 물체와 배경이 함께 있을 수 있고, 큰 물체는 5개 패치로 전부 가릴 수 없습니다.</p>
<details><summary>물체 크기에 따른 가림 정밀도</summary><table><tr><th>정답 물체의 크기</th><th>물체 언급 수</th><th>원래 패치·토큰</th><th>마지막 층 혼합 생략·별도 구문</th><th>같은 면적 무작위</th></tr>{''.join(area_rows)}</table></details>
<details><summary>범주별 가림 정밀도</summary><table><tr><th>평가한 물체 범주</th><th>물체 언급 수</th><th>원래 패치·토큰</th><th>마지막 층 혼합 생략·별도 구문</th><th>같은 면적 무작위</th></tr>{''.join(category_rows)}</table></details>
<h2>그래도 해당 단어에 대응하는 영역을 찾는 신호는 있는가</h2>
<p>같은 캡션에 평가 가능한 물체가 둘 이상 있는 {pair['images']}장, {pair['mentions']}개 물체 언급에서 구문만 바꾸어 비교했습니다. 예를 들어 컵의 영역을 평가할 때, 컵이라는 표현으로 고른 영역과 같은 캡션의 다른 물체 표현으로 고른 영역을 비교했습니다. 이 표는 위 전체 표와 평가 표본 수가 다릅니다. 모든 행은 이 {pair['mentions']}개 언급에서 계산했습니다.</p>
{pair_table}
<p>정답 구문으로 선택할 때 가림 정밀도는 {percent(pair['methods']['correct']['precision'])}였고, 다른 물체 구문으로 선택하면 {percent(pair['methods']['wrong_phrase']['precision'])}였습니다. 단어에 따라 의미 있는 위치 차이가 생긴다는 근거는 있습니다. 다만 그 정확도 자체는 아직 낮습니다.</p>
<h2>양쪽을 가렸을 때 표현의 변화도 대응하는가</h2>
<p>선택한 이미지 패치를 CLIP 입력 평균색으로 채우고, 캡션에서는 해당 표현을 삭제한 뒤 두 입력을 다시 인코딩했습니다. 최종 토큰을 단순히 지우는 방식을 사용하지 않았습니다. 비교하는 변화 벡터는 정규화한 원본 임베딩에서 정규화한 가림본 임베딩을 뺀 값입니다.</p>
{delta_table}
<p>목표 구문과의 유사도 감소는 원본 이미지와 해당 구문의 코사인 유사도에서 가림 이미지와 해당 구문의 유사도를 뺀 값입니다. 두 변화의 유사도는 이미지 변화 벡터와 텍스트 변화 벡터 사이의 코사인 유사도입니다. 마지막 열은 같은 캡션에 다른 물체 표현도 있는 경우에만 계산했습니다. 이미지에서 목표 물체 후보를 가린 변화가, 다른 물체 표현을 삭제한 텍스트 변화보다 정답 표현을 삭제한 변화와 더 비슷하면 양수입니다. 무작위 가림은 표본마다 5번 계산했습니다.</p>
<p class="note">변화 유사도도 같은 CLIP으로 계산했으므로 독립적인 의미 검증 지표는 아닙니다. 단어 삭제로 문법이 어색해질 수 있고, 가림 영역에 다른 물체가 포함될 수 있습니다. 새 SAE나 대응 행렬을 학습하지 않았으므로 검색 Recall과 기존 42개 범주 매칭 성능이 개선되었다는 결과는 아직 없습니다.</p>
<h2>실제 대응 영역과 가림 결과</h2>
<p>마지막 층의 혼합을 생략한 구문 비교의 가림 정밀도 순으로 정렬하고, 0·12.5·25·37.5·50·62.5·75·87.5·100 백분위에서 사례를 하나씩 골랐습니다. 실패 사례부터 성공 사례까지 보여줍니다. 녹색은 평가에 사용한 정답 영역이고 청록색 테두리는 실제로 선택한 패치 5개입니다. 유사도 색상은 각 그림 안에서 비교해야 합니다. 캡션의 강조한 표현을 이미지 영역과 함께 가렸습니다.</p>
{''.join(cards)}
<details><summary>중앙 사례에서 모든 패치와 모든 캡션 토큰을 비교한 행렬</summary><img style="max-width:100%" src="figures/all-tokens.png"><p>행은 49개 이미지 패치이고 열은 문장 안의 토큰입니다. 패치는 이미지의 왼쪽 위부터 행 우선 순서로 나열했습니다. 이 행렬은 원래 CLIP 마지막 층의 패치 표현과 토큰 표현으로 계산했습니다.</p></details>
<h2>재현 자료와 확인한 한계</h2>
<p>코드와 결과는 <code>{ROOT}</code>에 저장했습니다. 실제 계산 시간은 {s['elapsed_seconds']/60:.1f}분이었습니다. 기존 CLIP의 전체 이미지·문장 임베딩을 그대로 재현하는지 확인한 최대 원소 오차는 각각 {s['checks']['global_image_max_abs_error']:.2g}, {s['checks']['global_text_max_abs_error']:.2g}였습니다.</p>
<p><a href="results/protocol.json">계산 전 고정한 실험 조건</a> · <a href="results/summary.json">집계 결과</a> · <a href="results/records.json">표본별 결과</a> · <a href="uncertainty.json">이미지 단위 재표집으로 계산한 차이의 95% 구간</a></p>
<p><a href="https://github.com/chongzhou96/MaskCLIP/blob/master/mmseg/models/backbones/vit.py">MaskCLIP의 지역 표현 추출 구현</a> · <a href="https://arxiv.org/pdf/2111.07783">FILIP의 패치·토큰 대응</a> · <a href="https://arxiv.org/html/2401.09865v1">SPARC의 토큰별 패치 집합</a></p>
</main></html>'''
(ROOT/'report.html').write_text(content)
print(ROOT/'report.html')
