from pathlib import Path
import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
ROOT=Path('/Users/jihoonkwon/Desktop/projects/project-pages/f4750a4b/multimodal-SAE')
ASSETS=ROOT/'assets/2026-10-06-mapping'
ASSETS.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'mathtext.fontset':'cm','svg.fonttype':'path','font.family':'serif'})
def eq(name,tex,height=62):
    fig=plt.figure(figsize=(12,1.5)); text=fig.text(.01,.5,'$'+tex+'$',fontsize=26,color='#1a1d21')
    fig.canvas.draw();bbox=text.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.025,1.16)
    fig.savefig(ASSETS/f'{name}.svg',bbox_inches=bbox,transparent=True);plt.close(fig)
    return f'<img class="equation" src="assets/2026-10-06-mapping/{name}.svg" style="height:{height}px" alt="{html.escape(tex,quote=True)}">'
slides=[]
def slide(title,body,kind='',note=''):
    n=len(slides)+1
    slides.append(f'<section class="slide {kind}" id="slide-{n}"><p class="eyebrow">2026-10-06 진행 상황 공유 · Baselines</p><h2>{title}</h2><div class="content">{body}</div><aside class="speaker-notes">{note}</aside></section>')
E={}
E[0]=eq('input',r'X_0\in\mathbb{R}^{N\times d_I},\quad Y_0\in\mathbb{R}^{N\times d_T}',36)
E[1]=eq('correlation',r'C=\frac{X^{\mathsf{T}}Y}{N}\in\mathbb{R}^{d_I\times d_T}',56)
E[2]=eq('cosine-direct',r's_W(x,y)=\cos(xW,y)=\frac{xWy^{\mathsf{T}}}{\|xW\|_2\|y\|_2}',71)
E[3]=eq('factorization',r'W=AB^{\mathsf{T}}',37)
E[4]=eq('cosine-common',r's_{A,B}(x,y)=\cos(xA,yB)=\frac{xWy^{\mathsf{T}}}{\|xA\|_2\|yB\|_2}',70)
def alignment_diagram(common=False):
    # Nonnegative sparse inputs, followed by standardization and learned transformation.
    def bars(x,y,values):
        cells=[]
        for i,v in enumerate(values):
            xx=x+i*11
            if v==0:
                fill='#edf0f3'
            else:
                base=(117,0,20) if v>0 else (67,106,148)
                strength=min(abs(v)/.8,1)
                rgb=tuple(round(247+(c-247)*strength) for c in base)
                fill='#'+''.join(f'{c:02x}' for c in rgb)
            cells.append(f'<rect x="{xx}" y="{y-5}" width="9" height="9" rx="1" fill="{fill}"/>')
        return ''.join(cells)
    xx=[0,.8,0,0,0,.6,0,0];yy=[0,0,0,.8,0,0,0,.6]
    out=[.62,-.16,.40,-.10] if common else [-.15,-.15,-.15,.72,-.15,-.15,-.15,.53]
    other=[.58,-.12,.46,-.08] if common else [-.15,-.15,-.15,.8,-.15,-.15,-.15,.6]
    right='共通' if common else 'text'
    return f"""<svg class="alignment-diagram" viewBox="0 0 296 174" role="img" aria-label="같은 이미지·텍스트 쌍의 서로 다른 희소 활성 좌표를 표준화하고 {'A와 B로 공통 좌표에' if common else 'W로 텍스트 좌표에'} 맞추는 개념도">
<text x="2" y="14" class="diagram-head">같은 이미지·텍스트 쌍</text>
<text x="192" y="14" class="diagram-head">{'공통 좌표' if common else '텍스트 좌표'}</text>
<text x="2" y="39">이미지 latent</text><text x="194" y="39">{'xA' if common else 'xW'}</text>
{bars(3,65,xx)}{bars(195,65,out)}
<text x="2" y="106">텍스트 latent</text><text x="194" y="106">{'yB' if common else 'y'}</text>
{bars(3,132,yy)}{bars(195,132,other)}
<path d="M103 62 H182 M103 129 H182" stroke="#77818d" stroke-width="1.3"/>
<path d="M177 59 L183 62 L177 65 M177 126 L183 129 L177 132" fill="none" stroke="#77818d" stroke-width="1.3"/>
<text x="140" y="53" text-anchor="middle">{'표준화 · A' if common else '표준화 · W'}</text>
<text x="140" y="120" text-anchor="middle">{'표준화 · B' if common else '표준화'}</text>
<text x="2" y="170" class="diagram-note">희소 activation</text><text x="187" y="170" class="diagram-note">정렬 목표 예시</text>
</svg>"""
D0=alignment_diagram(False)
D1=alignment_diagram(True)
slide('이미지·텍스트 activation의 유사도 기반 정렬',f'''
<p class="goal">대응 행렬 <b>W</b>를 이용해 비교했을 때, <b>정답 이미지·텍스트 쌍의 유사도가 다른 쌍보다 높아지도록</b> 합니다.</p>
<div class="setup-columns"><div>
<h3>입력과 좌표별 표준화</h3>
{E[0]}
<p>같은 행은 같은 이미지·캡션 쌍입니다.<br>N은 쌍의 수, d<sub>I</sub>·d<sub>T</sub>는 양쪽 activation 좌표 수입니다.</p>
<p class="space"><b>각 좌표의 전체 N개 표본 평균을 뺀 뒤,<br>같은 좌표의 표준편차로 나눕니다.</b></p>
{E[1]}
<p>X·Y는 표준화한 행렬입니다.<br>C는 이미지·텍스트 activation 좌표 사이의 <b>coactivation correlation</b>입니다.</p>
</div><div>
<div class="alignment-row"><div class="alignment-formula">
<h3>한쪽 activation을 W로 변환하는 경우</h3>
{E[2]}
</div>{D0}</div>
<div class="alignment-row"><div class="alignment-formula">
<h3>양쪽을 변환하는 경우</h3>
{E[3]}{E[4]}
</div>{D1}</div>
<p class="alignment-definition">x·y는 표준화한 activation, A·B는 공통 좌표로 조합하는 행렬입니다.<br>칸의 진하기는 값의 크기를 나타냅니다. 붉은색은 양수, 푸른색은 음수입니다.</p>
</div></div>
<div class="bottom">핵심 질문은 <b>어떤 관계를 맞추면 유사도도 높아질 것이라고 보는가</b>입니다.<br><span>아래 목적식은 오답 쌍과 직접 비교하지 않습니다. 검색 성능과 조합의 의미는 별도로 확인해야 합니다.</span></div>
''','setup','표준편차는 N으로 나눈 분산의 제곱근을 사용한다. 상수 좌표를 제외한다. 따라서 C는 Pearson 상관행렬이다. 첫 번째 유사도 식에서 Procrustes의 차원이 다르면 0으로 채운 벡터를 사용한다. 두 번째 식의 분모는 A·B에 의존하므로 W만으로 정해지지 않는다. 그림은 같은 쌍의 원래 희소 activation 좌표가 다르다는 개념도이며 실제 결과가 아니다. 입력은 표준화 전이고 변환 뒤에는 음수와 밀집된 값이 있을 수 있다.')
def method(title,formula,definitions,purpose,constraint,logic,limit,note='',visual=''):
    layout='hungarian-three' if visual else 'method-columns'
    extra=f'<div class="procrustes-visual">{visual}</div>' if visual else ''
    slide(title,f'''<div class="{layout}"><div class="formula-column">{formula}<div class="definitions">{definitions}</div></div><div class="explain"><h3>목적</h3><p>{purpose}</p><h3>제약</h3><p>{constraint}</p><h3>유사도와의 연결</h3><p>{logic}</p></div>{extra}</div><div class="bottom">{limit}</div>''',note=note)
def procrustes_diagram():
    import math
    original=[(1.8,.2),(.4,1.6),(1.3,1.2)]
    c=math.sqrt(.5)
    target=[((x-y)*c,(x+y)*c) for x,y in original]
    pieces=['<svg viewBox="0 0 280 425" class="procrustes-diagram" role="img" aria-label="2次元の回転例"><g font-family="inherit" fill="#4f5660" font-size="12">']
    pieces[0]=pieces[0].replace('2次元の回転例','모든 이미지 activation 벡터를 원점 중심으로 45도 회전해 대응 텍스트 벡터에 맞추는 2차원 예시')
    for index,top in enumerate([43,231]):
        oy=top+123
        def pos(p):return 123+p[0]*43,oy-p[1]*43
        points=original if index==0 else target
        pieces.append(f'<text x="5" y="{top+3}" fill="#750014" font-size="15" font-weight="650">{"변환 전" if index==0 else "같은 Q로 45° 회전한 후"}</text>')
        pieces.append(f'<path d="M34 {oy} H253 M123 {top+19} V{oy+7}" stroke="#c6ccd3" fill="none"/><text x="244" y="{oy+18}">1축</text><text x="127" y="{top+28}">2축</text><text x="112" y="{oy+16}">0</text>')
        poly=' '.join(f'{pos(p)[0]:.2f},{pos(p)[1]:.2f}' for p in points)
        pieces.append(f'<polygon points="{poly}" fill="#750014" fill-opacity=".05" stroke="#750014" stroke-width="1"/>')
        for i,(p,q) in enumerate(zip(points,target)):
            px,py=pos(p);qx,qy=pos(q)
            pieces.append(f'<path d="M123 {oy} L{px} {py}" stroke="#b5bbc4" stroke-width=".8"/>')
            if index==0:pieces.append(f'<path d="M{px} {py} L{qx} {qy}" stroke="#9ca6b2" stroke-width="1.2" stroke-dasharray="3 3"/>')
            pieces.append(f'<circle cx="{qx}" cy="{qy}" r="7" fill="white" stroke="#2584a4" stroke-width="2"/><circle cx="{px}" cy="{py}" r="4" fill="#750014"/>')
            if index==0:pieces.append(f'<text x="{px+6}" y="{py+12}" fill="#750014">{i+1}</text><text x="{qx-13}" y="{qy-8}" fill="#2584a4">{i+1}</text>')
    pieces.append('<circle cx="10" cy="15" r="4" fill="#750014"/><text x="20" y="19">이미지</text><circle cx="86" cy="15" r="6" fill="white" stroke="#2584a4" stroke-width="2"/><text x="98" y="19">정답 텍스트</text><text x="5" y="218">점선은 정답 쌍 사이의 거리</text><text x="5" y="411" fill="#750014" font-size="14" font-weight="650">길이와 점들 사이의 각도는 유지</text></g></svg>')
    return ''.join(pieces)
PV=procrustes_diagram()
def hungarian_diagram():
    return '''<svg class="hungarian-diagram swap-diagram" viewBox="0 0 280 370" role="img" aria-label="2차원 activation 좌표를 맞바꾸면 점 (2,1)이 직선 x1=x2에 반사되어 (1,2)로 이동합니다. 대응 행렬은 0,1;1,0입니다.">
<g font-family="inherit" font-size="13" fill="#4f5660">
<circle cx="9" cy="12" r="4" fill="#750014"/><text x="19" y="16" fill="#750014">이미지</text><circle cx="88" cy="12" r="6" fill="white" stroke="#2584a4" stroke-width="2"/><text x="100" y="16" fill="#2584a4">정답 텍스트</text>
<path d="M38 222 H257 M38 222 V35" stroke="#aeb7c2" stroke-width="1.2" fill="none"/>
<text x="246" y="241">1축</text><text x="10" y="34">2축</text><text x="22" y="241">0</text>
<path d="M38 222 L219 41" stroke="#aeb7c2" stroke-dasharray="4 4" fill="none"/><text x="188" y="33">x₁ = x₂</text>
<path d="M108 218 V226 M178 218 V226 M34 152 H42 M34 82 H42" stroke="#aeb7c2"/>
<text x="104" y="241">1</text><text x="174" y="241">2</text><text x="22" y="157">1</text><text x="22" y="87">2</text>
<path d="M38 222 L178 152 M38 222 L108 82" stroke="#cbd1d9" fill="none"/>
<path d="M173 147 L115 89" stroke="#750014" stroke-width="2" fill="none"/><path d="M115 98 L114 88 L124 89" stroke="#750014" stroke-width="2" fill="none"/>
<circle cx="178" cy="152" r="5" fill="#750014"/><text x="170" y="174" fill="#750014">이미지 x = (2, 1)</text>
<circle cx="108" cy="82" r="8" fill="white" stroke="#2584a4" stroke-width="2"/><circle cx="108" cy="82" r="4" fill="#750014"/><text x="49" y="53" fill="#2584a4">정답 y = (1, 2)</text><text x="49" y="69" fill="#750014">변환 후 xP = (1, 2)</text>
<text x="102" y="192" fill="#750014">座標</text>
<text x="28" y="278" fill="#750014" font-size="15" font-weight="650">2つ</text>
<text x="28" y="310" font-size="21" font-style="italic">P =</text>
<path d="M84 284 H79 V347 H84 M166 284 H171 V347 H166" stroke="#4f5660" fill="none"/>
<rect x="91" y="285" width="29" height="28" rx="2" fill="#edf0f3"/><rect x="129" y="285" width="29" height="28" rx="2" fill="#750014"/><rect x="91" y="319" width="29" height="28" rx="2" fill="#750014"/><rect x="129" y="319" width="29" height="28" rx="2" fill="#edf0f3"/>
<g text-anchor="middle" font-size="17"><text x="105" y="305">0</text><text x="143" y="305" fill="white">1</text><text x="105" y="339" fill="white">1</text><text x="143" y="339">0</text></g>
<text x="188" y="305">1번과 2번</text><text x="188" y="326">좌표 교환</text>
</g></svg>'''.replace('座標','좌표 교환').replace('2つ','두 좌표를 섞지 않고 맞바꿈')
HM=hungarian_diagram()
slide('헝가리안 · 원래 activation 좌표의 일대일 대응',
 '<div class="hungarian-three"><div class="hungarian-math">'+
 eq('hungarian-objective',r'\max_P\;\sum_{i,j}C_{ij}P_{ij}',65)+
 eq('hungarian-constraints',r'P_{ij}\in\{0,1\}',39)+
 eq('hungarian-rowcol',r'\sum_jP_{ij}\leq1,\quad\sum_iP_{ij}\leq1',55)+
 eq('hungarian-count',r'\sum_{i,j}P_{ij}=\min(d_I,d_T)',56)+
 '<p class="definitions">W=P이며, 1은 연결된<br>activation 좌표 쌍을 나타냅니다.</p></div>'+
 '<div class="explain"><h3>목적</h3><p>선택한 activation 좌표 쌍들의 <b>coactivation correlation 합을 최대화</b>합니다.</p>'+
 '<h3>제약</h3><p>한 activation 좌표는 상대 좌표와 <b>최대 하나만 연결</b>합니다. 좌표 수가 적은 쪽은 모두 연결합니다.</p>'+
 '<h3>유사도와의 연결</h3><p>함께 변하는 원래 좌표끼리 짝지으면, <b>정답 쌍의 변환 후 평균 내적</b>이 커집니다.</p><p style="font-size:18px;margin-top:12px">좌표별로 표준화했을 때, <b>상관계수에 연결 가중치를 곱해 더한 값</b>이 이 평균 내적과 같기 때문입니다.</p><img class="equation" src="assets/2026-10-06-mapping/hungarian-innerproduct.svg" style="height:43px;max-width:100%;margin-top:10px" alt="상관계수 가중합은 정답 쌍의 변환 후 평균 내적과 같음"></div>'+
 '<div class="hungarian-visual"><h3>두 activation 좌표의 교환</h3>'+HM+'</div></div>'+
 '<div class="bottom">좌표마다 하나의 상대만 허용하므로, 여러 좌표에 나뉜 대응을 직접 표현하기 어렵습니다.</div>',kind='hungarian-slide',note='그림은 2차원 완전 대응에서 두 activation 좌표를 교환하는 순열의 예시다. 정답 텍스트 벡터를 (1,2)로 놓고 이미지 벡터 (2,1)이 같은 위치로 이동하는 모습을 보인다. 하나의 표본이 대응행렬을 결정한다는 뜻은 아니며 같은 P를 모든 표본에 적용한다.')
method('Procrustes · 정답 쌍의 거리 최소화',
 eq('procrustes-objective',r'\min_Q\;\|XQ-Y\|_F^2',75)+
 eq('procrustes-expanded',r'=\min_Q\;\sum_{n=1}^{N}\|x_nQ-y_n\|_2^2',86)+
 eq('procrustes-constraint',r'Q^{\mathsf{T}}Q=I,\qquad W=Q',54)+
 eq('procrustes-identity',r'\|xQ-y\|_2^2=\|x\|_2^2+\|y\|_2^2-2(xQ)y^{\mathsf{T}}',65),
 'Q는 activation을 변환하는 행렬입니다.<br>차원이 다르면 0으로 채워 같은 차원으로 맞춥니다.',
 '변환한 이미지 activation과 대응하는 텍스트 activation 사이의 <b>L2 거리의 제곱을 모든 쌍에 대해 더한 값</b>을 최소화합니다.',
 '<b>길이와 각도를 보존하는 회전·반사</b>만 허용합니다.',
 'Q가 길이를 보존하므로, <b>정답 쌍의 거리를 줄이면 내적이 커집니다.</b>',
 '벡터 길이가 표본마다 다르면, 평균 내적을 키우는 것과 평균 코사인 유사도를 키우는 것은 같지 않습니다.', visual=PV, note='그림은 모든 이미지 벡터에 같은 45도 회전을 적용하면 텍스트 벡터와 정확히 일치하도록 만든 설명용 예시다. 실제 실험에서는 회전만으로 잔차가 모두 없어지지 않을 수 있다. 각 점은 activation 벡터 한 개이며, 원점에서 점까지의 길이와 벡터들 사이의 각도는 보존된다.')
def pls_diagram(cca=False):
    import numpy as np
    eq("pls-scalar-image",r"u=xA=a_1x_1+a_2x_2",28)
    eq("pls-scalar-text",r"v=yB=b_1y_1+b_2y_2",28)
    # Same paired scalar signal occupies opposing coordinate directions.
    latent=np.array([-2.,-1.,0.,1.,2.])
    nx=np.array([1.,-2.,2.,-2.,1.])*.20
    ny=np.array([1.,0.,-2.,0.,1.])*.20
    x=np.outer(latent,[.8,.6])+np.outer(nx,[-.6,.8])
    y=np.outer(latent,[-.6,.8])+np.outer(ny,[-.8,-.6])
    x=(x-x.mean(0))/x.std(0);y=(y-y.mean(0))/y.std(0)
    c=np.einsum('ni,nj->ij',x,y)/len(x)
    aa,_,bt=np.linalg.svd(c);aa=aa[:,0];bb=bt.T[:,0]
    if cca:
        def whiten(z):
            val,vec=np.linalg.eigh(z.T@z/len(z))
            return (vec/np.sqrt(val))@vec.T
        wx,wy=whiten(x),whiten(y)
        left,_,right=np.linalg.svd(wx@c@wy)
        aa=wx@left[:,0];bb=wy@right.T[:,0]
    if aa.sum()<0:aa=-aa;bb=-bb
    u=np.einsum('ni,i->n',x,aa);v=np.einsum('ni,i->n',y,bb)
    parts=['<svg class="pls-diagram" viewBox="0 0 280 450" role="img" aria-label="画像とテキスト"><g font-family="inherit" font-size="12" fill="#4f5660"><text x="5" y="16" font-size="15" font-weight="650" fill="#750014">変換前</text>']
    for offset,pts,w,title,letter,color in [(0,x,aa,'이미지 activation','A','#750014'),(143,y,bb,'텍스트 activation','B','#2584a4')]:
        ox=offset+65;oy=102
        parts.append(f'<text x="{offset+5}" y="43" fill="{color}" font-weight="650">{title}</text><path d="M{ox-53} {oy} H{ox+54} M{ox} {oy-49} V{oy+50}" stroke="#cbd1d9" fill="none"/>')
        dx,dy=w/np.linalg.norm(w)*63
        parts.append(f'<path d="M{ox-dx} {oy+dy} L{ox+dx} {oy-dy}" stroke="{color}" stroke-width="1.4" fill="none"/><text x="{offset+43}" y="159" fill="{color}">{letter} 방향</text>')
        for i,pt in enumerate(pts):
            px=ox+pt[0]*25;py=oy-pt[1]*25
            point_fill=color if letter == "A" else "white"
            parts.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{point_fill}" stroke="{color}" stroke-width="1.5"/><text x="{px+6}" y="{py+10}" fill="{color}" font-size="10">{i+1}</text>')
    parts.append('<text x="5" y="173">같은 번호가 정답 쌍 · 조합 1개(r=1)의 예시</text><image href="assets/2026-10-06-mapping/pls-scalar-image.svg" x="5" y="188" width="255" height="28"/><image href="assets/2026-10-06-mapping/pls-scalar-text.svg" x="5" y="224" width="255" height="28"/><text x="5" y="275">각 점의 두 좌표를 가중합한 결과가 u·v</text><text x="5" y="302" font-size="15" font-weight="650" fill="#750014">조합 후 · 같은 눈금에서 u·v 비교</text><path d="M38 330 H250 M38 372 H250" stroke="#bdc6cf"/><text x="5" y="334" fill="#750014">u</text><text x="5" y="376" fill="#2584a4">v</text>')
    for i,(uu,vv) in enumerate(zip(u,v)):
        px=144+uu*43;qx=144+vv*43
        parts.append(f'<path d="M{px} 330 L{qx} 372" stroke="#b7c0c9" stroke-dasharray="3 3"/><circle cx="{px}" cy="330" r="4.5" fill="#750014"/><circle cx="{qx}" cy="372" r="4.5" fill="white" stroke="#2584a4" stroke-width="1.8"/><text x="{px}" y="318" text-anchor="middle" font-size="10" fill="#750014">{i+1}</text><text x="{qx}" y="390" text-anchor="middle" font-size="10" fill="#2584a4">{i+1}</text>')
    parts.append('<text x="36" y="411">−2</text><text x="141" y="411">0</text><text x="239" y="411">2</text><text x="5" y="442">공분산을 높여 정답 쌍의 정렬을 유도</text></g></svg>')
    if cca:
        parts=[part.replace('각 점의 두 좌표를 가중합한 결과가 u·v','조합 결과 u·v의 평균 0, 분산 1').replace('공분산을 높여 정답 쌍의 정렬을 유도','상관계수를 높여 정답 쌍의 정렬을 유도') for part in parts]
    return ''.join(parts).replace('画像とテキスト','서로 다른 방향의 이미지·텍스트 activation을 A와 B로 조합해 같은 축에서 정답 쌍의 값을 비교하는 설명용 예시').replace('変換前','조합 전 · 서로 다른 활성화 방향')
PLSV=pls_diagram()
method('PLS-SVD · 함께 변하는 (공분산) activation 조합의 탐색',
 eq('pls-objective',r'\max_{A,B}\;\operatorname{tr}(A^{\mathsf{T}}CB)',65)+
 eq('pls-cov',r'=\max_{A,B}\;\sum_{k=1}^{r}\operatorname{Cov}((XA)_{:k},(YB)_{:k})',88)+
 eq('pls-constraints',r'A^{\mathsf{T}}A=I,\quad B^{\mathsf{T}}B=I',55)+
 eq('pls-dimensions',r'A\in\mathbb{R}^{d_I\times r},\quad B\in\mathbb{R}^{d_T\times r},\quad W=AB^{\mathsf{T}}',54),
 'A·B의 각 열은 여러 activation을 섞는 가중치입니다.<br>r은 조합 수, :k는 전체 표본의 k번째 조합 값입니다.<br>Cov는 공분산, tr은 대각 원소의 합입니다.',
 'A와 B를 구해, <b>같은 번호의 r개 조합 사이 공분산의 합</b>을 최대화합니다.',
 '<b>가중치 열의 길이는 1</b>이고, 서로 다른 열의 내적은 0입니다.',
 '평균을 뺀 activation에서는 <b>공분산의 합이 정답 쌍의 평균 내적과 같습니다.</b></p>' +
 eq('pls-cov-innerproduct',r'\sum_{k=1}^{r}\operatorname{Cov}(u_k,v_k)=\frac{1}{N}\sum_{n=1}^{N}u_n v_n^{\mathsf{T}}',72) +
 '<p class="cov-note">u<sub>n</sub>=x<sub>n</sub>A, v<sub>n</sub>=y<sub>n</sub>B는 n번째 정답 쌍의 조합 벡터입니다.',
 '원래 좌표의 상위 r개를 고르는 것이 아닙니다. 여러 activation을 가중합한 새 조합을 r개 만듭니다.',
 'PLS는 Partial Least Squares이다. 중심화된 출력에서 공분산은 N으로 나눈 값을 사용한다. C=UΣVᵀ의 상위 r개 좌우 특이벡터가 A·B이다. 최대화 대상은 전체 표본에서의 공분산 합이다. 그림은 표준화한 2차원 합성 자료의 cross covariance에 SVD를 적용한 예시다. 이미지와 텍스트는 서로 다른 방향에서 변하며 점 번호로 정답 쌍을 표시했다. 아래 두 가로축은 같은 눈금으로 조합 값을 표시한다. 공분산 최대화가 모든 쌍의 값 일치나 코사인 검색 개선을 보장하는 것은 아니다.',visual=PLSV)
CCAV=pls_diagram(cca=True)
method('CCA · 변동 크기를 맞춘 activation 조합의 탐색',
 eq('cca-objective',r'\max_{A,B}\;\operatorname{tr}(A^{\mathsf{T}}CB)',65)+
 eq('cca-corr',r'=\max_{A,B}\;\sum_{k=1}^{r}\operatorname{Corr}((XA)_{:k},(YB)_{:k})',88)+
 eq('cca-constraints',r'\frac{(XA)^{\mathsf{T}}(XA)}{N}=I,\quad\frac{(YB)^{\mathsf{T}}(YB)}{N}=I',75)+
 eq('cca-dimensions',r'A\in\mathbb{R}^{d_I\times r},\quad B\in\mathbb{R}^{d_T\times r},\quad W=AB^{\mathsf{T}}',54),
 'A·B는 여러 activation을 조합하는 가중치 행렬입니다.<br>r은 조합 수, Corr는 상관계수입니다.',
 'A와 B를 구해, <b>같은 번호의 r개 조합 사이 상관계수의 합</b>을 최대화합니다.',
 '<b>출력 분산은 1</b>이고, 같은 모달리티의 서로 다른 조합 사이 상관은 0입니다.',
 '각 조합의 평균이 0, 분산이 1이므로 <b>상관계수의 합이 정답 쌍의 평균 내적과 같습니다.</b></p>' +
 eq('cca-corr-innerproduct',r'\sum_{k=1}^{r}\operatorname{Corr}(u_k,v_k)=\frac{1}{N}\sum_{n=1}^{N}u_n v_n^{\mathsf{T}}',72) +
 '<p class="cov-note">u<sub>n</sub>=x<sub>n</sub>A, v<sub>n</sub>=y<sub>n</sub>B는 n번째 정답 쌍의 조합 벡터입니다.',
 'PLS-SVD는 가중치 길이를, CCA는 조합한 activation의 분산을 맞춥니다. 위 식은 안정화 항을 제외한 기본 CCA입니다.',note='그림은 합성 자료에 기본 CCA를 적용한 조합 한 쌍의 예시다. 각 출력의 표본 분산은 N으로 나누어 1이다. 평균 내적의 최대화는 코사인 검색 성능을 보장하지 않는다.',visual=CCAV)
def sinkhorn_diagram():
    import numpy as np
    c=np.array([[.85,.30,.15],[.30,.75,.30],[.15,.30,.85]])
    eps=.3
    kernel=np.exp(c/eps)
    v=np.ones(3)
    for _ in range(300):
        u=(np.ones(3)/3)/(kernel@v)
        v=(np.ones(3)/3)/(kernel.T@u)
    p=u[:,None]*kernel*v[None,:]
    parts=['<svg class="pls-diagram" viewBox="0 0 280 450" role="img" aria-label="3つの座標"><g font-family="inherit" font-size="12" fill="#4f5660">']
    for top,mat,title,scale in [(0,c,'coactivation correlation',1),(226,p,'学習した接続の重み',.333333333)]:
        parts.append(f'<text x="5" y="{top+16}" fill="#750014" font-size="15" font-weight="650">{title}</text><text x="90" y="{top+38}" fill="#2584a4">텍스트 activation</text><text x="3" y="{top+62}" fill="#750014">이미지</text>')
        for i in range(3):
            parts.append(f'<circle cx="31" cy="{top+87+i*39}" r="4" fill="#750014"/><text x="8" y="{top+92+i*39}">{i+1}</text>')
            for j in range(3):
                value=mat[i,j]; strength=value/scale
                rgb=tuple(round(248+(v-248)*strength) for v in (117,0,20))
                fill='#'+''.join(f'{v:02x}' for v in rgb)
                fg='white' if strength>.57 else '#1a1d21'
                parts.append(f'<rect x="{52+j*49}" y="{top+69+i*39}" width="47" height="37" rx="3" fill="{fill}"/><text x="{75+j*49}" y="{top+93+i*39}" text-anchor="middle" fill="{fg}">{value:.2f}</text>')
        for j in range(3):
            parts.append(f'<circle cx="{70+j*49}" cy="{top+56}" r="4" fill="white" stroke="#2584a4" stroke-width="1.5"/><text x="{80+j*49}" y="{top+60}" fill="#2584a4">{j+1}</text>')
        if top:
            for i in range(3):
                parts.append(f'<text x="210" y="{top+93+i*39}">합 1/3</text>')
            for j in range(3):
                parts.append(f'<text x="{75+j*49}" y="{top+207}" text-anchor="middle">합 1/3</text>')
    parts.append('<path d="M126 190 V209 M121 204 L126 210 L131 204" fill="none" stroke="#7c858d"/><text x="145" y="204">Sinkhorn</text></g></svg>')
    return ''.join(parts).replace('3つの座標','세 이미지·텍스트 activation 좌표에 Sinkhorn을 적용한 예시. 행과 열의 합은 각각 3분의 1이며 소수 둘째 자리로 표시').replace('学習した接続の重み','연결 가중치 · 여러 좌표에 분배')
SINKV=sinkhorn_diagram()
method('Sinkhorn · 원래 activation 좌표의 가중 대응',
 eq('sinkhorn-objective',r'\max_{P\geq0}\;\left[\sum_{i,j}C_{ij}P_{ij}+\varepsilon H(P)\right]',82)+
 eq('sinkhorn-expanded',r'=\max_{P\geq0}\;\left[\frac{1}{N}\sum_n(x_nP)y_n^{\mathsf{T}}\right.',64)+
 eq('sinkhorn-entropy-line',r'\left.{}-\varepsilon\sum_{i,j}P_{ij}\log P_{ij}\right]',57)+
 eq('sinkhorn-constraints',r'P\mathbf{1}=a,\quad P^{\mathsf{T}}\mathbf{1}=b,\quad W=P',53),
 'P<sub>ij</sub>는 두 activation 좌표의 연결 가중치입니다.<br>a는 각 행의 합을 이미지 activation 좌표 수의 역수로, b는 각 열의 합을 텍스트 activation 좌표 수의 역수로 균등하게 고정했습니다.<br>ε는 가중치의 과도한 집중을 완화하는 강도입니다.',
 '<b>coactivation correlation이 높은 좌표 쌍에 큰 가중치</b>를 배정하면서, 가중치의 과도한 집중을 완화합니다.',
 '가중치는 음수가 아니며, <b>각 행·열의 총가중치를 고정</b>합니다.',
 '<b>상관계수에 연결 가중치를 곱해 더한 값</b>은 정답 쌍의 변환 후 평균 내적과 같습니다.</p>' +
 eq('sinkhorn-innerproduct',r'\sum_{i,j}C_{ij}P_{ij}=\frac{1}{N}\sum_n(x_nP)y_n^{\mathsf{T}}',60)+'<p>',
 '헝가리안의 0·1 연결을 연속 가중치로 완화합니다.',note='그림은 대칭인 3×3 상관행렬에 epsilon=0.3과 균등 주변합을 사용하여 Sinkhorn 반복으로 계산했다. 표시된 값은 반올림했다. 정답 쌍의 평균 내적과 엔트로피의 합을 최적화하며 평균 내적만의 최대화와는 다르다.',visual=SINKV)

slide('어떤 방법론을 목표로 해야할까? 해석과 정렬이 모두 개선된 방법론','''<div class="scope-block">
<h3>1. 해석: 개념과 activation의 다대다 대응을 표현</h3>
<p>여러 activation이 하나의 개념을 나타내거나,<br>하나의 activation이 여러 개념에 관여하는 관계를 표현합니다.</p>
</div><div class="scope-block">
<h3>2. 정렬: 이미지·텍스트 activation을 대응시켜 유사도 기반으로 정렬</h3>
<p>대응 행렬 W를 이용해 비교했을 때, 같은 의미의 이미지와 텍스트 activation이 같은 위치의 좌표를 활성화해야 합니다 <b>(정답 이미지·텍스트 쌍의 유사도가 높아야 함)</b></p>
</div>''','scope')
slides[-1]=slides[-1].replace('진행 상황 공유 · Baselines','진행 상황 공유 · Method Requirements')
slides[0]=slides[0].replace('진행 상황 공유 · Baselines','진행 상황 공유 · 2. 정렬 · Baselines')

# Keep the continuous coordinate matching method directly after Hungarian.
slides=[slides[i] for i in (6,0,1,5,2,3,4)]
slides=[re.sub(r'id="slide-\d+"',f'id="slide-{i}"',markup,count=1)
        for i,markup in enumerate(slides,1)]

# Follow the baseline explanations with interpretation-oriented sparse extensions.
slide('Soft 대응의 해석과 희소화','''<div class="sparse-intro">
<div><h3>헝가리안 · activation 하나와 하나의 대응</h3><p>어떤 이미지 activation이 어떤 텍스트 activation에 연결되는지 개별 연결을 확인하기 쉬운 구조</p></div>
<div><h3>Soft 대응 · 여러 activation의 연결 또는 조합</h3><p>여러 activation을 함께 활용할 수 있지만, 참여하는 activation이 많아질수록 대응의 의미를 설명하기 어려운 구조</p></div>
<div class="sparse-question"><h3>아이디어: 소수의 activation만 연결하거나 조합하면 어떨까?</h3><p><b>일대일 제약을 완화하면서도, 대응에 참여하는 activation을 하나씩 확인할 수 있도록 희소성(sparsity)을 부여</b></p></div>
</div><div class="bottom">희소성은 해석할 대상을 줄여 <b>해석 가능성을 높이기 위한 방향</b></div>''','sparse-section')
sparse_example=eq('sparse-concept-example',r'u_1=a_{3,1}x_3+a_{17,1}x_{17},\qquad v_1=b_{8,1}y_8+b_{42,1}y_{42}',59)
slide('예시: Sparse CCA',f'''<div class="sparse-case">
<div><h3>여러 activation 중 일부만 선택하여 조합</h3><p>예를 들어, 이미지와 텍스트에서 각각 <b>activation 2개만 골라 가중합으로 조합</b>하고, 정답 쌍에서 두 조합의 상관계수가 높아지도록 학습</p>
{sparse_example}
<p class="sparse-example-note">이미지의 3·17번 activation과 텍스트의 8·42번 activation을 고른 예시. 고른 activation과 가중치는 모든 표본에 동일하게 적용</p>
<h3 class="space">같은 개념에 대응하는 두 조합의 해석</h3><p>u_1과 v_1이 모두 같은 개념에 반응한다면, <b>u_1은 그 개념의 이미지 표현, v_1은 그 개념의 텍스트 표현으로 해석 가능</b></p>
<p class="sparse-example-note space">어떤 activation을 고르고 얼마나 반영할지는 학습으로 결정하며, 두 조합이 같은 개념을 나타내는지는 별도로 검증</p></div>
<div class="sparse-case-visual"><h3>CCA에서 두 조합을 만드는 과정</h3>{CCAV}</div>
</div><div class="bottom">Sparse PLS-SVD도 같은 형태의 집합 쌍을 구성하되, <b>상관계수 대신 공분산을 높이는 방식</b></div>''','sparse-section')
slide('희소화 대상에 따른 대응의 해석','''<table class="sparse-table sparse-ref-table"><thead><tr><th>방법</th><th>희소하게 만드는 대상</th><th>해석할 수 있는 대응</th></tr></thead><tbody>
<tr><td><b><span class="sparse-check" aria-label="기존 희소 방법 있음">✓</span> Sparse CCA [1]</b></td><td>양쪽 조합 행렬 A·B의 각 열에서 사용하는 activation 수 제한</td><td>소수의 이미지 activation 집합과 소수의 텍스트 activation 집합의 대응</td></tr>
<tr><td><b><span class="sparse-check" aria-label="기존 희소 방법 있음">✓</span> Sparse PLS-SVD [2]</b></td><td>양쪽 조합 행렬 A·B의 각 열에서 사용하는 activation 수 제한</td><td>같은 형태의 집합 쌍을 구성하되, 상관계수 대신 공분산을 높이는 대응</td></tr>
<tr><td><b><span class="sparse-check" aria-label="기존 희소 방법 있음">✓</span> 희소 최적 수송 [3, 4]</b></td><td>연결 행렬 P에서 0이 아닌 연결 수 제한</td><td>원래 이미지 activation 하나에 연결된 소수의 텍스트 activation을 확인</td></tr>
<tr><td><b><span class="sparse-cross" aria-label="최적화가 어려운 설계 후보">✕</span> 희소한 Procrustes 대응</b><small class="design-label">희소성과 직교성을 함께 만족하도록 최적화하기 어려움</small></td><td>직교행렬 Q에서 0이 아닌 계수 수 제한</td><td>소수의 이미지 activation 조합과 원래 텍스트 activation 하나의 대응</td></tr>
</tbody></table><p class="sparse-reference-note">✓ 기존 희소 방법 있음　✕ 단순한 계수 제거로 구현하기 어려움. 희소 최적 수송은 기본 Sinkhorn의 정규화 또는 제약을 변경</p><div class="bottom sparse-references"><div>[1] Witten et al. (2009). <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC2697346/" target="_blank">A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis.</a></div><div>[2] Lê Cao et al. (2008). <a href="https://pubmed.ncbi.nlm.nih.gov/19049491/" target="_blank">A sparse PLS for variable selection when integrating omics data.</a></div><div>[3] Blondel et al. (2018). <a href="https://proceedings.mlr.press/v84/blondel18a.html" target="_blank">Smooth and Sparse Optimal Transport.</a></div><div>[4] Liu et al. (2023). <a href="https://arxiv.org/abs/2209.15466" target="_blank">Sparsity-Constrained Optimal Transport.</a></div></div>''','sparse-section',note='인용은 희소화 아이디어의 선행 연구이며 현재 코드가 해당 알고리즘을 그대로 재현했다는 의미는 아니다. Witten 등의 PMD 기반 sparse CCA는 모달리티 내 공분산을 단위행렬로 대체하는 변형이다. 2개 선택 예시는 개수 제한을 설명하는 예시이며 L1 제약과 동일한 명세가 아니다. 여기서 희소 Procrustes는 Q 계수의 희소성과 직교성을 함께 요구하는 설계 방향이다. Rotational Lasso의 회전 수 희소성과 동일하지 않다. Sinkhorn은 기본 엔트로피 정규화만으로 정확한 희소 연결을 만들지 않으므로 희소 최적 수송 등의 수정이 필요하다. 두 방법도 연결망 차원에서 다대다일 수 있으나 CCA처럼 명시적인 양쪽 공통 조합 좌표가 없다는 구분이다.')
for i in range(7,len(slides)):
    slides[i]=slides[i].replace('진행 상황 공유 · Baselines','진행 상황 공유 · 해석을 위한 희소 대응')

# Read the saved standardized-input comparisons, with a fixed method list.
import csv,json
RUNS=Path('/Users/jihoonkwon/Desktop/projects/research/MM-SAE/mm-sae/runs')
conditions=[
 ('COCO2017','COCO2017','mapping-ablation-2026-10-04'),
 ('CC3M','CC3M','mapping-ablation-cc3m-2026-10-04'),
 ('CC3M','COCO2017','cc3m-followup-2026-10-05/coco-fit/ablation')]
method_rows=[
 ('헝가리안','hungarian__standardized__text_projected_to_image'),
 ('Sinkhorn<small>텍스트 변환</small>','sinkhorn__standardized__text_projected_to_image'),
 ('Sparse Sinkhorn<small>텍스트 변환 · 16개</small>','sparse_sinkhorn_text'),
 ('Sinkhorn<small>이미지 변환</small>','sinkhorn__standardized__image_projected_to_text'),
 ('Sparse Sinkhorn<small>이미지 변환 · 16개</small>','sparse_sinkhorn_image'),
 ('Procrustes','procrustes'),('PLS-SVD','cross_svd_256'),
 ('Sparse PLS-SVD<small>조합당 16개</small>','sparse_pls'),('CCA','cca_256'),
 ('Sparse CCA<small>조합당 16개</small>','sparse_cca')]
cards=[];provenance=[]
for sae_data,map_data,folder in conditions:
    source=RUNS/folder/'retrieval_summary.csv'
    rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
    values=[]
    supplemental = (RUNS/'cc3m-followup-2026-10-05'/('cc3m-fit' if map_data=='CC3M' else 'coco-fit')) if sae_data=='CC3M' else None
    pruning = supplemental/'pruning' if supplemental else RUNS/'mapping-pruning-2026-10-04'
    semantics = supplemental/'semantics' if supplemental else RUNS/'mapping-semantics-2026-10-04'
    pls_condition = ('cc3m-' if sae_data=='CC3M' else 'coco-') + ('cc3m' if map_data=='CC3M' else 'coco')
    sparse_paths = {
      'sparse_sinkhorn_text': pruning/'sinkhorn/results/sinkhorn_e0.05_k16__text_projected_to_image.json',
      'sparse_sinkhorn_image': pruning/'sinkhorn/results/sinkhorn_e0.05_k16__image_projected_to_text.json',
      'sparse_cca': semantics/'results/sparse_cca_16.json',
      'sparse_pls': RUNS/'sparse-pls-2026-10-06'/pls_condition/'results/sparse_pls_16.json',
    }
    for label,key in method_rows:
        vals=[]
        if key in sparse_paths:
            path=sparse_paths[key]
            if path.exists():
                result=json.loads(path.read_text())
                for direction in ['image_to_text','text_to_image']:
                    metrics=result['retrieval'][direction]
                    assert {metrics['query_count'],metrics['candidate_count']}=={5000,25014}
                    vals.extend(metrics['recall'][str(k)]*100 for k in [1,5,10])
                    provenance.append({'source':str(path),'sae_training':sae_data,'mapping_training':map_data,
                                       'key':key,'direction':direction,'recall':metrics['recall']})
            else:
                vals=[None]*6
            values.append(vals)
            continue
        for direction in ['image_to_text','text_to_image']:
            matches=[r for r in rows if r['key']==key and r['direction']==direction]
            assert len(matches)==1,(source,key,direction)
            row=matches[0]
            assert {int(row['query_count']),int(row['candidate_count'])}=={5000,25014}
            vals.extend(float(row[f'recall_at_{k}'])*100 for k in [1,5,10])
            provenance.append({'source':str(source),'sae_training':sae_data,'mapping_training':map_data,**row})
        values.append(vals)
    maxima=[max(round(v[c],2) for v in values if v[c] is not None) for c in range(6)]
    sparse_maxima=[max((round(v[c],2) for (_,key),v in zip(method_rows,values)
                       if key.startswith('sparse_') and v[c] is not None), default=None) for c in range(6)]
    sparse_seconds=[]
    for c in range(6):
        ranks=sorted({round(v[c],2) for (_,key),v in zip(method_rows,values)
                      if key.startswith('sparse_') and v[c] is not None}, reverse=True)
        sparse_seconds.append(ranks[1] if len(ranks)>1 else None)
    body=''
    for i,((label,key),vals) in enumerate(zip(method_rows,values)):
        cells=''
        for c,v in enumerate(vals):
            if v is None:
                cells+='<td class="pending-result" title="실험 결과 대기">&nbsp;</td>'
                continue
            t=f'{v:.2f}'
            if key.startswith('sparse_') and round(v,2)==sparse_maxima[c]:
                t=f'<b class="sparse-best">{t}</b>'
            elif key.startswith('sparse_') and round(v,2)==sparse_seconds[c]:
                t=f'<span class="sparse-second">{t}</span>'
            elif round(v,2)==maxima[c]:t=f'<b>{t}</b>'
            cells+=f'<td>{t}</td>'
        rowclass=' class="sparse-result"' if key.startswith('sparse_') else (' class="soft-start"' if i==1 else '')
        body+=f'<tr{rowclass}><th>{label}</th>{cells}</tr>'
    cards.append(f'<div class="result-card"><h3>SAE 학습 · {sae_data}<br>대응 학습 · {map_data}</h3><table class="baseline-table"><colgroup><col style="width:28%"><col span="6" style="width:12%"></colgroup><thead><tr><th rowspan="2">방법</th><th colspan="3">이미지로 텍스트 검색</th><th colspan="3">텍스트로 이미지 검색</th></tr><tr><th>R@1</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@5</th><th>R@10</th></tr></thead><tbody>{body}</tbody></table></div>')
(ASSETS/'baseline-results.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
slide('16개 activation으로 제한한 희소 대응의 검색 성능', '<div class="baseline-grid">'+''.join(cards)+'</div><ul class="sparse-takeaways"><li><b>희소 Sinkhorn은 검색 후보 쪽을 변환할 때 더 높았음</b> · 세 학습 조건의 양방향 Recall@5에서 공통으로 관찰</li><li><b>Sinkhorn의 약한 연결 제거가 검색에 도움이 될 가능성</b> · 다만 모든 조건과 검색 방향에서 개선되지는 않음</li><li><b>소수의 activation을 조합하는 방식 중 Sparse CCA가 유망</b> · 대응을 COCO로 학습한 두 조건에서 Sparse PLS-SVD보다 양방향 Recall@5가 높았음</li><li><b>같은 16개 제한도 자료와 방법에 따라 성능 감소 폭이 달랐음</b> · SAE·대응 모두 CC3M인 CCA의 이미지 질의 Recall@5는 36.34%에서 23.86%로 감소</li></ul><div class="bottom"><span>Sinkhorn ε=0.05 · 텍스트 변환은 이미지 좌표로, 이미지 변환은 반대 · PLS-SVD·CCA 공통 좌표 256개 · 희소 PLS-SVD·CCA 조합당 최대 16개 activation · 희소 Sinkhorn 행·열당 최대 16개 연결</span></div>','sparse-baseline-results',note='희소 Sinkhorn은 학습 후 mutual top16 연결 제거이며 인용한 sparse OT 알고리즘의 재현이 아니다. Sparse PLS는 hard top16 교대 최적화와 순차 잔차 제거를 쓰는 탐색 구현이며 L1 sparse PLS의 재현이 아니다. Sparse CCA도 기존 cardinality ridge CCA 탐색 구현이다. 결과는 각 조건의 retrieval_summary.csv에서 고정된 방법과 표준화 조건으로 읽었다. Procrustes는 기존 ablation의 common 조건이다. COCO SAE는 top-k 8 및 30 epochs, CC3M SAE는 top-k 32 및 10 epochs여서 데이터만 바꾼 통제가 아니다. PLS-SVD는 저장된 cross_svd_256 결과이다. 선택하지 않은 raw/centered 조건과 결과가 다르므로 앞선 조건 혼합 표와 직접 동일시하지 않는다. 굵은 값은 소수 둘째 자리 표시 기준의 최댓값이다.')
# Keep the sparse comparison after the sparse-method explanation; restore dense baselines before it.
def rank_dense_card(card):
    card = re.sub(r'<tr class="sparse-result">.*?</tr>', '', card, flags=re.S)
    cells = re.findall(r'<td>(.*?)</td>', card, flags=re.S)
    numbers = [float(re.sub(r'<[^>]+>', '', cell)) for cell in cells]
    ranks = [sorted(set(numbers[c::6]), reverse=True) for c in range(6)]
    index = 0
    def decorate(match):
        nonlocal index
        value = numbers[index]
        column = index % 6
        index += 1
        text = f'{value:.2f}'
        if value == ranks[column][0]:
            text = f'<b>{text}</b>'
        elif len(ranks[column]) > 1 and value == ranks[column][1]:
            text = f'<span class="sparse-second">{text}</span>'
        return f'<td>{text}</td>'
    return re.sub(r'<td>.*?</td>', decorate, card, flags=re.S)
dense_cards = [rank_dense_card(card) for card in cards]
slide('학습 자료에 따른 Baselines의 검색 성능', '<div class="baseline-grid">'+''.join(dense_cards)+'</div><p class="result-message"><b>공분산을 최대화하는 PLS-SVD는 세 조건 모두 헝가리안·Sinkhorn·Procrustes보다 양방향 검색 성능이 높음</b></p><p class="result-qualification">CCA도 대체로 높지만, SAE와 대응을 모두 CC3M으로 학습한 조건에서는 Procrustes보다 낮은 항목이 있음</p><div class="bottom"><span>Sinkhorn의 텍스트 변환은 텍스트를 이미지 좌표로, 이미지 변환은 그 반대 · ε=0.05<br>PLS-SVD·CCA는 공통 좌표 256개 사용</span></div>',note='희소화 전 결과만 비교한다. 각 조건의 retrieval_summary.csv에서 읽은 값이다. COCO SAE와 CC3M SAE의 학습 조건도 달라 자료만의 효과로 해석하지 않는다.')
slides.insert(7,slides.pop())
slides=[re.sub(r'id="slide-\d+"',f'id="slide-{i}"',markup,count=1) for i,markup in enumerate(slides,1)]

# Explain the retrieval protocol immediately before the results.
caption_source=RUNS.parent/'data/coco/annotations/captions_val2017.json'
coco_captions=json.loads(caption_source.read_text())
caption=next(item['caption'] for item in coco_captions['annotations'] if item['image_id']==285 and item['id']==662422)
rank_boxes=''.join(f'<div class="rank-box {"correct" if i==3 else ""}"><span>{i}위</span><b>{"정답" if i==3 else "후보"}</b></div>' for i in range(1,11))
slide('SAE activation을 이용한 크로스모달 검색',f'''<p class="retrieval-lead">일반적인 검색은 이미지·텍스트 <b>임베딩의 유사도</b>를 비교 · 이번 실험은 <b>SAE의 희소 activation을 표준화하고 대응시킨 뒤</b> 유사도로 검색</p>
<div class="training-flow" aria-label="모달리티별 SAE 학습, SAE 고정 후 대응행렬 학습, 고정한 모델로 검색 평가 순서">
<div class="training-stage"><h4>1. 모달리티별 SAE 학습</h4><div class="sae-branches"><span class="image-sae">이미지 임베딩으로 이미지 SAE 학습</span><span class="text-sae">텍스트 임베딩으로 텍스트 SAE 학습</span></div></div>
<div class="flow-connector" aria-hidden="true"></div>
<div class="training-stage"><h4>2. SAE 고정 후 대응 행렬 W 학습</h4><p>학습 쌍의 SAE activation을 추출·표준화한 뒤 W 또는 양쪽 조합 행렬 A·B 학습</p></div>
<div class="flow-connector" aria-hidden="true"></div>
<div class="training-stage"><h4>3. 검색 평가</h4><p>SAE와 대응 행렬을 고정하고 평가 이미지·캡션의 유사도 계산</p></div>
</div><div class="retrieval-layout"><div class="retrieval-formulas">
<div class="setup-alignment-row"><div><h3>한쪽 activation을 W로 변환하는 경우</h3>{E[2].replace("height:71px","height:44px")}</div>{D0}</div>
<div class="setup-alignment-row"><div><h3>양쪽을 변환하는 경우</h3>{E[3].replace("height:37px","height:24px")}{E[4].replace("height:70px","height:44px")}</div>{D1}</div>
<p class="retrieval-definition">x·y는 표준화한 activation, A·B는 공통 좌표로 조합하는 행렬</p>
<p class="retrieval-definition space">이미지로 텍스트 검색과 텍스트로 이미지 검색을 모두 평가</p>
</div><div class="retrieval-demo">
<h3>COCO2017의 정답 이미지·캡션 쌍</h3>
<div class="retrieval-pair"><img src="assets/2026-09-16/coco-bear.jpg" alt="COCO val2017의 285번 이미지, 잔디 위의 갈색 곰"><div><p class="caption-tag">정답 캡션</p><p class="caption-text">{html.escape(caption)}</p><p class="caption-tag">같은 이미지에 연결된 캡션은 모두 정답</p></div></div>
<div class="retrieval-step">이미지를 질의로 두고, 전체 캡션 25,014개를 <b>대응 후 cosine similarity 순으로 정렬</b></div>
<div class="rank-heading"><b>상위 10개 캡션</b><span>검색 순위를 설명하는 예시</span></div>
<div class="rank-grid">{rank_boxes}</div>
<p class="rank-reading">정답이 3위라면 <b>Recall@1은 실패, Recall@5·10은 성공</b></p>
</div></div><div class="bottom dataset-footnote">학습 데이터셋 · CC3M, COCO2017 train2017　|　평가 데이터셋 · COCO2017 val2017을 test set으로 사용</div>''','retrieval-setup',note='예시 이미지는 COCO val2017 image_id=285, caption_id=662422의 실제 캡션을 사용했다. 3위라는 순위와 후보 상자는 설명용이며 실제 모델의 검색 결과가 아니다. 이미지 질의에서는 해당 이미지에 연결된 캡션 중 하나 이상이 상위 K에 있으면 성공이고, 텍스트 질의에서는 원래 대응 이미지가 상위 K에 있으면 성공이다. SAE 원본 출력은 희소하지만 평균을 빼는 표준화 후 값은 희소하지 않을 수 있다. 한쪽 변환 수식은 이미지 변환 예시이며 텍스트 변환도 반대 방향으로 적용한다.')
slides[-1]=slides[-1].replace('진행 상황 공유 · Baselines','진행 상황 공유 · Experimental Setup')
slides.insert(7,slides.pop())
slides=[re.sub(r'id="slide-\d+"',f'id="slide-{i}"',markup,count=1) for i,markup in enumerate(slides,1)]

# Refresh the support-size figure from completed results only.
import runpy
runpy.run_path(str(Path(__file__).with_name('plot_support_sweep.py')), run_name='__main__')
sweep_status=json.loads((ASSETS/'sparsity-recall5-status.json').read_text())
pending_note=(f"현재 {sweep_status['completed']}/60개 결과 반영 · 미완료 지점은 빈칸으로 유지" if sweep_status['pending'] else "세 학습 조건의 8·16·32·64·128개 결과 모두 반영")
slide('activation 수에 따른 희소 대응의 검색 성능', '<p style="font-size:18px;margin-bottom:12px"><b>위</b> · 이미지로 텍스트 검색　<b>아래</b> · 텍스트로 이미지 검색 · 공통 좌표는 256개로 고정</p><img src="assets/2026-10-06-mapping/sparsity-recall5.svg" style="display:block;width:100%;height:420px;object-fit:contain" alt="세 학습 조건에서 activation 또는 연결 수에 따른 양방향 Recall@5 선 그래프"><div class="bottom"><p style="font-size:20px;margin-bottom:6px">희소 조건 k를 변화시키며 성능을 비교했을 때, <b>전반적으로 어느 한 방식이 가장 좋다고 말하기는 어려움</b></p><span style="font-size:14px">Sparse PLS-SVD·CCA는 조합당 activation 수, Sinkhorn은 각 좌표에 남기는 최대 연결 수를 변경</span></div>', 'support-curves',note='네 색은 Sinkhorn 텍스트 변환, Sinkhorn 이미지 변환, Sparse PLS-SVD, Sparse CCA를 나타낸다. 실선과 점선은 검색 질의의 모달리티이다. 계열별 희소화 단위가 다르며 미완료 값은 추정하지 않는다.')

slides.append(Path(__file__).with_name('three-matrix-idea.html').read_text())

# Annotation-supervised feature sets, with fixed semantic identity across modalities.
annotation_prediction = eq('annotation-prediction',r'\hat t_{nc}=z_n w_c+b_c',40)
annotation_objective = eq('annotation-objective',r'\min_{w_c}\;\frac{1}{2N}\sum_{n=1}^{N}(z_nw_c+b_c-t_{nc})^2+\frac{\lambda}{2}\|w_c\|_2^2,\quad\lambda=0.01',68)
annotation_dims = eq('annotation-weight-dimensions',r'W_I\in\mathbb{R}^{d_I\times171},\qquad W_T\in\mathbb{R}^{d_T\times171}',48)
annotation_support = eq('annotation-weight-support',r'\|W_I[:,c]\|_0\leq16,\qquad\|W_T[:,c]\|_0\leq16',43)
annotation_scores = eq('annotation-cosine',r's(x,y)=\cos(xW_I,\,yW_T),\qquad W=W_I I_{171}W_T^{\mathsf{T}}',47)
slides.append(Path(__file__).with_name('dataset-recap.html').read_text())
slides.append(Path(__file__).with_name('oracle-method.html').read_text())

annotation_cards=[]
annotation_provenance=[]
for condition, title, annotation_folder in [
    ('coco-coco','SAE 학습 · COCO2017<br>대응 학습 · COCO2017','oracle-sets-2026-10-04'),
    ('cc3m-coco','SAE 학습 · CC3M<br>대응 학습 · COCO2017','cc3m-followup-2026-10-05/coco-fit/annotation-sets')]:
    sources=[
      ('Sparse Sinkhorn<small>텍스트 변환 · 연결 최대 16개</small>', RUNS/'sparsity-sweep-2026-10-06'/condition/'results/sinkhorn_text_16.json'),
      ('Sparse Sinkhorn<small>이미지 변환 · 연결 최대 16개</small>', RUNS/'sparsity-sweep-2026-10-06'/condition/'results/sinkhorn_image_16.json'),
      ('Sparse PLS-SVD<small>공통 좌표 256개 · 조합당 최대 16개</small>', RUNS/'sparsity-sweep-2026-10-06'/condition/'results/pls_16.json'),
      ('Sparse CCA<small>공통 좌표 256개 · 조합당 최대 16개</small>', RUNS/'sparsity-sweep-2026-10-06'/condition/'results/cca_16.json'),
      ('Sparse CCA<small>공통 좌표 171개 · 조합당 최대 16개</small>', RUNS/annotation_folder/'results/sparse_cca16_171.json'),
      ('Oracle 매핑<small>개념 좌표 171개 · 개념당 최대 16개</small>', RUNS/annotation_folder/'results/propagated_presence_16.json')]
    vals=[]
    for label,path in sources:
        data=json.loads(path.read_text())
        v=[100*data['retrieval'][direction]['recall'][str(k)] for direction in ('image_to_text','text_to_image') for k in (1,5,10)]
        vals.append(v)
        annotation_provenance.append(dict(condition=condition,source=str(path),values=v))
    ranks=[sorted({round(v[c],2) for v in vals}, reverse=True) for c in range(6)]
    best=[r[0] for r in ranks]
    second=[r[1] if len(r)>1 else None for r in ranks]
    baseline_csv=RUNS/('mapping-ablation-2026-10-04' if condition=='coco-coco' else 'cc3m-followup-2026-10-05/coco-fit/ablation')/'retrieval_summary.csv'
    baseline_rows=list(csv.DictReader(baseline_csv.open(encoding='utf-8-sig')))
    hungarian_values=[]
    for direction in ('image_to_text','text_to_image'):
        row=next(r for r in baseline_rows if r['key']=='hungarian__standardized__text_projected_to_image' and r['direction']==direction)
        hungarian_values.extend(float(row[f'recall_at_{k}'])*100 for k in (1,5,10))
    annotation_provenance.append(dict(condition=condition,source=str(baseline_csv),method='hungarian',values=hungarian_values))
    body='<tr style="border-bottom:2px solid #b9bec5"><th>헝가리안<small>일대일 대응</small></th>'+''.join(f'<td>{v:.2f}</td>' for v in hungarian_values)+'</tr>'
    for (label,path),v in zip(sources,vals):
        cells=''.join('<td>'+ (f'<b>{a:.2f}</b>' if round(a,2)==best[c]
                     else f'<span class="sparse-second">{a:.2f}</span>' if round(a,2)==second[c]
                     else f'{a:.2f}')+'</td>' for c,a in enumerate(v))
        style=' style="background:#f3ebed;border-top:2px solid #bca4aa"' if 'Oracle 매핑' in label else ''
        body+=f'<tr{style}><th>{label}</th>{cells}</tr>'
    annotation_cards.append(f'<div class="result-card"><h3>{title}</h3><table class="baseline-table"><colgroup><col style="width:34%"><col span="6" style="width:11%"></colgroup><thead><tr><th rowspan="2">방법</th><th colspan="3">이미지로 텍스트 검색</th><th colspan="3">텍스트로 이미지 검색</th></tr><tr><th>R@1</th><th>R@5</th><th>R@10</th><th>R@1</th><th>R@5</th><th>R@10</th></tr></thead><tbody>{body}</tbody></table></div>')
(ASSETS/'annotation-comparison-results.json').write_text(json.dumps(annotation_provenance,ensure_ascii=False,indent=2))
slide('Oracle 매핑과 희소 대응의 검색 성능','<div class="annotation-results-grid">'+''.join(annotation_cards)+'</div><ul style="font-size:17px;line-height:1.6;margin-top:12px;padding-left:22px"><li><b>정렬을 유도하는 목적 없이 개념 주석(annotated concept)을 예측하도록 구성한 Oracle 매핑도, 두 학습 조건 모두 헝가리안보다 양방향 검색 성능이 높았음</b></li><li><b>다만 개념 의미를 직접 지도하지 않고 희소성을 제한한 Sparse CCA·Sparse PLS-SVD보다 검색 성능이 낮았음</b></li></ul><div class="bottom"><span>수치는 Recall(%) · 희소 조합은 최대 16개 activation 사용 · Sparse CCA 171개는 기존 256개 조합 중 앞 171개를 사용<br>주석은 COCO 대응 학습에만 사용 · Sinkhorn은 공통 개념 좌표 대신 원래 activation 좌표에서 비교</span></div>','annotation-results',note='주석 기반 propagated_presence_16만 사용한다. 사전 언급으로 학습한 caption_mentions 조건과 섞지 않는다. Sparse CCA 171차원은 원래 좌표 크기를 사용하는 native 결과이며 별도 unit_variance 변형과 구분한다. 두 조건 모두 평가 주석 없이 같은 COCO val2017 검색을 수행한 저장 결과다.')

from build_oracle_variants import render as render_oracle_variants
slides.append(render_oracle_variants())
slides.append(Path(__file__).with_name('takeaway.html').read_text())

# Render prose notation with the same Computer Modern math font as display equations.
_inline_cache={}
def inline_math(tex):
    if tex not in _inline_cache:
        name=f'inline-{len(_inline_cache)+1}'
        eq(name,tex)
        # Preserve the renderer's aspect ratio while sizing relative to surrounding prose.
        svg=(ASSETS/f'{name}.svg').read_text()
        w,h=map(float,re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"',svg).groups())
        _inline_cache[tex]=f'<img class="inline-math" src="assets/2026-10-06-mapping/{name}.svg" alt="{html.escape(tex,quote=True)}" style="width:{w/26:.4f}em;height:{h/26:.4f}em">'
    return _inline_cache[tex]
def render_prose_math(markup):
    markup=markup.replace('u<sub>n</sub>=x<sub>n</sub>A, v<sub>n</sub>=y<sub>n</sub>B', 'u_n=x_nA, v_n=y_nB')
    markup=markup.replace('d<sub>I</sub>·d<sub>T</sub>','d_I·d_T').replace('P<sub>ij</sub>','P_{ij}')
    pattern=re.compile(r'(?<![A-Za-z])(?:u_1|v_1|u_n=x_nA, v_n=y_nB|d_I·d_T|P_\{ij\}|A·B|X·Y|x·y|a·b|:k|Cov|Corr|tr|[ABCPQWXYNabkrnxy]|ε)(?![A-Za-z])')
    aliases={'u_n=x_nA, v_n=y_nB':r'u_n=x_nA,\;v_n=y_nB','d_I·d_T':r'd_I,\,d_T','A·B':r'A,\,B','X·Y':r'X,\,Y','x·y':r'x,\,y','a·b':r'a,\,b','Cov':r'\operatorname{Cov}','Corr':r'\operatorname{Corr}','tr':r'\operatorname{tr}','ε':r'\varepsilon'}
    parts=re.split(r'(<[^>]+>)',markup)
    skip=False
    for i,part in enumerate(parts):
        if part.startswith('<svg') or part.startswith('<aside'):skip=True
        elif part.startswith('</svg') or part.startswith('</aside'):skip=False
        elif not skip and not part.startswith('<') and not (i>0 and 'class="caption-text"' in parts[i-1]):
            parts[i]=pattern.sub(lambda m:inline_math(aliases.get(m[0],m[0])),part)
    return ''.join(parts)
slides=[render_prose_math(s) for s in slides]

css='''
.annotation-results-grid{display:grid;grid-template-columns:1fr 1fr;gap:32px}.annotation-results .baseline-table th,.annotation-results .baseline-table td{padding:3px 2px}.annotation-results .result-card h3{font-size:18px;margin-bottom:8px}.annotation-results .baseline-table small{font-size:10px}.annotation-results .bottom span{font-size:13px}.annotation-method .bottom span{font-size:14px}.retrieval-lead{font-size:20px;line-height:1.5;margin-bottom:23px}.retrieval-layout{display:grid;grid-template-columns:540px 1fr;gap:30px}.retrieval-formulas h3{font-size:20px;margin-bottom:10px}.retrieval-formulas .equation{max-width:100%;margin:8px 0 15px}.retrieval-definition{font-size:17px;line-height:1.55;color:var(--muted)}.retrieval-demo{border-left:1px solid var(--line);padding-left:26px}.retrieval-demo h3{font-size:20px;margin-bottom:12px}.retrieval-pair{display:grid;grid-template-columns:118px 1fr;gap:18px;align-items:center}.retrieval-pair>img{width:118px;height:129px;object-fit:cover;border-radius:8px}.caption-tag{font-size:13px;color:var(--muted);margin:4px 0 9px}.caption-text{font-size:20px;line-height:1.45;color:#2584a4}.retrieval-step{font-size:17px;line-height:1.5;margin:17px 0 14px}.rank-heading{display:flex;align-items:baseline;justify-content:space-between;font-size:17px;margin-bottom:9px}.rank-heading span{font-size:12px;color:var(--muted)}.rank-grid{display:grid;grid-template-columns:repeat(10,1fr);gap:5px}.rank-box{border:1px solid #d6dce3;border-radius:5px;padding:8px 0;text-align:center;background:#f0f3f6}.rank-box span{display:block;font-size:12px;margin-bottom:5px}.rank-box b{font-size:13px;color:#6a7380}.rank-box.correct{background:#f9e3e7;border:2px solid var(--accent);padding:7px 0}.rank-box.correct b{color:var(--accent)}.rank-reading{font-size:18px;line-height:1.5;margin-top:12px}.retrieval-setup .bottom{font-size:17px}.result-setup{font-size:18px;margin-bottom:20px}.baseline-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.result-card h3{font-size:19px;line-height:1.5;margin-bottom:13px}.baseline-table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums;letter-spacing:-.04em}.baseline-table th,.baseline-table td{padding:6px 2px;text-align:center;border-bottom:1px solid #dde1e6;white-space:nowrap}.baseline-table thead th{font-size:12px;color:var(--muted);padding:5px 0}.baseline-table tbody th{text-align:left;font-weight:500;font-size:14px}.baseline-table small{display:block;font-size:11px;color:var(--muted);font-weight:400;margin-top:3px}.baseline-table .soft-start>*{border-top:2px solid #b8bec8}.baseline-table .sparse-second{text-decoration:underline;text-underline-offset:3px;text-decoration-thickness:1px}.baseline-table b.sparse-best{color:#111;font-weight:800}.baseline-table b{color:var(--accent);font-weight:800}.sparse-baseline-results .baseline-table td,.sparse-baseline-results .baseline-table th{padding:1px;font-size:12px}.sparse-baseline-results .baseline-table thead th{font-size:11px}.sparse-baseline-results .baseline-table small{font-size:10px;margin-top:1px}.sparse-baseline-results .result-card h3{font-size:18px;margin-bottom:8px}.sparse-takeaways{margin:12px 0 0;padding-left:20px;font-size:15px;line-height:1.45}.sparse-takeaways li+li{margin-top:5px}.sparse-baseline-results .result-message{font-size:18px;margin-top:12px}.sparse-baseline-results .bottom{font-size:13px;padding-top:8px}.sparse-baseline-results .bottom span{font-size:12px;white-space:nowrap}.sparse-result{background:#f2edef}.result-message{margin-top:20px;font-size:21px;line-height:1.55;color:var(--accent)}.result-qualification{margin-top:8px;font-size:16px;line-height:1.5;color:var(--muted)}.result-notes{margin-top:12px}.result-notes p{font-size:15px;line-height:1.7;color:var(--muted)}.sparse-intro>div{margin-bottom:27px}.sparse-intro h3{font-size:24px;margin-bottom:10px}.sparse-intro p{font-size:22px}.sparse-question{border-top:1px solid var(--line);padding-top:24px}.sparse-case{display:grid;grid-template-columns:1fr 330px;gap:34px}.sparse-case h3{font-size:23px}.sparse-case p{font-size:21px;line-height:1.6}.sparse-case .sparse-example-note{font-size:17px;color:var(--muted)}.sparse-case-visual{border-left:1px solid var(--line);padding-left:25px}.sparse-case-visual h3{font-size:17px;margin-bottom:8px}.sparse-case-visual .pls-diagram{height:430px}.sparse-table.sparse-ref-table td{padding:11px 12px;font-size:18px}.sparse-table.sparse-ref-table th{padding:10px 12px}.sparse-check{color:#16852b;font-size:24px;font-weight:800}.sparse-cross{color:#d52222;font-size:24px;font-weight:800}.design-label{display:block;font-size:12px;font-weight:400;margin-top:5px}.sparse-reference-note{font-size:15px;line-height:1.5;margin-top:12px;color:var(--muted)}.bottom.sparse-references{display:flex;flex-direction:column;gap:4px;font-size:12px;line-height:1.5;padding-top:12px}.sparse-references>div{white-space:nowrap}.sparse-references a{color:inherit;text-decoration:underline}.sparse-table{width:100%;border-collapse:collapse;font-size:19px;line-height:1.55}.sparse-table th{text-align:left;color:var(--accent);padding:14px 12px;border-bottom:2px solid var(--accent)}.sparse-table td{padding:14px 12px;border-bottom:1px solid var(--line);vertical-align:top}.sparse-table th:first-child{width:25%}.sparse-table th:nth-child(2){width:33%}.scope-block{padding:28px 0 32px}.scope-block+.scope-block{border-top:1px solid var(--line);padding-top:32px}.scope-block h3{font-size:26px;margin-bottom:18px}.scope-block p{font-size:24px;line-height:1.65}.inline-math{display:inline-block;vertical-align:-.16em;max-width:none;object-fit:contain;margin:0 .06em;letter-spacing:0}:root{--accent:#750014;--ink:#1a1d21;--muted:#4f5660;--line:#d9dde2}*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",Arial,sans-serif;background:#eef0f2;color:var(--ink);overflow:hidden;letter-spacing:-.02em}.deck{position:fixed;inset:0;display:flex;align-items:center;justify-content:center}.stage{width:1280px;height:720px;position:relative;flex-shrink:0}.slide{position:absolute;inset:0;padding:50px 60px;background:linear-gradient(145deg,#fff,#f5f7f9);border:1px solid var(--line);border-radius:22px;box-shadow:0 18px 40px #0001;visibility:hidden;opacity:0;overflow:hidden}.slide.active{visibility:visible;opacity:1}.slide:before{content:"";position:absolute;left:60px;right:60px;top:26px;height:3px;background:linear-gradient(90deg,var(--accent),transparent 70%)}.eyebrow{color:var(--accent);font-size:13px;font-weight:600;letter-spacing:.10em;margin-bottom:12px}h2{font-size:34px;line-height:1.22;font-weight:720;letter-spacing:-.035em;margin-bottom:26px}.content{position:relative;height:530px}h3{font-size:22px;line-height:1.4;margin-bottom:12px;color:var(--accent)}p{font-size:21px;line-height:1.55;word-break:keep-all}b{font-weight:680}.goal{font-size:22px;margin-bottom:24px}.setup-columns{display:grid;grid-template-columns:320px 1fr;gap:26px}.setup-columns>div+div{border-left:1px solid var(--line);padding-left:23px}.setup-columns p{font-size:17px;line-height:1.5}.setup-columns h3{font-size:19px}.alignment-row{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:16px;align-items:center;min-height:158px}.alignment-row+.alignment-row{margin-top:2px}.alignment-formula h3{font-size:18px;margin-bottom:8px;white-space:nowrap}.alignment-formula .equation{margin:5px 0}.alignment-diagram{display:block;width:280px;height:153px;font:12px -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;letter-spacing:0;fill:#39424d}.diagram-head{font-weight:650;fill:#750014}.diagram-note{font-size:11px;fill:#727b85}.setup-columns .alignment-definition{font-size:15px;line-height:1.5;margin-top:6px}.space{margin-top:16px}.equation{display:block;object-fit:contain;object-position:left center;max-width:100%;margin:8px 0 12px;width:auto}.method-columns{display:grid;grid-template-columns:540px 1fr;gap:45px}.explain{border-left:1px solid var(--line);padding-left:30px}.explain h3:not(:first-child){margin-top:22px}.explain p{font-size:21px;line-height:1.53}.hungarian-three{display:grid;grid-template-columns:350px 440px 1fr;gap:26px}.hungarian-three .explain{padding-left:25px}.hungarian-three .explain p{font-size:20px}.hungarian-three .explain .cov-note{font-size:16px;line-height:1.45}.hungarian-three .explain h3:not(:first-child){margin-top:22px}.procrustes-visual{border-left:1px solid var(--line);padding-left:20px;display:flex;align-items:flex-start;justify-content:center}.pls-diagram{width:280px;height:450px;display:block}.procrustes-diagram{width:280px;height:425px;display:block}.hungarian-visual{border-left:1px solid var(--line);padding-left:22px;display:flex;flex-direction:column;align-items:center;justify-content:center}.hungarian-visual h3{font-size:20px;white-space:nowrap}.hungarian-visual .hungarian-diagram{width:260px;height:235px}.hungarian-visual .swap-diagram{width:280px;height:370px}.hungarian-visual p{font-size:19px;color:var(--accent);line-height:1.6}.hungarian-example{display:grid;grid-template-columns:265px 1fr;gap:18px;align-items:center}.hungarian-diagram{width:265px;height:235px}.hungarian-example p{font-size:18px;color:var(--accent);line-height:1.5;font-weight:650}.hungarian-example .equation{margin:5px 0}.definitions{font-size:18px;color:var(--muted);line-height:1.5;margin-top:12px}.bottom{position:absolute;left:0;right:0;bottom:0;border-top:1px solid var(--line);padding-top:16px;font-size:19px;line-height:1.5;color:var(--muted)}.bottom b{color:var(--accent)}.bottom span{font-size:16px}.speaker-notes{display:none}.nav{position:fixed;bottom:14px;display:flex;gap:14px;align-items:center;left:50%;transform:translateX(-50%);font-size:14px;color:var(--muted)}button{font:inherit;border:1px solid #ccd1d6;background:white;border-radius:6px;padding:5px 12px;cursor:pointer}button:disabled{opacity:.4}.counter{min-width:54px;text-align:center}.shortcuts{position:fixed;right:24px;bottom:21px;font-size:12px;color:var(--muted)}@media print{body{overflow:visible;background:white}.deck{position:static;display:block}.stage{width:auto;height:auto;transform:none!important}.slide{position:relative;visibility:visible;opacity:1;width:1280px;height:720px;page-break-after:always;border:0;box-shadow:none;border-radius:0}.nav,.shortcuts{display:none}@page{size:1280px 720px;margin:0}}
'''
css+='.training-flow{display:grid;grid-template-columns:1.1fr 26px 1.18fr 26px .93fr;align-items:center;gap:9px;margin:12px 0 20px}.training-stage{height:101px;border:1px solid var(--line);border-radius:9px;padding:11px 13px;background:#fafbfc}.training-stage h4{font-size:17px;color:var(--accent);margin-bottom:9px;white-space:nowrap}.training-stage p{font-size:15px;line-height:1.55}.sae-branches{display:flex;flex-direction:column;gap:5px;font-size:15px}.image-sae{color:#750014}.text-sae{color:#2584a4}.flow-connector{height:1px;background:#9da6b1;position:relative}.flow-connector:after{content:"";position:absolute;right:-1px;top:-4px;border-left:6px solid #9da6b1;border-top:4px solid transparent;border-bottom:4px solid transparent}.retrieval-setup .annotation-results-grid{display:grid;grid-template-columns:1fr 1fr;gap:32px}.annotation-results .baseline-table th,.annotation-results .baseline-table td{padding:3px 2px}.annotation-results .result-card h3{font-size:18px;margin-bottom:8px}.annotation-results .baseline-table small{font-size:10px}.annotation-results .bottom span{font-size:13px}.annotation-method .bottom span{font-size:14px}.retrieval-lead{font-size:17px;margin-bottom:8px}.retrieval-setup .retrieval-formulas h3{font-size:18px;margin-bottom:7px}.retrieval-setup .retrieval-formulas .equation{margin:5px 0 10px}.retrieval-setup .retrieval-definition{font-size:15px;line-height:1.5}.retrieval-setup .retrieval-demo h3{font-size:18px;margin-bottom:10px}.retrieval-setup .retrieval-pair{grid-template-columns:86px 1fr;gap:14px}.retrieval-setup .retrieval-pair>img{width:86px;height:94px}.retrieval-setup .caption-text{font-size:17px}.retrieval-setup .caption-tag{font-size:12px;margin:3px 0 6px}.retrieval-setup .retrieval-step{font-size:15px;margin:10px 0}.retrieval-setup .rank-heading{font-size:15px;margin-bottom:6px}.retrieval-setup .rank-box{padding:5px 0}.retrieval-setup .rank-box.correct{padding:4px 0}.retrieval-setup .rank-reading{font-size:16px;margin-top:9px}.retrieval-setup .retrieval-layout{padding-bottom:8px}'
js='''const slides=[...document.querySelectorAll('.slide')];let index=0;function show(n){index=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>{s.classList.toggle('active',i===index);s.inert=i!==index;s.setAttribute('aria-hidden',i!==index)});document.querySelector('.counter').textContent=`${index+1} / ${slides.length}`;document.querySelector('#prev').disabled=index===0;document.querySelector('#next').disabled=index===slides.length-1;history.replaceState(null,'',`#${index+1}`)}function resize(){document.querySelector('.stage').style.transform=`scale(${Math.min((innerWidth-40)/1280,(innerHeight-76)/720,2)})`}function hash(){show((parseInt(location.hash.slice(1))||1)-1)}document.querySelector('#prev').onclick=()=>show(index-1);document.querySelector('#next').onclick=()=>show(index+1);addEventListener('keydown',e=>{if(e.ctrlKey||e.metaKey||e.altKey)return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}if(e.key==='Home')show(0);if(e.key==='End')show(slides.length-1);if(e.key.toLowerCase()==='f'){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen()}});addEventListener('resize',resize);addEventListener('hashchange',hash);window.deck={show,slides:slides.length,get index(){return index}};resize();hash();'''
css+='.retrieval-setup .retrieval-layout{grid-template-columns:610px 1fr;gap:24px}.setup-alignment-row{display:grid;grid-template-columns:minmax(0,1fr) 240px;gap:14px;align-items:center;min-height:140px}.retrieval-setup .setup-alignment-row h3{font-size:16px;white-space:nowrap}.setup-alignment-row .alignment-diagram{width:240px;height:141px}.retrieval-setup .retrieval-definition{font-size:14px}.retrieval-setup .retrieval-definition.space{margin-top:7px}'
css+='.retrieval-setup .alignment-diagram{font-size:16px}.retrieval-setup .alignment-diagram .diagram-head{font-size:14px}.retrieval-setup .alignment-diagram .diagram-note{font-size:15px}'
page='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>2026-10-07 진행 상황 공유</title><style>'+css+'</style></head><body><main class="deck"><div class="stage">'+''.join(slides)+'</div></main><nav class="nav"><button id="prev" aria-label="이전 슬라이드">‹</button><span class="counter"></span><button id="next" aria-label="다음 슬라이드">›</button></nav><div class="shortcuts">방향키로 이동 · F 전체 화면</div><script>'+js+'</script></body></html>'
from prepend_recap import prepend_recap
page = prepend_recap(page)
page = page.replace('<div class="stage">', '<div class="stage">' + Path(__file__).with_name("title-agenda.html").read_text(), 1)
(ROOT/'2026-10-07.html').write_text(page)
print(ROOT/'2026-10-07.html')
print(f'{len(slides)} slides, {len(list(ASSETS.glob("*.svg")))} equations')
