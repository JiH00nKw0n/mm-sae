"""Write a self-contained TeX version; no external figure/font files required."""
from pathlib import Path
import draw as source

ROOT = Path(__file__).resolve().parent
lines = [r'''\documentclass{article}
\usepackage[paperwidth=46cm,paperheight=39.5cm,margin=7mm]{geometry}
\usepackage{kotex}
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\definecolor{ink}{HTML}{183348}
\definecolor{muted}{HTML}{516475}
\definecolor{line}{HTML}{D2DDE5}
\definecolor{surface}{HTML}{F9FBFD}
\begin{document}
\begin{tikzpicture}[x=1cm,y=1cm,>=Stealth]
\path[use as bounding box] (0,0) rectangle (44,37.7);
''']

def node(x, y, text, size=17, anchor='center', color='ink', bold=False):
    weight = r'\bfseries' if bold else ''
    lines.append(fr'\node[anchor={anchor},text={color},font=\fontsize{{{size}}}{{{size+4}}}\selectfont{weight}] at ({x:.3f},{y:.3f}) {{{text}}};')

node(.1,37,'이미지·텍스트 대응행렬을 구하는 네 가지 방법',29,'west',bold=True)
node(.1,35.8,'공통 입력은 같은 이미지·캡션 쌍의 SAE activation입니다. 학습 평균을 빼고 표준편차로 나눕니다.',17,'west','muted')
node(.1,34.7,r'$X\in\mathbb{R}^{N\times d_I}$ 이미지\qquad $Y\in\mathbb{R}^{N\times d_T}$ 텍스트\qquad $C=\dfrac{X^\top Y}{N}$ coactivation correlation',18,'west')
node(.1,33.45,r'$S_I=\dfrac{X^\top X}{N},\quad S_T=\dfrac{Y^\top Y}{N}$ 모달리티 내부의 공분산\qquad $G_I=S_I+\lambda I,\quad G_T=S_T+\lambda I$',17,'west','muted')

W,H = 21.5,14.25
def panel(x,y,title,subtitle,color):
    lines.append(fr'\draw[rounded corners=8pt,fill=surface,draw=line] ({x},{y}) rectangle ({x+W},{y+H});')
    lines.append(fr'\definecolor{{accent}}{{HTML}}{{{color}}}')
    lines.append(fr'\fill[accent,rounded corners=4pt] ({x+.5},{y+H-1.9}) rectangle ({x+1.9},{y+H-.5});')
    node(x+1.2,y+H-1.2,title[:2],21)
    node(x+2.4,y+H-.85,title[3:],25,'west',bold=True)
    node(x+2.4,y+H-1.8,subtitle,17,'west','muted')
    return lambda px,py:(x+W*px,y+H*py)

def text(at, px, py, text, size=18, color='ink'):
    node(*at(px,py),text,size,color=color)

def matrix(at, values, px, py, w, h, label, positive=False):
    left,bottom=at(px,py); width=W*w; height=H*h
    vmax=max(abs(values).max(),1e-10)
    nr,nc=values.shape
    for i in range(nr):
        for j in range(nc):
            rgb=(source.POS(values[i,j]/vmax) if positive else source.SIGNED((values[i,j]/vmax+1)/2))[:3]
            hexcolor=''.join(f'{round(v*255):02X}' for v in rgb)
            lines.append(fr'\definecolor{{cell}}{{HTML}}{{{hexcolor}}}')
            x=left+j*width/nc; y=bottom+(nr-i-1)*height/nr
            lines.append(fr'\filldraw[fill=cell,draw=white,line width=.6pt] ({x:.4f},{y:.4f}) rectangle ({x+width/nc:.4f},{y+height/nr:.4f});')
    node(left+width/2,bottom+height+.55,label,21)

def arrow(at, label):
    a=at(.34,.345); b=at(.61,.345)
    lines.append(fr'\draw[->,draw=muted,line width=1.2pt] ({a[0]},{a[1]}) -- ({b[0]},{b[1]});')
    text(at,.475,.405,label,14,'muted')

at=panel(0,17.6,'01 Procrustes','전체 방향을 유지하며 회전·반사합니다.','FFD6A5')
text(at,.5,.735,r'$\displaystyle\max_Q\;\langle\bar C,Q\rangle\quad\text{s.t.}\quad Q^\top Q=I_d$',25)
text(at,.5,.631,r'$\displaystyle\Longleftrightarrow\;\min_Q\;\|\bar XQ-\bar Y\|_F^2$',22)
matrix(at,source.c,.09,.235,.22,.255,r'$\bar C$');arrow(at,'SVD')
matrix(at,source.q,.67,.235,.22,.255,r'$Q=UV^\top$')
text(at,.5,.158,'직교행렬 하나를 구합니다. 음수 계수도 허용합니다.',17)
text(at,.5,.090,r'부족한 차원을 0으로 채워 $d=\max(d_I,d_T)$, $\bar C=\bar X^\top\bar Y/N$으로 둡니다.',14,'muted')
text(at,.5,.037,r'검색은 $\cos(\bar xQ,\bar y)$로 계산합니다.',17)

at=panel(22.3,17.6,'02 PLS-SVD (Cross-SVD)','함께 변하는 상위 $r$개 방향만 남깁니다.','CAFFBF')
text(at,.5,.735,r'$\displaystyle\max_{A,B}\;\operatorname{tr}(A^\top CB)$',27)
text(at,.5,.631,r'$\displaystyle A^\top A=I_r,\qquad B^\top B=I_r$',24)
matrix(at,source.c,.08,.235,.22,.255,r'$C$');arrow(at,'그대로 SVD')
matrix(at,source.a,.655,.235,.10,.255,r'$A=U_r$')
matrix(at,source.b,.835,.235,.10,.255,r'$B=V_r$')
text(at,.5,.158,'계수 방향끼리 직교하도록 조합행렬 두 개를 구합니다.',17)
text(at,.5,.090,r'$C=U\Sigma V^\top$에서 상위 $r$개 열을 사용합니다. $\Sigma$는 출력에 곱하지 않습니다.',14,'muted')
text(at,.5,.037,r'검색은 $\cos(xA,yB)$로 계산합니다.',17)

at=panel(0,2.65,'03 CCA','내부의 분산·상관까지 보정해 조합합니다.','9BF6FF')
text(at,.5,.735,r'$\displaystyle\max_{A,B}\;\operatorname{tr}(A^\top CB)$',27)
text(at,.5,.631,r'$\displaystyle A^\top G_I A=I_r,\qquad B^\top G_T B=I_r$',24)
matrix(at,source.c,.08,.235,.22,.255,r'$C$');arrow(at,'상관 보정 후 SVD')
matrix(at,source.ca,.655,.235,.10,.255,r'$A$');matrix(at,source.cb,.835,.235,.10,.255,r'$B$')
text(at,.5,.158,'계수가 아니라 공분산을 기준으로 제약을 둡니다.',17)
text(at,.5,.090,r'$\widetilde C=G_I^{-1/2}CG_T^{-1/2}$의 SVD 후 $A=G_I^{-1/2}U_r$, $B=G_T^{-1/2}V_r$',14,'muted')
text(at,.5,.037,r'검색은 $\cos(xA,yB)$로 계산합니다. 현재 $\lambda=0.01$입니다.',17)

at=panel(22.3,2.65,'04 Sinkhorn','양수 연결을 분배하고 행·열합을 맞춥니다.','FFADAD')
text(at,.5,.735,r'$\displaystyle\max_{P\geq0}\;\langle C,P\rangle+\varepsilon H(P)$',26)
text(at,.5,.631,r'$\displaystyle P\mathbf{1}=a,\qquad P^\top\mathbf{1}=b$',25)
matrix(at,source.c,.08,.235,.22,.255,r'$C$');arrow(at,'지수화·행열 보정')
matrix(at,source.p,.66,.235,.22,.255,r'$P$',True)
for j in range(6):
    a=at(.89,.235+.255*(j+.5)/6);b=at(.927,.235+.255*(j+.5)/6)
    lines.append(fr'\draw[green!50!black,line width=2pt] ({a[0]},{a[1]})--({b[0]},{b[1]});')
    a=at(.66+.22*(j+.5)/6,.211);b=at(.66+.22*(j+.5)/6,.225)
    lines.append(fr'\draw[green!50!black,line width=2pt] ({a[0]},{a[1]})--({b[0]},{b[1]});')
text(at,.5,.158,r'$P=\operatorname{diag}(u)\exp(C/\varepsilon)\operatorname{diag}(v)$',23)
text(at,.5,.090,r'$u,v$를 반복 조정해 $a_i=1/d_I$, $b_j=1/d_T$인 행·열합을 맞춥니다.',15)
text(at,.5,.037,r'$H(P)=-\sum_{ij}P_{ij}\log P_{ij}$이며, $\varepsilon$이 클수록 연결이 고르게 분산됩니다.',14,'muted')

node(.1,1.85,r'$A\in\mathbb{R}^{d_I\times r}$, $B\in\mathbb{R}^{d_T\times r}$, $P\in\mathbb{R}^{d_I\times d_T}$\quad $N$은 학습 쌍 수, $r$은 공통 좌표 수입니다. 실험에서는 $r=256$을 사용했습니다.',16,'west')
node(.1,1.10,'행렬 색은 설명용 합성 예시입니다. 파랑은 음수, 주황은 양수, 초록은 양수 연결이며 각 행렬 안에서 색을 정규화했습니다.',14,'west','muted')
node(.1,.35,r'$\langle C,P\rangle=\sum_{ij}C_{ij}P_{ij}$\quad tr은 대각 원소의 합, $I$는 단위행렬, $\mathbf{1}$은 모든 원소가 1인 벡터입니다. exp는 원소별로 적용합니다.',14,'west','muted')
lines.append(r'\end{tikzpicture}\end{document}')
(ROOT/'four-mapping-methods.tex').write_text('\n'.join(lines))
