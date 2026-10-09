import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
out=Path(__file__).parent
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family':'Apple SD Gothic Neo','axes.unicode_minus':False,'mathtext.fontset':'cm','svg.fonttype':'path','pdf.fonttype':42})
HEAD=5.4
W,H=18,26.3+HEAD
fig=plt.figure(figsize=(W,H),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H-HEAD,-HEAD);ax.axis('off')
ink='#183348';muted='#4d6270'
def t(x,y,s,size=15,bold=False,color=ink):
    return ax.text(x,y,s,fontsize=size,color=color,weight='bold' if bold else 'normal',va='top',linespacing=1.5)
def m(x,y,s,size=23):return t(x,y,s,size)
t(.65,-5.10,'대응 행렬 W를 학습하는 두 가지 목표',28,True)
t(.65,-4.46,'1. 개념과 activation의 다대다 대응을 표현',19,True)
t(.95,-4.04,'여러 activation이 하나의 개념을 나타내거나, 하나의 activation이 여러 개념에 관여하는 관계를 표현합니다.',16)
t(.65,-3.49,'2. 이미지·텍스트 activation을 대응시켜 유사도 기반으로 정렬',19,True)
t(.95,-3.07,'대응 행렬 W를 이용해 비교했을 때, 정답 이미지·텍스트 쌍의 유사도가 다른 쌍보다 높아지도록 합니다.',16)
m(.95,-2.53,r'$s_W(x,y)=\cos(xW,y)=\frac{xWy^{\mathsf{T}}}{\|xW\|_2\,\|y\|_2}$',23)
t(9.8,-2.39,'한쪽 activation을 W로 변환하는 경우:',15)
m(.95,-1.64,r'$W=AB^{\mathsf{T}},\qquad s_{A,B}(x,y)=\cos(xA,yB)=\frac{xWy^{\mathsf{T}}}{\|xA\|_2\,\|yB\|_2}$',23)
t(12.45,-1.48,'양쪽을 변환하는 경우:',14)
t(.95,-.87,r'$x\in\mathbb{R}^{1\times d_I},\ y\in\mathbb{R}^{1\times d_T}$는 표준화한 이미지·텍스트 activation 벡터입니다. W는 이미지 좌표와 텍스트 좌표를 연결합니다.',14)
t(.65,-.40,'아래 방법은 정렬을 위한 후보입니다. 학습된 activation 조합이 어떤 개념을 나타내는지는 별도로 검증해야 합니다.',14,color=muted)
ax.plot([.65,17.35],[-.03,-.03],color='#ccd6dc',lw=1)
t(.65,.35,'네 가지 대응 방법의 목적식과 제약',30,True)
t(.65,.94,'표준화한 activation에서 무엇을 최적화하고, 어떤 변환을 허용하는가',18,color=muted)
t(.65,1.50,'입력과 표준화',18,True)
m(.65,1.93,r'$X_0\in\mathbb{R}^{N\times d_I},\quad Y_0\in\mathbb{R}^{N\times d_T}$',22)
t(7.0,1.97,'원래 이미지·텍스트 activation 행렬. 같은 행은 같은 이미지·캡션 쌍.',14)
t(.65,2.42,r'$N$은 쌍의 수, $d_I,d_T$는 이미지·텍스트의 activation 좌표 수. 각 열은 SAE activation 좌표 하나.',15)
t(.65,2.94,'좌표별 표준화',17,True)
t(.65,3.36,'이미지·텍스트 각각에서, 각 activation 좌표의 전체 N개 표본 평균을 뺀 뒤 같은 좌표의 표준편차로 나눕니다.',16)
t(.65,3.80,'표준편차는 N으로 나누어 계산합니다. 표준편차가 0인 좌표는 제외합니다.',14)
m(.65,4.25,r'$X\in\mathbb{R}^{N\times d_I},\quad Y\in\mathbb{R}^{N\times d_T},\qquad C=\frac{X^{\mathsf{T}}Y}{N}\in\mathbb{R}^{d_I\times d_T}$',23)
t(.65,4.82,r'$C$는 이미지 activation 좌표와 텍스트 activation 좌표 사이의 coactivation correlation 행렬입니다.',16,True)
t(.65,5.20,'X·Y가 위와 같이 표준화되어 있으므로 C의 각 원소가 Pearson 상관계수입니다.',14)
rows=[(5.80,'01  Procrustes','#fff3e6'),(10.90,'02  PLS-SVD · Partial Least Squares SVD','#eff8ed'),(16.00,'03  CCA · Canonical Correlation Analysis','#edf8fc'),(21.10,'04  Sinkhorn · 엔트로피 정규화 최적 수송','#fff0f0')]
for top,title,col in rows:
    ax.add_patch(FancyBboxPatch((.35,top),17.3,4.90,boxstyle='round,pad=.02,rounding_size=.15',facecolor=col,edgecolor='none'))
    t(.65,top+.20,title,22,True)
    ax.plot([9,9],[top+.85,top+4.47],color='#ccd6dc',lw=1)
# Procrustes
z=rows[0][0]
m(.75,z+.90,r'$\min_Q\;\|XQ-Y\|_F^2$',27)
m(.75,z+1.65,r'$Q^{\mathsf{T}}Q=I$',24)
m(.75,z+2.25,r'$Q\in\mathbb{R}^{d\times d},\quad d=\max(d_I,d_T)$',21)
t(.75,z+2.82,'Q는 최적화로 구하는 activation 변환 가중치 행렬입니다.',15)
t(.75,z+3.23,'X·Y의 열 수가 다르면 0으로 채워 d개로 맞춥니다.',15)
t(9.35,z+.95,'목적: 변환한 이미지 activation과 대응하는 텍스트 activation\n사이의 L2 거리의 제곱을 모든 쌍에 대해 더한 값을 최소화',17,True)
t(9.35,z+2.75,'제약: 길이와 각도를 보존하는 회전·반사만 허용',18,True)
# PLS
z=rows[1][0]
m(.75,z+.87,r'$\max_{A,B}\;\operatorname{tr}(A^{\mathsf{T}}CB)$',25)
m(.75,z+1.43,r'$=\max_{A,B}\;\sum_{k=1}^{r}\mathrm{Cov}((XA)_{:k},(YB)_{:k})$',22)
m(.75,z+2.42,r'$A^{\mathsf{T}}A=I,\quad B^{\mathsf{T}}B=I$',23)
m(.75,z+3.00,r'$A\in\mathbb{R}^{d_I\times r},\quad B\in\mathbb{R}^{d_T\times r}$',21)
t(.75,z+3.48,'A·B는 최적화로 구하는 조합 가중치 행렬입니다. 각 열이 조합 하나입니다.',14)
t(.75,z+3.85,r'$XA,YB\in\mathbb{R}^{N\times r}$는 조합한 activation. $r$은 만들 조합 수입니다.',14)
t(9.35,z+.95,'목적: A와 B를 학습해서, 서로 대응하는 r개 조합의\n공분산을 더한 값이 최대가 되도록 하는 것',17,True)
t(9.35,z+2.65,'제약: 가중치 열의 길이는 1, 서로 다른 열의 내적은 0',17,True)
# CCA
z=rows[2][0]
m(.75,z+.87,r'$\max_{A,B}\;\operatorname{tr}(A^{\mathsf{T}}CB)$',25)
m(.75,z+1.43,r'$=\max_{A,B}\;\sum_{k=1}^{r}\mathrm{Corr}((XA)_{:k},(YB)_{:k})$',22)
m(.75,z+2.42,r'$\frac{(XA)^{\mathsf{T}}(XA)}{N}=I,\quad\frac{(YB)^{\mathsf{T}}(YB)}{N}=I$',22)
m(.75,z+3.10,r'$A\in\mathbb{R}^{d_I\times r},\quad B\in\mathbb{R}^{d_T\times r}$',21)
t(.75,z+3.55,'A·B는 최적화로 구하는 조합 가중치 행렬입니다. 각 열이 조합 하나입니다.',14)
t(.75,z+3.93,r'Corr는 상관계수. $(XA)_{:k},(YB)_{:k}$는 전체 표본의 k번째 조합 값입니다.',14)
t(.75,z+4.27,'이해를 위해 안정화 항을 제외한 기본 CCA 식을 표시했습니다.',12,color=muted)
t(9.35,z+.95,'목적: A와 B를 학습해서, 서로 대응하는 r개 조합의\n상관계수를 더한 값이 최대가 되도록 하는 것',17,True)
t(9.35,z+2.65,'제약: 출력 분산은 1, 같은 모달리티의 조합 간 상관은 0',17,True)
# Sinkhorn
z=rows[3][0]
m(.75,z+.87,r'$\max_{P\geq 0}\;\left[\sum_{i,j}C_{ij}P_{ij}+\varepsilon H(P)\right]$',23)
m(.75,z+1.69,r'$=\max_{P\geq 0}\;\left[\frac{1}{N}\sum_{n,i,j}X_{ni}Y_{nj}P_{ij}-\varepsilon\sum_{i,j}P_{ij}\log P_{ij}\right]$',19)
m(.75,z+2.42,r'$P\mathbf{1}=a,\quad P^{\mathsf{T}}\mathbf{1}=b$',20)
m(.75,z+3.07,r'$P\in\mathbb{R}^{d_I\times d_T},\quad a\in\mathbb{R}^{d_I},\quad b\in\mathbb{R}^{d_T}$',21)
t(.75,z+3.52,'P는 최적화로 구하는 연결 가중치 행렬입니다. i·j는 양쪽 좌표 번호입니다.',14)
t(.75,z+3.88,'a·b는 미리 고정한 각 행·열의 총가중치입니다. 학습하는 값이 아닙니다.',14)
t(.75,z+4.24,'ε는 연결 가중치의 집중을 완화하는 강도입니다.',14)
t(9.35,z+.95,'목적: coactivation correlation이 높은 쌍에 큰 가중치 배정',17,True)
t(9.35,z+2.34,'제약: 음의 연결 금지, 각 행·열의 총가중치 고정',17,True)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for text in ax.texts:
    box=text.get_window_extent(renderer)
    if box.x1>W*fig.dpi-15:
        raise ValueError('Text exceeds page: '+text.get_text())
for ext in ['png','pdf','svg']:
    fig.savefig(out/f'four-mapping-methods-explained.{ext}',dpi=160,facecolor='white')
