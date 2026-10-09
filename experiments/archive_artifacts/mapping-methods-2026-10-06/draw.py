"""An explanatory, not empirical, diagram of the four implemented objectives."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False,
                     'mathtext.fontset': 'cm', 'svg.fonttype': 'path', 'pdf.fonttype': 42})
INK, MUTED, LINE = '#183348', '#516475', '#d2dde5'
SIGNED = LinearSegmentedColormap.from_list('signed', ['#4083b5', '#f7fafc', '#e79455'])
POS = LinearSegmentedColormap.from_list('mass', ['#f5faf7', '#219179'])
rng = np.random.default_rng(11)
h = rng.normal(size=(1500, 3))
x = np.einsum('ij,jk->ik', h, rng.normal(size=(3, 6))) + rng.normal(size=(1500, 6)) * .8
y = np.einsum('ij,jk->ik', h, rng.normal(size=(3, 6))) + rng.normal(size=(1500, 6)) * .8
x = (x-x.mean(0))/x.std(0)
y = (y-y.mean(0))/y.std(0)
c = np.einsum('ni,nj->ij', x, y)/len(x)
si, st = np.einsum('ni,nj->ij', x, x)/len(x), np.einsum('ni,nj->ij', y, y)/len(y)
u, _, vt = np.linalg.svd(c)
q, a, b = u@vt, u[:, :2], vt.T[:, :2]
def invroot(s):
    vals, vec = np.linalg.eigh(s + .01*np.eye(6))
    return (vec / np.sqrt(vals)) @ vec.T
gi, gt = invroot(si), invroot(st)
cu, _, cvt = np.linalg.svd(gi@c@gt)
ca, cb = gi@cu[:, :2], gt@cvt.T[:, :2]
kernel = c/.15
lv = np.zeros(6)
for _ in range(1000):
    lu = -np.log(6)-logsumexp(kernel+lv[None, :], axis=1)
    lv = -np.log(6)-logsumexp(kernel+lu[:, None], axis=0)
p = np.exp(kernel+lu[:, None]+lv[None, :])
assert np.allclose(q.T@q, np.eye(6))
assert np.allclose(ca.T@(si+.01*np.eye(6))@ca, np.eye(2))
assert np.allclose(p.sum(0), 1/6) and np.allclose(p.sum(1), 1/6)

fig = plt.figure(figsize=(18, 15.5), facecolor='white')
fig.text(.04, .962, '이미지·텍스트 대응행렬을 구하는 네 가지 방법', fontsize=29, weight='bold', color=INK)
fig.text(.04, .925, '공통 입력은 같은 이미지·캡션 쌍의 SAE activation입니다. 학습 평균을 빼고 표준편차로 나눕니다.', fontsize=16, color=MUTED)
fig.text(.04, .893, r'$X\in\mathbb{R}^{N\times d_I}$  이미지     '+r'$Y\in\mathbb{R}^{N\times d_T}$  텍스트     '
         +r'$C=X^\top Y/N$  coactivation correlation', fontsize=17, color=INK)
fig.text(.04, .861, r'$S_I=X^\top X/N,\quad S_T=Y^\top Y/N$'
         +'   모달리티 내부의 공분산     '+r'$G_I=S_I+\lambda I,\quad G_T=S_T+\lambda I$', fontsize=16, color=MUTED)

def panel(left, bottom, num, title, subtitle, accent):
    ax = fig.add_axes([left, bottom, .447, .355]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0,0),1,1, boxstyle='round,pad=0,rounding_size=.022',
                              facecolor='#f9fbfd', edgecolor=LINE, linewidth=1))
    ax.add_patch(FancyBboxPatch((.025,.845),.071,.108,boxstyle='round,pad=0,rounding_size=.014',
                              facecolor=accent,edgecolor='none'))
    ax.text(.0605,.896,num,ha='center',va='center',fontsize=19,color=INK,weight='bold')
    ax.text(.117,.917,title,fontsize=23,color=INK,weight='bold',va='center')
    ax.text(.117,.855,subtitle,fontsize=15,color=MUTED,va='center')
    return ax

def formula(ax, text, y, size=20):
    ax.text(.5,y,text,fontsize=size,ha='center',va='center',color=INK)

def matrix(ax, values, left, bottom, width, height, label, positive=False):
    # Each matrix has its own color scale; color indicates sign and relative magnitude.
    extent = (left, left+width, bottom, bottom+height)
    vmax = max(abs(values).max(), 1e-10)
    ax.imshow(values, extent=extent, origin='upper', aspect='auto', interpolation='nearest',
              cmap=POS if positive else SIGNED, vmin=0 if positive else -vmax, vmax=vmax, zorder=2)
    nr,nc = values.shape
    for z in range(nc+1): ax.plot([left+width*z/nc]*2,[bottom,bottom+height],color='white',lw=1,zorder=3)
    for z in range(nr+1): ax.plot([left,left+width],[bottom+height*z/nr]*2,color='white',lw=1,zorder=3)
    ax.text(left+width/2,bottom+height+.025,label,ha='center',va='bottom',fontsize=17,color=INK)

def arrow(ax, x1, x2, y, text):
    ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle='-|>',mutation_scale=16,color=MUTED,lw=1.7))
    ax.text((x1+x2)/2,y+.047,text,ha='center',va='bottom',fontsize=12.5,color=MUTED)

ax = panel(.04,.47,'01','Procrustes','전체 방향을 유지하며 회전·반사합니다.','#FFD6A5')
formula(ax,r'$\max_Q\;\langle\bar C,Q\rangle\quad\mathrm{s.t.}\quad Q^\top Q=I_d$',.732,21)
formula(ax,r'$\Leftrightarrow\;\min_Q\;\|\bar XQ-\bar Y\|_F^2$',.633,19)
matrix(ax,c,.09,.235,.22,.26,r'$\bar C$')
arrow(ax,.34,.63,.34,'SVD')
matrix(ax,q,.67,.235,.22,.26,r'$Q=UV^\top$')
ax.text(.5,.157,'직교행렬 하나를 구합니다. 음수 계수도 허용합니다.',ha='center',fontsize=15,color=INK,weight='bold')
ax.text(.5,.092,'부족한 차원을 0으로 채워 '+r'$d=\max(d_I,d_T)$'+', '+r'$\bar C=\bar X^\top\bar Y/N$'+'으로 둡니다.',ha='center',fontsize=12.5,color=MUTED)
ax.text(.5,.037,r'검색은 $\cos(\bar xQ,\bar y)$'+'로 계산합니다.',ha='center',fontsize=14,color=INK)

ax = panel(.514,.47,'02','PLS-SVD  (Cross-SVD)','함께 변하는 상위 r개 방향만 남깁니다.','#CAFFBF')
formula(ax,r'$\max_{A,B}\;\mathrm{tr}(A^\top CB)$',.732,23)
formula(ax,r'$A^\top A=I_r,\qquad B^\top B=I_r$',.631,21)
matrix(ax,c,.08,.235,.22,.26,r'$C$')
arrow(ax,.33,.61,.34,'그대로 SVD')
matrix(ax,a,.655,.235,.10,.26,r'$A=U_r$')
matrix(ax,b,.835,.235,.10,.26,r'$B=V_r$')
ax.text(.5,.154,'계수 방향끼리 직교하도록 조합행렬 두 개를 구합니다.',ha='center',fontsize=15,color=INK,weight='bold')
ax.text(.5,.090,r'$C=U\Sigma V^\top$'+'에서 상위 '+r'$r$'+'개 열을 사용합니다. '+r'$\Sigma$'+'는 출력에 곱하지 않습니다.',ha='center',fontsize=12.5,color=MUTED)
ax.text(.5,.037,r'검색은 $\cos(xA,yB)$'+'로 계산합니다.',ha='center',fontsize=14,color=INK)

ax = panel(.04,.095,'03','CCA','내부의 분산·상관까지 보정해 조합합니다.','#9BF6FF')
formula(ax,r'$\max_{A,B}\;\mathrm{tr}(A^\top CB)$',.732,23)
formula(ax,r'$A^\top G_I A=I_r,\qquad B^\top G_T B=I_r$',.631,20)
matrix(ax,c,.08,.235,.22,.26,r'$C$')
arrow(ax,.33,.61,.34,'상관 보정 후 SVD')
matrix(ax,ca,.655,.235,.10,.26,r'$A$')
matrix(ax,cb,.835,.235,.10,.26,r'$B$')
ax.text(.5,.154,'계수가 아니라 공분산을 기준으로 제약을 둡니다.',ha='center',fontsize=15,color=INK,weight='bold')
ax.text(.5,.090,r'$\widetilde C=G_I^{-1/2}CG_T^{-1/2}$'+r' 의 SVD 후 $A=G_I^{-1/2}U_r,\ B=G_T^{-1/2}V_r$',ha='center',fontsize=12.5,color=MUTED)
ax.text(.5,.037,r'검색은 $\cos(xA,yB)$'+'로 계산합니다. 현재 '+r'$\lambda=0.01$'+'입니다.',ha='center',fontsize=14,color=INK)

ax = panel(.514,.095,'04','Sinkhorn','양수 연결을 분배하고 행·열합을 맞춥니다.','#FFADAD')
formula(ax,r'$\max_{P\geq 0}\;\langle C,P\rangle+\varepsilon H(P)$',.732,22)
formula(ax,r'$P\mathbf{1}=a,\qquad P^\top\mathbf{1}=b$',.631,21)
matrix(ax,c,.08,.235,.22,.26,r'$C$')
arrow(ax,.33,.61,.34,'지수화·행열 보정')
matrix(ax,p,.66,.235,.22,.26,r'$P$',positive=True)
for j in range(6):
    ax.plot([.89,.927],[.235+.26*(j+.5)/6]*2,color='#219179',lw=3)
for j in range(6):
    ax.plot([.66+.22*(j+.5)/6]*2,[.211,.224],color='#219179',lw=3)
ax.text(.5,.160,r'$P=\mathrm{diag}(u)\,\exp(C/\varepsilon)\,\mathrm{diag}(v)$',ha='center',fontsize=18,color=INK)
ax.text(.5,.091,r'$u,v$'+'를 반복 조정해 '+r'$a_i=1/d_I,\ b_j=1/d_T$'+'인 행·열합을 맞춥니다.',ha='center',fontsize=13,color=INK)
ax.text(.5,.037,r'$H(P)=-\sum_{ij}P_{ij}\log P_{ij}$'+'이며, '+r'$\varepsilon$'+'이 클수록 연결이 고르게 분산됩니다.',ha='center',fontsize=12.5,color=MUTED)

fig.text(.04,.060, r'$A\in\mathbb{R}^{d_I\times r},\ B\in\mathbb{R}^{d_T\times r},\ P\in\mathbb{R}^{d_I\times d_T}$'
         +'     '+r'$N$'+'은 학습 쌍 수, '+r'$r$'+'은 공통 좌표 수입니다. 실험에서는 '+r'$r=256$'+'을 사용했습니다.',fontsize=13.5,color=INK)
fig.text(.04,.033,'행렬 색은 설명용 합성 예시입니다. 파랑은 음수, 주황은 양수, 초록은 양수 연결이며 각 행렬 안에서 색을 정규화했습니다.',fontsize=12.5,color=MUTED)
fig.text(.04,.012,r'$\langle C,P\rangle=\sum_{ij}C_{ij}P_{ij}$'+'     tr은 대각 원소의 합, '+r'$I$'+'는 단위행렬, '+r'$\mathbf{1}$'+'은 모든 원소가 1인 벡터입니다. exp는 원소별로 적용합니다.',fontsize=12,color=MUTED)
fig.savefig(ROOT/'four-mapping-methods.png',dpi=170,facecolor='white')
fig.savefig(ROOT/'four-mapping-methods.svg',facecolor='white')
fig.savefig(ROOT/'four-mapping-methods.pdf',facecolor='white')
plt.close(fig)
