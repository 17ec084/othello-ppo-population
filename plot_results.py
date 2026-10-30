"""Plot the archived fixed-opponent evaluations, without changing the experiment."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

import argparse
import gzip
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,default=None)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
if args.run:
    RUN=args.run.resolve()
    history=[json.loads(x) for x in (RUN/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
    state=json.loads((RUN/'progress.json').read_text(encoding='utf-8'))
    metrics=[json.loads(x) for x in (RUN/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
else:
    with gzip.open(ROOT/'data/evaluations.jsonl.gz','rt',encoding='utf-8') as f:history=[json.loads(x) for x in f]
    state=json.loads((ROOT/'data/reference-results.json').read_text(encoding='utf-8'))
    with gzip.open(ROOT/'data/learner-history.jsonl.gz','rt',encoding='utf-8') as f:metrics=[json.loads(x) for x in f]
steps=np.array([x['step'] for x in history]);END=int(steps[-1]);COUNT=len(history[-1]['individuals'])
available_fonts={font.name for font in font_manager.fontManager.ttflist}
font_family=next((name for name in ['Meiryo','Noto Sans CJK JP','Yu Gothic','DejaVu Sans'] if name in available_fonts),'sans-serif')
plt.rcParams.update({'font.family':font_family,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
COLORS={'random':'#137743','position':'#2066B5'}
LABELS={'random':'ランダム器','position':'位置評価器'}

def plot(ax,index=None,zoom=None,shade=False,names=('random','position')):
    if shade:
        start=None
        for row in metrics+[{'step':steps[-1]+1,'learner':-1}]:
            active=row['learner']==index
            if active and start is None:start=row['step']-1
            if not active and start is not None:
                ax.axvspan(start,row['step']-1,color='#79B77C',alpha=.20,lw=0)
                start=None
    for name in names:
        values=np.array([x['aggregate'][name]['win_rate'] if index is None else x['individuals'][index][name]['win_rate'] for x in history])*100
        ax.plot(steps,values,color=COLORS[name],lw=1.7,ls='-' if name=='random' else '--',label=LABELS[name])
        measured=np.array([index is None or x['individuals'][index]['evaluated_step']==x['step'] for x in history])
        if index is None:measured=steps%25==0
        ax.scatter(steps[measured],values[measured],color=COLORS[name],s=4,alpha=.6,zorder=3)
    ax.set(xlim=zoom or (0,max(1,END)),ylim=(0,100),xlabel='集団全体の step（1 step = 2,432局）',ylabel='勝率')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.grid(alpha=.18)
    ax.legend(loc='lower right',framealpha=.95)

fig,axes=plt.subplots(2,1,figsize=(12,8.7),gridspec_kw={'height_ratios':[1.6,1]},layout='constrained')
plot(axes[0]);axes[0].set_title(f'固定の相手への勝率：全個体の平均（{COUNT}体）',loc='left',fontweight='bold')
if not args.run:axes[0].annotate('最終：93.1% / 80.9%',xy=(4085,80.8984),xytext=(2700,55),arrowprops={'arrowstyle':'->','color':'#555'})
plot(axes[1],zoom=(0,max(1,min(700,END))),names=('position',));axes[1].set_ylim(0,40 if not args.run else 100)
if not args.run:axes[1].axvspan(170,370,color='#E7A146',alpha=.16)
axes[1].set_title('序盤の拡大：170〜370 step付近で低下し、その後回復' if not args.run else '位置評価器への勝率：序盤の拡大',loc='left')
if not args.run:axes[1].annotate('361 step：14.9%',xy=(361,14.921875),xytext=(410,6),arrowprops={'arrowstyle':'->','color':'#555'})
fig.savefig(OUT/'figure-average.png',dpi=160,facecolor='white');plt.close(fig)

for index in [i for i in [0,12,11] if i<COUNT]:
    fig,ax=plt.subplots(figsize=(12,5.5),layout='constrained')
    plot(ax,index,shade=True)
    ax.set_title(f'固定の相手への勝率：AI {index}',loc='left',fontweight='bold')
    fig.supxlabel('薄緑の背景：この個体の担当区間（55%超で更新を省略した回も含む）／点：実測値',fontsize=10)
    if index==11 and not args.run:
        ax.annotate('最後の重み更新：1105 step\n以降は固定されたまま',xy=(2000,57.03125),xytext=(2100,28),arrowprops={'arrowstyle':'->','color':'#555'})
    if index==12 and not args.run:
        ax.annotate('対ランダム：94.5% → 84.4%\n同期間の対位置評価：60.9% → 74.2%',xy=(1800,84.375),xytext=(1900,35),fontsize=10,arrowprops={'arrowstyle':'->','color':'#555'})
    fig.savefig(OUT/f'figure-ai{index:02d}.png',dpi=160,facecolor='white');plt.close(fig)

print('Created figures from',len(history),'evaluation snapshots in',OUT)
