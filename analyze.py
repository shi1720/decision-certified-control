import sys,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).parent/'src'))
from safe_dwell import rotation
from run_experiments import plants,METHODS
ROOT=Path(__file__).parent;FIG=ROOT/'paper/figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
colors=['#777777','#e69f00','#56b4e9','#0072b2','#cc79a7','#009e73']
labels=['Longest','Fixed-6','One-step','Minimax','Gaussian proxy','RCD']
rows=list(csv.DictReader(open(ROOT/'results/runs.csv')))
summary=list(csv.DictReader(open(ROOT/'results/summary.csv')))
traces=json.loads((ROOT/'results/traces.json').read_text())
# Mechanism and stable damping limit.
fig,axs=plt.subplots(1,2,figsize=(7.0,2.65),layout='constrained')
for a,co in zip([.95,.98,.995],['#e69f00','#0072b2','#009e73']):
 p=rotation(a,2*np.pi/24,eta=.001)
 ds=[p.resolution(.25,h,1) for h in range(1,25)]
 axs[0].plot(range(1,25),ds,'o-',ms=3,color=co,label=f'a={a}')
axs[0].axhline(.08,color='black',ls=':',lw=1,label='resolution target')
axs[0].set(xlabel='Hold duration h',ylabel='Worst-case diameter',yscale='log',xticks=[1,6,12,18,24]);axs[0].legend(fontsize=8)
for H,co in zip([8,12,24],['#e69f00','#0072b2','#009e73']):
 a=np.linspace(.8,.9999,400);ratio=np.abs(1-a*np.exp(2j*np.pi/H))/(1-a)
 axs[1].plot(1-a,ratio,color=co,label=f'Period {H}')
axs[1].set(xlabel='Damping gap 1-a',ylabel='One-step / full-period resolution score',xscale='log',yscale='log');axs[1].invert_xaxis();axs[1].legend(fontsize=8)
fig.savefig(FIG/'mechanism.pdf');fig.savefig(FIG/'mechanism.png',dpi=220);plt.close(fig)
# Frontier conditioned on state, no empirical policy fitting.
p=rotation(.995,2*np.pi/24,eta=.001);x=np.array([.15,-.08]);I=p.safe_intervals(x,[.5,1.5])
U=np.max(np.abs(I),axis=1);ds=np.array([p.resolution(u,h,1) for h,u in enumerate(U,1)])
fig,ax=plt.subplots(figsize=(6.8,2.35),layout='constrained')
ax.plot(range(1,25),ds,'o-',color='#0072b2',ms=4)
ax.axhline(.08,color='black',ls=':',label='target 0.08')
chosen=p.choose(x,[.5,1.5],'resolution_constrained')[1]
ax.scatter([chosen],[ds[chosen-1]],s=90,facecolor='none',edgecolor='#009e73',lw=2,zorder=3,label=f'RCD: h={chosen}')
ax.scatter([24],[ds[23]],marker='x',s=65,color='#d55e00',zorder=3,label='longest: h=24')
ax.set(xlabel='Safe hold duration',ylabel='Guaranteed posterior diameter',yscale='log',xticks=[1,6,12,18,24]);ax.legend(fontsize=8,loc='upper left')
fig.savefig(FIG/'frontier.pdf');fig.savefig(FIG/'frontier.png',dpi=220);plt.close(fig)
# Endpoint-only temporal trace, plot as step after each observation.
fig,axs=plt.subplots(1,2,figsize=(7,2.65),layout='constrained')
for k in [0,1,3,5]:
 tr=np.array(traces['rotation24/random/'+METHODS[k]])
 time=np.r_[0,tr[:,0]];width=np.r_[1,tr[:,2]-tr[:,1]]
 axs[0].step(time,width,where='post',color=colors[k],label=labels[k])
 axs[1].plot(tr[:,0],tr[:,5],'.-',color=colors[k],ms=3,label=labels[k])
axs[0].set(xlabel='Physical time step',ylabel='Gain interval diameter',yscale='log');axs[0].axhline(.08,color='black',ls=':',lw=1);axs[0].legend(fontsize=8)
axs[1].set(xlabel='Physical time step',ylabel='Endpoint state norm',ylim=(0,1.04));axs[1].axhline(1,color='black',ls=':',lw=1)
fig.savefig(FIG/'trace.pdf');fig.savefig(FIG/'trace.png',dpi=220);plt.close(fig)
# Cross-plant comparison, all controlled rules.
fig,axs=plt.subplots(1,4,figsize=(7.2,2.45),layout='constrained',sharey=True)
for ax,(name,title) in zip(axs,zip(plants(),['Rotation-8','Rotation-24','No rotation','Mixed 3D'])):
 for k,method in enumerate(METHODS):
  r=next(r for r in summary if (r['plant'],r['regime'],r['method'])==(name,'random',method))
  ax.scatter(float(r['observations_mean']),float(r['diameter_mean']),color=colors[k],s=35,marker=['o','s','^','D','v','P'][k],label=labels[k])
 ax.set(title=title,xscale='log',yscale='log',xlabel='Observations',xticks=[10,40,240]);ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter());ax.axhline(.08,color='black',ls=':',lw=.8)
axs[0].set_ylabel('Mean final interval diameter');handles,ls=axs[0].get_legend_handles_labels();fig.legend(handles,ls,ncol=6,loc='outside lower center',fontsize=8)
fig.savefig(FIG/'comparison.pdf');fig.savefig(FIG/'comparison.png',dpi=220);plt.close(fig)
# Exact sweep evidence over damping, period, detuning, h.
sweep=[]
for a in [.9,.95,.98,.995,.999,.9999]:
 for H in [4,8,12,24]:
  for detune in [-.1,-.05,0.,.05,.1]:
   p=rotation(a,2*np.pi/H*(1+detune),eta=.001)
   for h in range(1,25):
    sweep.append([a,H,detune,h,np.linalg.norm(p.q[h]),p.rho[h],p.resolution(.25,h,1)])
with open(ROOT/'results/sweep.csv','w') as f:
 wr=csv.writer(f);wr.writerow(['damping','period','relative_detuning','duration','sensitivity_norm','noise_radius','diameter']);wr.writerows(sweep)
# Paired, simultaneous 98.75% marginal bootstrap CIs for four prespecified contrasts.
rng=np.random.default_rng(987);contrasts=[]
for name in plants():
 vals={method:np.array([float(r['diameter']) for r in rows if (r['plant'],r['regime'],r['method'])==(name,'random',method)]) for method in ['longest','resolution_constrained']}
 diff=vals['resolution_constrained']-vals['longest'];boot=diff[rng.integers(0,50,(20000,50))].mean(1)
 lo,hi=np.quantile(boot,[.00625,.99375]);contrasts.append(dict(plant=name,mean_difference=float(diff.mean()),lower=float(lo),upper=float(hi)))
(ROOT/'results/contrasts.json').write_text(json.dumps(contrasts,indent=2))
# Generated tables avoid manual transcription.
for regime in ['random','adversarial']:
 lines=[r'\begin{tabular}{llrrr}',r'\toprule',r'Plant & Rule & Diameter & Observations & Time resolved \\',r'\midrule']
 for name in plants():
  for k,method in enumerate(METHODS):
   r=next(r for r in summary if (r['plant'],r['regime'],r['method'])==(name,regime,method))
   d=float(r['diameter_mean']);sd=float(r['diameter_sd']);obs=float(r['observations_mean']);res=float(r['resolved_fraction_mean'])
   lines.append(f'{name} & {labels[k]} & ${d:.5f}\\pm{sd:.5f}$ & {obs:.1f} & {res:.3f} '+r'\\')
  if name!='mixed3d':lines.append(r'\midrule')
 lines.extend([r'\bottomrule',r'\end{tabular}'])
 (ROOT/f'paper/table_{regime}.tex').write_text('\n'.join(lines))
print('contrasts',contrasts,'sweep cases',len(sweep))
