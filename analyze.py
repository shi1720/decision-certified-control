"""Audit retained results and regenerate every manuscript statistic and plot."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results');F=Path('paper/figures');F.mkdir(exist_ok=True)
plants=['thermal4','mechanical4','nonnormal6','dense8']; labels=['Thermal (4,2)','Mechanical (4,2)','Nonnormal (6,2)','Dense (8,3)']
methods=['full','DCC_1e-5','DCC_1e-3','random','logdet','rolling','prior']
frames=[]; audit={'training_histories':0,'transitions':0,'max_dynamics_residual':0.,'max_prior_violation':0.,'max_noise_violation':0.,'failures':[]}
for name in plants:
 p=R/name
 failures=json.loads((p/'failures.json').read_text());audit['failures']+=failures
 d=pd.read_csv(p/'queries.csv');assert len(d)==16128 and (d.status=='ok').all();frames.append(d)
 arrays=np.load(p/'query_arrays.npz');assert arrays['queries'].shape==(2304,11)
 n={'thermal4':4,'mechanical4':4,'nonnormal6':6,'dense8':8}[name];m=3 if n==8 else 2
 actions=arrays['actions'][:,:,:m];v=arrays['queries'][:,8:8+m]
 obj=.5*np.sum((actions-v[:,None,:])**2,axis=2)
 np.testing.assert_allclose(obj.ravel(),d.objective,atol=1e-9)
 assert np.max(abs(actions))<=.8+1e-7
 archives=list((p/'archives').glob('*.npz'));assert len(archives)==24
 for ap in archives:
  a=np.load(ap);res=a['Y']-a['Z']@a['theta'].T
  audit['max_dynamics_residual']=max(audit['max_dynamics_residual'],float(np.max(abs(res-a['W']))))
  audit['max_prior_violation']=max(audit['max_prior_violation'],float(np.max(abs(a['theta']-a['center'])-a['radius'])))
  audit['max_noise_violation']=max(audit['max_noise_violation'],float(np.max(abs(res)-a['eta'])))
  np.testing.assert_allclose(a['Z'][1:,:n],a['Y'][:-1],atol=1e-14)
  audit['training_histories']+=1;audit['transitions']+=len(a['Z'])
d=pd.concat(frames,ignore_index=True);d.to_csv(R/'all_queries.csv',index=False)
c=d[d.method.str.startswith('DCC')].copy();eps=c.method.map({'DCC_1e-5':1e-5,'DCC_1e-3':1e-3}).values
assert np.max(c.certificate_gap.values-eps)<1.1e-9
assert np.max(c.certificate_lower-c.full_objective)<1e-7
assert np.max(c.full_objective-c.certificate_upper)<1e-7
assert np.max(c.action_error-np.sqrt(2*c.certificate_gap))<2e-5
for name,n,m in zip(plants,[4,4,6,8],[2,2,2,3]):
 s=c[c.plant==name];assert s.scalar_constraints.max()<=2*n*(n+m);assert s.model_witnesses.max()<=m
assert d.max_true_next.max()<=1+1e-7 and d.max_robust_next.max()<=1+1e-7
cl=pd.read_csv(R/'closed_loop.csv');assert len(cl)==168 and (cl.status=='ok').all() and (cl.steps==96).all()
assert cl.max_state.max()<=1+1e-7
traj=np.load(R/'closed_loop_trajectories.npz',allow_pickle=False)
trajectory_residual=0.
for key in traj.files:
 if not key.endswith('_states'):continue
 prefix=key[:-7];states=traj[key];inputs=traj[prefix+'_inputs'];noise=traj[prefix+'_noise'];theta=traj[prefix+'_theta']
 pred=np.c_[states[:-1],inputs]@theta.T+noise
 trajectory_residual=max(trajectory_residual,float(np.max(abs(pred-states[1:]))))
 assert np.max(abs(states))<=1+1e-7 and np.max(abs(inputs))<=.8+1e-7
 assert np.max(abs(noise))<=.015+1e-12
assert trajectory_residual<1e-12

s=pd.read_csv(R/'streaming.csv');assert len(s)==720
assert (s.lower-s.full).max()<1e-7 and (s.full-s.upper).max()<1e-7
assert s.portable_error.max()<2e-5 and s.portable_gap.max()<1e-4+1e-7
sc=pd.read_csv(R/'scaling.csv');assert len(sc)==768 and sc.gap.max()<1e-4+1e-7
ab=pd.read_csv(R/'ablation.csv');assert len(ab)==768
# Bootstrap whole independent training histories, preserving all common queries
# and the paired method comparisons inside a history.
rng=np.random.default_rng(82001)
summary=[]
for name in plants:
 for method in methods:
  z=d[(d.plant==name)&(d.method==method)];clusters=z.groupby('seed').objective_gap.mean().values
  boots=np.mean(rng.choice(clusters,(10000,len(clusters)),replace=True),axis=1)
  summary.append(dict(plant=name,method=method,gap_mean=z.objective_gap.mean(),gap_ci_low=np.quantile(boots,.025),gap_ci_high=np.quantile(boots,.975),gap_max=z.objective_gap.max(),action_max=z.action_error.max(),constraints_mean=z.scalar_constraints.mean(),constraints_max=z.scalar_constraints.max(),transitions_mean=z.raw_transitions.mean(),models_max=z.model_witnesses.max(),local_hit_rate=(z.archive_calls==0).mean(),archive_lp_mean=z.archive_calls.mean(),total_ms_mean=1000*z.elapsed.mean(),local_ms_mean=1000*(z.elapsed-z.archive_seconds).mean(),archive_ms_mean=1000*z.archive_seconds.mean()))
su=pd.DataFrame(summary);su.to_csv(R/'summary.csv',index=False)
audit['max_closed_loop_dynamics_residual']=trajectory_residual
audit.update(query_instances=len(d)//7,method_queries=len(d),closed_loop_episodes=len(cl),closed_loop_steps=int(cl.steps.sum()),streaming_queries=len(s),scaling_queries=len(sc),ablation_method_queries=len(ab),certificate_max_gap=float(c.certificate_gap.max()),certificate_max_lower_error=float((c.certificate_lower-c.full_objective).max()),certificate_max_action_bound_error=float((c.action_error-np.sqrt(2*c.certificate_gap)).max()),max_true_next=float(d.max_true_next.max()),max_robust_next=float(d.max_robust_next.max()),max_closed_loop_state=float(cl.max_state.max()),portable_max_action_error=float(s.portable_error.max()),invalidated_witnesses=int(s.invalidated.sum()),portable_ratio_at_1024=float((s[s.N==1024].payload_json_bytes/s[s.N==1024].raw_float64_bytes).median()),ablation_inner_violations=int((ab.inner_only_violation>1e-7).sum()))
(R/'audit.json').write_text(json.dumps(audit,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
colors=['#133c55','#008c95','#e49335','#9d5c91','#7b8d35','#776a5c','#8a8f98']
fig,axes=plt.subplots(1,4,figsize=(10,2.5),sharey=True)
for ax,name,label in zip(axes,plants,labels):
 z=su[(su.plant==name)&su.method.isin(methods[1:])].set_index('method').loc[methods[1:]]
 ax.bar(np.arange(6),np.maximum(z.gap_mean,1e-10),color=colors[1:]);ax.set_yscale('log');ax.set_ylim(1e-9,1);ax.set_title(label);ax.set_xticks(range(6),['D5','D3','Rnd','LD','Win','Prior'],rotation=45)
 lo=np.maximum(z.gap_ci_low,1e-10);hi=np.maximum(z.gap_ci_high,1e-10)
 ax.errorbar(range(6),np.maximum(z.gap_mean,1e-10),yerr=[np.maximum(0,z.gap_mean-lo),np.maximum(0,hi-z.gap_mean)],fmt='none',ecolor='black',capsize=2)
axes[0].set_ylabel('Mean excess intervention cost');fig.tight_layout();fig.savefig(F/'performance.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,2.6))
z=c[c.method=='DCC_1e-3'];axes[0].scatter(1000*z.certificate_gap,1000*z.objective_gap,s=3,alpha=.25,color=colors[1]);axes[0].plot([0,1],[0,1],'k--',lw=1);axes[0].set(xlabel=r'Certified gap ($10^{-3}$)',ylabel=r'Actual gap ($10^{-3}$)')
axes[0].set_xticks([0,.5,1]);axes[0].set_yticks([0,.5,1])
axes[1].scatter(np.sqrt(2*z.certificate_gap),z.action_error,s=3,alpha=.25,color=colors[2]);axes[1].plot([0,.045],[0,.045],'k--',lw=1);axes[1].set(xlabel=r'Action bound $\sqrt{2g}$',ylabel='Action error');axes[1].set_xticks([0,.02,.04])
for n,g in sc.groupby('n'):
 stats=g.groupby('N').scalar_constraints.agg(['mean','max']);axes[2].plot(stats.index,stats['mean'],'o-',label=f'n={n}');axes[2].plot(stats.index,stats['max'],':',alpha=.6)
axes[2].set(xscale='log',xlabel='Archive transitions',ylabel='Retained data inequalities');axes[2].legend(frameon=False,ncol=2,fontsize=9);fig.tight_layout();fig.savefig(F/'certificates_scaling.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,2.6))
for method in ['reuse','no_compaction','no_reuse']:
 g=ab[ab.method==method].groupby('query');axes[0].plot(g.scalar_constraints.mean(),label=method.replace('_',' '));axes[1].plot(g.archive_calls.mean(),label=method.replace('_',' '))
axes[0].set(xlabel='Query',ylabel='Mean retained inequalities');axes[1].set(xlabel='Query',ylabel='Archive support LPs / query');axes[0].legend(frameon=False,fontsize=9)
for name,label in zip(plants,labels):
 g=s[s.plant==name].groupby('N');ratio=g.apply(lambda q:(q.payload_json_bytes/q.raw_float64_bytes).median(),include_groups=False);axes[2].plot(ratio.index,ratio,'o-',label=label)
axes[2].set(xscale='log',yscale='log',xlabel='Archive transitions',ylabel='JSON certificate / raw float64 bytes');axes[2].axhline(1,color='k',ls='--',lw=1);axes[2].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(F/'ablation_streaming.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(9.8,2.25));ax.axis('off')
boxes=[(.02,.22,.26,.62,'Archive: all transitions\nFull model polytope $P_N$\nSupport LP + sparse dual'),(.37,.22,.28,.62,'Controller certificate\nSelected data + models\n$P_C\supseteq P_N\supseteq V$'),(.75,.22,.23,.62,'Accept if $g\leq\epsilon$\nSafe action $u_C$\nOtherwise refine')]
from matplotlib.patches import FancyBboxPatch
for x,y,w,h,t in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc='#eaf2f5',ec='#133c55',transform=ax.transAxes));ax.text(x+w/2,y+h/2,t,ha='center',va='center',transform=ax.transAxes,fontsize=12)
for x1,x2 in [(.29,.36),(.66,.74)]:ax.annotate('',xy=(x2,.53),xytext=(x1,.53),arrowprops={'arrowstyle':'->','color':'#008c95'},xycoords='axes fraction')
ax.text(.50,.04,'Compact both certificates after acceptance; screen model witnesses when new data arrive.',ha='center',transform=ax.transAxes,fontsize=9)
fig.savefig(F/'architecture.pdf');plt.close(fig)
# TeX data table generated from the complete audited CSV, not hand transcribed.
lines=[]
for name,label in zip(plants,labels):
 for method in methods[1:]:
  q=su[(su.plant==name)&(su.method==method)].iloc[0]
  display={'DCC_1e-5':'DCC ($10^{-5}$)','DCC_1e-3':'DCC ($10^{-3}$)','random':'Random','logdet':'Log-det','rolling':'Window','prior':'Prior'}[method]
  gap=f"{q.gap_mean:.2e}";ci=f"[{q.gap_ci_low:.1e},{q.gap_ci_high:.1e}]"
  lines.append(f"{label if method==methods[1] else ''} & {display} & {gap} & {ci} & {q.constraints_mean:.1f} & {q.total_ms_mean:.1f} \\\\")
  if method==methods[-1] and name!=plants[-1]:lines.append('\\midrule')
Path('paper/results_table.tex').write_text('\n'.join(lines)+'\n')
pd.DataFrame([dict(noise_factor=f,exposed_queries=int(((c[c.method=='DCC_1e-3'].max_true_next+(f-1)*.015)>1+1e-7).sum()),queries=9216) for f in [1,2,5]]).to_csv(R/'noise_envelope_stress.csv',index=False)
print(json.dumps(audit,indent=2));print(su.to_string(index=False));print('\nABLATION\n',ab.groupby('method')[['archive_calls','scalar_constraints','gap','upper_only_gap','elapsed']].mean().to_string());print('\nSCALING\n',sc.groupby(['n','N'])[['scalar_constraints','full_time','cache_time','archive_calls']].mean().to_string());print('\nREGIMES\n',d.groupby(['regime','method'])[['objective_gap','elapsed','scalar_constraints']].mean().to_string());print('\nCLOSED LOOP\n',cl.groupby('method')[['max_state','intervention','elapsed']].mean().to_string())
