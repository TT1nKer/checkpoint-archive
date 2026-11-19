"""Audit the full byte ledgers and build the pilot summary and decision figure."""
import json
from pathlib import Path

from experiment import ingredients, measure

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def load(name):
    return json.loads((RESULTS / name).read_text())


def summarize():
    rows = []
    for model in ('14m','70m'):
        for start in (32000,112000):
            stem=f'{model}-{start}'
            costs=load(f'full-{stem}.json')
            a,s,bridges,ends=ingredients(costs)
            plans=load(f'fullplans-{stem}.json')
            verified=load(f'verified-{stem}.json')
            replay=load(f'replay-guide-{stem}.json')
            cert=load(f'milp-{stem}.json')
            cold_audit=load(f'cold-{stem}.json')['layouts']
            for k, info in plans.items():
                single, dual = info['single'],info['dual']
                for name, candidate in info.items():
                    if not isinstance(candidate,dict):
                        continue
                    metric=measure(tuple(map(tuple,candidate['plan'])),a,s,ends if name=='endpoints_fixed' else bridges)
                    for field in ('archive_bytes','cold_bytes','max_records'):
                        assert metric[field]==candidate[field]
                    assert candidate['max_records']<=int(k)
                sb,db=single['archive_bytes'],dual['archive_bytes']
                lower=info['dual_lower_bound']
                if k=='5':
                    for name, record in verified.items():
                        assert record['sha256_verified']==17
                        for field in ('archive_bytes','max_records','cold_bytes','per_record_bytes'):
                            assert record[field]==info[name][field]
                    for kind in ('single','dual'):
                        audit=cold_audit[kind]
                        assert audit['sha256_verified']==17 and len(audit['targets'])==17
                        assert {x['i'] for x in audit['targets']}==set(range(17))
                        assert all(x['read_bytes']==info[kind]['cold_bytes'][x['i']] for x in audit['targets'])
                        assert replay[kind]['archive_bytes']==verified['guide_'+kind]['archive_bytes']
                    d=verified['dual']['directory_bytes']
                    assert sb==cert['exact_single_parent_bytes']+d
                    cert_plan=tuple(map(tuple,cert['parents']))
                    assert measure(cert_plan,a,s,bridges)['archive_bytes']==cert['feasible_cost']+d
                    lower=max(lower,cert['lower_bound']+d)
                assert lower<=db<=sb
                row=dict(model=model,start=start,end=start+16000,cap=int(k),single_bytes=sb,
                         dual_bytes=db,savings_pct=100*(sb-db)/sb,
                         lower_bound_bytes=lower,max_possible_savings_pct=100*(sb-lower)/sb,
                         mean_cold_single=single['mean_cold_bytes'],mean_cold_dual=dual['mean_cold_bytes'],
                         cold_change_pct=100*(dual['mean_cold_bytes']/single['mean_cold_bytes']-1),
                         dual_records=dual['dual_records'],all_standalone_bytes=22+8*(len(a)+1)+sum(a))
                if k=='5':
                    if model=='70m':
                        narrow=load(f'narrow-{stem}.json')
                        v=narrow['budgets']['5']
                        for kind in ('single','dual'):
                            assert narrow['physical_k5'][kind]['sha256_verified']==17
                            assert narrow['physical_k5'][kind]['archive_bytes']==v[kind]['archive_bytes']
                        ns=v['single']['archive_bytes']
                        row.update(exact_narrow_single_savings_pct=100*(1-ns/sb),
                                   exact_narrow_dual_extra_savings_pct=100*v['savings_fraction'],
                                   exact_narrow_dual_ceiling_pct=100*(1-v['dual_lower_bound']/ns))
                    row.update(guide_replay_savings_pct=100*(1-replay['dual']['archive_bytes']/replay['single']['archive_bytes']),
                               full_payloads_verified=verified['dual']['sha256_verified'],
                               fixed_dag_midpoint_savings_bytes=info['endpoints_fixed']['archive_bytes']-db,
                               dual_mode_blocks=verified['dual']['dual_mode_blocks'],
                               secondary_milp_feasible_bytes=cert['feasible_cost']+d)
                rows.append(row)
    main=[r for r in rows if r['cap']==5]
    successes=[r for r in main if r['savings_pct']>=3 and r['mean_cold_dual']<=r['mean_cold_single']]
    result=dict(primary_cap=5,primary_threshold_pct=3,primary_cohorts=len(main),
                primary_successes=len(successes),
                cohorts_with_ceiling_below_3pct=sum(r['max_possible_savings_pct']<3 for r in main),
                decision='continue' if len(successes)>=3 and any(r['model']=='14m' for r in successes) else 'stop',
                rows=rows)
    (RESULTS/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return main


def draw(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax=plt.subplots(figsize=(8.1,3.8),layout='constrained')
    y=[3,2,1,0]
    for r,h in zip(rows,y):
        a,b=r['savings_pct'],r['max_possible_savings_pct']
        ax.plot([a,b],[h,h],color='#337a8d',linewidth=2,zorder=2)
        ax.scatter(b,h,facecolors='white',edgecolors='#337a8d',s=62,zorder=3)
        ax.scatter(a,h,color='#003d54',s=25,zorder=4)
    ax.axvline(3,color='#a3482d',ls='--',lw=1.4)
    ax.set_yticks(y, ['14m · all tensors · 32–48k','14m · all tensors · 112–128k',
                      '70m · slices · 32–48k','70m · slices · 112–128k'])
    right = max(3,max(r['max_possible_savings_pct'] for r in rows))
    ax.set_xlim(-.08,right+.20)
    ax.set_ylim(-.60,3.55)
    ax.set_xlabel('Storage saved vs. the optimal single-reference archive (%)')
    ax.set_title('Cold restore limited to five records',loc='left',fontweight='bold',fontsize=14)
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',alpha=.18)
    ax.tick_params(axis='y',length=0)
    ax.legend(handles=[Line2D([],[],ls='',marker='o',color='#003d54',label='Built archive'),
                       Line2D([],[],ls='',marker='o',markerfacecolor='white',color='#337a8d',label='Savings ceiling'),
                       Line2D([],[],ls='--',color='#a3482d',label='3% decision threshold')],
              loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=9)
    fig.savefig(ROOT/'savings.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    draw(summarize())
