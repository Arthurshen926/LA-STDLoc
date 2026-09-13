"""Auditable paper tables from completed full-test artifacts, never selected best seeds."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common.hashing import sha256_file

POOL=Path('/mnt/pool/sqy')
V24=POOL/'lafgs_v24_cambridge_high_capacity_online_20260901'
PAPER=POOL/'anygsloc_paper_experiments_20260902'
NEW=POOL/'anygsloc_three_gpu_20260910'


def numbers(rows):
    t=np.array([r['translation_error_cm'] for r in rows]);r=np.array([r['rotation_error_deg'] for r in rows])
    ms=np.array([r['total_ms'] for r in rows])
    if not np.isfinite(t).all() or not np.isfinite(r).all():raise ValueError('nonfinite pose errors; explicit failure policy required')
    result={'n':len(rows),'te':float(np.median(t)),'re':float(np.median(r)),'mean_te':float(t.mean()),'p90_te':float(np.percentile(t,90)),
       'cat100':int((t>=100).sum()),'ms':float(ms.mean()),'p50_ms':float(np.median(ms)),'p90_ms':float(np.percentile(ms,90)),
       'hz':float(1000/ms.mean()),'deadline100':float(100*np.mean(ms<=100)),
       'warm10_ms':float(ms[10:].mean()) if len(ms)>10 else None,
       'second_solve_rate':float(100*np.mean([x.get('sparse_feedback_candidate_ransac_iterations',0)>0 for x in rows]))}
    for a,b in [(1,1),(2,2),(5,5),(10,5),(15,5),(50,5),(500,10)]:result[f'R{a}_{b}']=float(100*np.mean((t<a)&(r<b)))
    for key in ['frontend_ms','matching_ms','ransac_ms','feedback_geometry_ms','feedback_ransac_ms']:
        result[key]=float(np.mean([x.get(key,0) for x in rows]))
    return result


def test_names(dataset):
    path=dataset/'dataset_test.txt'
    if path.exists():return {line.split()[0] for line in path.read_text().splitlines() if line.strip() and Path(line.split()[0]).suffix.lower() in ['.png','.jpg','.jpeg','.webp']}
    return set((dataset/'sparse/0/list_test.txt').read_text().splitlines())


def primary_path(family,scene):
    if family=='Cambridge':return V24/scene/'evaluation/baseline_seed2026/summary.json'
    if scene in ['office','pumpkin','redkitchen','stairs']:return NEW/f'runs/primary_{scene}/evaluation/base_seed2026/summary.json'
    if scene=='office2_5a':return POOL/'anygsloc_primary_resume_20260909/12Scenes/office2_5a/evaluation/base_seed2026/summary.json'
    if scene=='office2_5b':return POOL/'anygsloc_execution_queue_20260909/runs/primary_office2_5b/evaluation/base_seed2026/summary.json'
    return PAPER/f'{family}/{scene}/evaluation/base_seed2026/summary.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads(Path('configs/paper_baselines_20260910.json').read_text())
    inventory=[];map_hashes={};all_rows={};tables=[]
    def read(path,family,scene,condition,seed):
        path=Path(path);s=json.loads(path.read_text());rp=path.parent/'results.json';rows=json.loads(rp.read_text())
        dataset=POOL/f'Cambridge_stdloc/{scene}' if family=='Cambridge' else POOL/f'datasets/{family}_pgt_full_reference_v5/{scene}'
        expected=test_names(dataset);names=[r['image_name'] for r in rows]
        if len(names)!=len(expected) or set(names)!=expected:raise ValueError(f'test split mismatch: {path}')
        m=numbers(rows)
        if not np.allclose([m['te'],m['re']],[s['median_te_cm'],s['median_ae_deg']],atol=1e-7):raise ValueError('summary mismatch')
        mp=Path(s['input_map_path']) if 'input_map_path' in s else None
        if mp is not None:
            if str(mp) not in map_hashes:map_hashes[str(mp)]=sha256_file(mp)
            if map_hashes[str(mp)]!=s['input_map_sha256']:raise ValueError(f'map mismatch: {mp}')
        report_path=mp.parent/'report.json' if mp else None
        report=json.loads(report_path.read_text()) if report_path and report_path.exists() else {}
        m.update(family=family,scene=scene,condition=condition,seed=seed,summary=str(path),
          summary_sha256=sha256_file(path),results_sha256=sha256_file(rp),map=str(mp),
          map_sha256=map_hashes.get(str(mp)),artifact_mib=mp.stat().st_size/2**20 if mp else None,
          anchors=report.get('counts',{}).get('total_anchors'),mapping_views=report.get('counts',{}).get('mapping_views'),
          feedback_used=report.get('contracts',{}).get('feedback_used'),profile_mode_values=sorted(set(x.get('profile_mode',-1) for x in rows)))
        inventory.append(m);all_rows[str(path)]=rows
        print('checked',family,scene,condition,seed,m['te'],flush=True)
        return m
    def table(title,header,rows):
        tables.append((title,header,rows))
    def pair(t,r):return f'{t:.3f}/{r:.3f}'
    def f(x):return f'{x:.2f}'
    primary={}
    for family,scenes in baseline['scene_order'].items():
        primary[family]=[read(primary_path(family,scene),family,scene,'Base',2026) for scene in scenes]
        rows=[]
        for index,m in enumerate(primary[family]):
            rows.append([m['scene'],str(m['n']),pair(*baseline['STDLoc'][family][index]),pair(*baseline['ULF-Loc'][family][index]),pair(m['te'],m['re'])])
        rows.append(['Macro mean','—',pair(*np.mean(baseline['STDLoc'][family],0)),pair(*np.mean(baseline['ULF-Loc'][family],0)),pair(np.mean([m['te'] for m in primary[family]]),np.mean([m['re'] for m in primary[family]]))])
        table(f'{family}: median TE cm / RE deg (seed 2026)', ['Scene','N','STDLoc paper','ULF-Loc paper','AnyGSLoc-Base'],rows)
        recalls=baseline['recall_thresholds_cm_deg'][family]
        table(f'{family}: scene-macro recall percent', ['Method']+[f'R@{a}cm/{b}deg' for a,b in recalls],
              [[name]+[f(v) for v in baseline[name]['recall'][family]] for name in ['STDLoc','ULF-Loc']]+[['AnyGSLoc-Base']+[f(np.mean([m[f'R{a}_{b}'] for m in primary[family]])) for a,b in recalls]])
    efficiency=[]
    aggregate=[]
    for family,ms in primary.items():
        for m in ms:efficiency.append([family,m['scene'],f(m['ms']),f(m['p50_ms']),f(m['p90_ms']),f(m['hz']),f(m['deadline100']),f(m['frontend_ms']),f(m['matching_ms']),f(m['ransac_ms'])])
        pooled=numbers(sum([all_rows[m['summary']] for m in ms],[]));aggregate.append(dict(family=family,**pooled,macro_te=float(np.mean([m['te'] for m in ms])),macro_re=float(np.mean([m['re'] for m in ms]))))
    table('Base latency: recorded localization-call timing (ms), excluding image I/O', ['Dataset','Scene','Mean','P50','P90','1000/mean Hz','<=100ms %','Frontend','Matching','PnP'],efficiency)
    table('Dataset aggregation: macro medians versus query-pooled performance', ['Dataset','N','Macro TE/RE','Pooled TE/RE','Mean ms','P50 ms','P90 ms','Hz','>=1m count'],
          [[g['family'],str(g['n']),pair(g['macro_te'],g['macro_re']),pair(g['te'],g['re']),f(g['ms']),f(g['p50_ms']),f(g['p90_ms']),f(g['hz']),str(g['cat100'])] for g in aggregate])
    matrix=json.loads((POOL/'anygsloc_invalid_primitive_recovery_20260910/cambridge_matrix.json').read_text())
    if matrix['complete_cells']!=30:raise ValueError('prior matrix incomplete')
    priors={}
    for cell in matrix['rows']:
        scene=next(s for s in baseline['scene_order']['Cambridge'] if cell['condition'].startswith(s+'_'));method=cell['condition'][len(scene)+1:]
        priors[(scene,method)]=[read(v['summary'],'Cambridge',scene,method,int(seed)) for seed,v in sorted(cell['seeds'].items())]
    methods=['vanilla_2dgs','vanilla_3dgs','long_lrm_raw','long_lrm_1k','i3dgs_raw','i3dgs_1k']
    table('Cambridge prior comparison: fixed method, seed 2026, median TE/RE', ['Scene','MAtCHA']+methods,
      [[scene,pair(primary['Cambridge'][i]['te'],primary['Cambridge'][i]['re'])]+[pair(priors[(scene,m)][0]['te'],priors[(scene,m)][0]['re']) for m in methods] for i,scene in enumerate(baseline['scene_order']['Cambridge'])]+[
      ['Macro mean',pair(np.mean([m['te'] for m in primary['Cambridge']]),np.mean([m['re'] for m in primary['Cambridge']]))]+[pair(np.mean([priors[(s,m)][0]['te'] for s in baseline['scene_order']['Cambridge']]),np.mean([priors[(s,m)][0]['re'] for s in baseline['scene_order']['Cambridge']])) for m in methods]])
    table('Prior robustness and speed: macro recall; query-weighted time, seed 2026', ['Prior','R50/5 %','R15/5 %','R10/5 %','Mean ms','P90 ms','Hz','>=1m count'],
      [[method]+[f(np.mean([priors[(s,method)][0][k] for s in baseline['scene_order']['Cambridge']])) for k in ['R50_5','R15_5','R10_5']]+
       [f(numbers(sum([all_rows[priors[(s,method)][0]['summary']] for s in baseline['scene_order']['Cambridge']],[]))[k]) for k in ['ms','p90_ms','hz','cat100']] for method in methods])
    supplemental=[]
    root=POOL/'anygsloc_long_lrm_two_epoch_gpu2_control_20260910'
    if not json.loads((root/'gpu2/state.json').read_text()).get('all_complete'):raise ValueError('supplemental queue incomplete')
    for scene in ['StMarysChurch','GreatCourt']:
        for seed in [2026,2027]:
            m=read(root/f'runs/{scene}_long_lrm_two_epochs/evaluation/base_seed{seed}/summary.json','Cambridge',scene,'long_lrm_two_epochs',seed)
            supplemental.append([scene,str(seed),pair(m['te'],m['re']),f(m['mean_te']),f(m['p90_te']),str(m['cat100']),f(m['R50_5']),f(m['ms']),f(m['p90_ms'])])
    table('Supplemental Long-LRM budget control: 2974 / 3062 steps, not +1k', ['Scene','Seed','Median TE/RE','Mean TE cm','P90 TE cm','>=1m count','R50/5 %','Mean ms','P90 ms'],supplemental)
    online=[]
    for scene in baseline['scene_order']['Cambridge']:
        path=V24/scene/'evaluation/refinement_v31_adaptive_reserve35_seed2026/summary.json'
        if path.exists():m=read(path,'Cambridge',scene,'historical_V31_optional_test_calibrated',2026)
        else:m=next(m for m in primary['Cambridge'] if m['scene']==scene)
        base=next(x for x in primary['Cambridge'] if x['scene']==scene)
        online.append([scene,pair(base['te'],base['re']),pair(m['te'],m['re']),str(base['cat100']),str(m['cat100']),f(m['ms']),f(m['second_solve_rate'])])
    table('Historical optional V31: not merged into Base; scene/test-calibrated policy', ['Scene','Base TE/RE','Optional TE/RE','Base >=1m','Optional >=1m','Recorded mean ms','Second solve %'],online)
    # Inventory every available Base seed, without changing the common seed-2026 main tables.
    for family,ms in primary.items():
        for m in ms:
            for seed in [2027,2028]:
                p=Path(m['summary'].replace('seed2026',f'seed{seed}'))
                if p.exists():read(p,family,m['scene'],'Base',seed)
    table('Base map artifact resources (not runtime-only resident memory)', ['Dataset','Scene','Anchors','Mapping views','Artifact MiB'],
          [[m['family'],m['scene'],str(m['anchors']),str(m['mapping_views']),f(m['artifact_mib']) if m['artifact_mib'] else 'NR'] for ms in primary.values() for m in ms])
    md=['# AnyGSLoc 论文实验表格汇总','',
        '所有主表使用固定 seed 2026；其他可用 seed 全部收入 CSV/JSON，不挑选最好 seed。TE 单位 cm，RE 单位 degree。Macro 为场景等权；pooled 为查询合并，二者不得混用。',
        '', '## 协议与证据边界','',
        '- 正式方法为 Base：不启用离线反馈、在线渲染或稠密 refinement。Cambridge 为历史高容量 MAtCHA F0；室内 19 场景为 Vanilla 2DGS，不是 24 场景均使用 MAtCHA。',
        '- Cambridge prior 矩阵完成 30/30 条件、60 次完整 test；Long-LRM 两轮追加实验完成 4 次 test。24 场景 Base 均有 seed 2026，但并非全部完成最初协议所列三 seed。',
        '- 每一行已重新验证完整 test 名单、summary 中位数、结果文件哈希、可用的输入地图哈希。没有重新估计位姿，也没有新调参。',
        '- 室内先验采用发布的 SfM pseudo-GT 坐标系；dataset manifest 指出初始化点云可能包含 test-view reconstruction evidence。不得宣称严格 train-only 几何初始化，且需在论文协议中披露。',
        '- MAtCHA 优化图像数与 Vanilla full mapping 不同；Cambridge prior 对比是可用性/敏感性实验，不是同训练预算重建质量排名。Long-LRM 32 输入，i3DGS 输入和优化预算另有区别。',
        '- STDLoc 的 12Scenes 行及扩展 recall 来自 ULF-Loc 论文所列对照，并非 STDLoc 原文给出的 12Scenes 表格。',
        '- 本地运行 RTX 3090；两篇论文为 RTX 4090。不能作严格同硬件加速比。',
        '- total_ms 是 localize 调用的壁钟时间，不含 dataset.load_image、valid_mask 获取、地图加载；原日志含首帧开销，未统一预热。CSV 另列每场景去前 10 帧均值，仅作敏感性诊断。不得称为完整端到端、稳定 10Hz。',
        '- 共享主机运行，性能百分位保留长尾；Hz = 1000 / 查询加权平均 ms，不平均逐帧 FPS。',
        '', '论文来源：[STDLoc](https://arxiv.org/html/2503.19358v1)，[ULF-Loc](https://arxiv.org/html/2605.04730v1)。外部数值转录配置见 configs/paper_baselines_20260910.json。','']
    for i,(title,header,rows) in enumerate(tables,1):
        md += [f'## Table {i}. {title}','','| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']
        md += ['| '+' | '.join(row)+' |' for row in rows];md.append('')
        with (args.output/f'table_{i:02d}.csv').open('w',newline='') as handle:writer=csv.writer(handle);writer.writerow(header);writer.writerows(rows)
        esc=lambda s:str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('>=',r'$\geq$').replace('<=',r'$\leq$')
        tex=['% '+title,r'\begin{tabular}{'+'l'*len(header)+'}',r'\hline',' & '.join(map(esc,header))+r' \\',r'\hline']
        tex += [' & '.join(map(esc,row))+r' \\' for row in rows];tex += [r'\hline',r'\end{tabular}']
        (args.output/f'table_{i:02d}.tex').write_text('\n'.join(tex)+'\n')
    md += ['## 论文实时性对照（不同测量口径，不计算直接加速比）','',
        '| 方法 / 来源 | Cambridge 耗时或速度 | 硬件 |','|---|---:|---|',
        '| STDLoc 原论文 Appendix A | 152.3 ms，约 6.57 Hz；正文约 7 FPS | RTX 4090 |',
        '| STDLoc，ULF-Loc 论文对照 | 3.9 FPS（倒数约 256.4 ms） | ULF-Loc 报告 RTX 4090 |',
        '| ULF-Loc，Appendix C.5 | 4.4 FPS（倒数约 227.3 ms） | RTX 4090 |','',
        '## 尚不能据此宣布论文实验全部完备','',
        '完整当前结果不等于原计划所有消融均已完成：V2/association/completion/fusion 的统一 24 场景消融、容量/建图视角预算曲线、所有场景三 seed、同硬件隔离端到端计时和统一峰值显存/RSS 尚无完整注册矩阵。历史弱先验 bug 前结果与离线反馈分支不进入主表。',
        '现有 V24–V36 在线研究保留为历史/负结果，V31 表是显式标注的可选示例，不把选过 test 的政策包装成独立泛化结果。','']
    (args.output/'paper_tables.md').write_text('\n'.join(md))
    (args.output/'results_inventory.json').write_text(json.dumps({'evaluations':inventory,'dataset_aggregates':aggregate,'paper_sources':baseline,'map_hashes':map_hashes},indent=2))
    keys=list(inventory[0])
    with (args.output/'all_evaluations.csv').open('w',newline='') as handle:writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(inventory)
    print('COMPLETE',args.output,'evaluations',len(inventory),flush=True)


if __name__=='__main__':main()
