"""Export all candidate metrics and revised paper tables (requires XlsxWriter)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import xlsxwriter
from analyze_paper_metrics import error_metrics
from figure_cache_reader import read_cache

REPO = Path(__file__).resolve().parents[1]
POOL = Path('/mnt/pool/sqy')
CAMPAIGN = POOL/'anygsloc_supplement_20261009'
CATALOG = REPO/'artifacts/metric_catalog_20261009'
EVIDENCE = REPO/'docs/paper/evidence_20260917'

LABELS = dict(complete='完整 selector／前缀＋位姿补点', full_map='Full（未压缩参照）',
    selector_quality='全局质量 Top-K', selector_spatial='空间均匀选择', classic_spatial='经典 RGB 三角化',
    geometry_ray='共同观测 Ray', geometry_depth='共同观测 Depth', geometry_center='共同观测 Center',
    no_screening='关闭 V2 筛选', no_completion='关闭候选轨迹补全',
    observation_all_off='观测因素全部关闭', mean_descriptor='普通多视图均值描述子',
    single_descriptor='确定性单观测描述子', quality_completion='相同前缀＋质量补点',
    selection_core='可靠核心点', selection_matching='核心＋匹配补全前缀',
    ransac_B='RANSAC B', ransac_C='RANSAC C', native_D='RANSAC D',
    prior_vanilla_2dgs='Vanilla 2DGS', prior_vanilla_3dgs='Vanilla 3DGS',
    prior_i3dgs_raw='i3DGS raw', prior_i3dgs_1k='i3DGS +1k',
    prior_long_lrm_raw='Long-LRM raw', prior_long_lrm_1k='Long-LRM +1k')
HEADERS = dict(label='条件',condition='条件 ID',stage='定位阶段',mask='本阶段语义 mask',
    median_TE_cm='中位 TE (cm)',mean_TE_cm='平均 TE (cm)',P90_TE_cm='P90 TE (cm)',P95_TE_cm='P95 TE (cm)',
    median_RE_deg='中位 RE (deg)',mean_RE_deg='平均 RE (deg)',P90_RE_deg='P90 RE (deg)',P95_RE_deg='P95 RE (deg)',
    total_failures='无有效解查询数',failures='无有效解查询数',failure_pct='无有效解比例 (%)',
    catastrophic_100cm_pct='TE≥100 cm 比例 (%)',successful_only_mean_TE_cm='仅有效解平均 TE (cm)',
    successful_only_mean_RE_deg='仅有效解平均 RE (deg)',points='五场景总点数',map_MB='五场景地图 MB',
    mean_ms='平均完整定位 ms',p95_ms='场景 P95 宏平均 ms',timing_source='计时批次',
    dataset='数据集',scene='场景',queries='查询数',psnr_db='参考视图 PSNR (dB)')
for t in [1,2,5,10,15,20,25,30,50,100]:
    HEADERS[f'R{t}_pct']=f'R({t}cm,5deg) %'
for t,a in [(1,1),(2,2),(200,5),(500,10)]:
    HEADERS[f'R{t}cm_{a}deg_pct']=f'R({t}cm,{a}deg) %'


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    hashes={}
    def read(path):
        raw=path.read_bytes(); hashes[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def csv_read(path):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as f: rows=list(csv.DictReader(f))
        for r in rows:
            for k,v in list(r.items()):
                if k not in ['condition','scene','method','mapping_mask']:
                    try:r[k]=float(v)
                    except (ValueError,TypeError):pass
        return rows
    macro=csv_read(CATALOG/'macro.csv'); metrics={r['condition']:r for r in macro}
    protocol=read(CAMPAIGN/'protocol.json')
    resources={}; map_inventory={}
    def inventory(key,paths):
        counts=[];sizes=[]
        for path in paths:
            state=read_cache(path)
            counts.append(state['anchor_xyz'].size[0]);sizes.append(path.stat().st_size)
            state['anchor_xyz'].storage[0].close()
        resources[key]=dict(points=sum(counts),map_MB=sum(sizes)/1e6)
        map_inventory[key]=[dict(path=str(p),points=n,bytes=p.stat().st_size) for p,n in zip(paths,counts)]
    inventory('complete',[Path(s['compact_map']) for s in protocol['scenes'].values()])
    for key in ['native_D','ransac_B','ransac_C']:
        resources[key]=dict(resources['complete'])
    for key,group,arm in [('selector_quality','selector','quality'),('selector_spatial','selector','spatial'),
            ('geometry_ray','geometry','ray'),('geometry_depth','geometry','depth'),('geometry_center','geometry','center')]:
        inventory(key,[CAMPAIGN/group/s/arm/'deployment_map.pt' for s in protocol['scenes']])
    inventory('classic_spatial',[CAMPAIGN/'classic'/s/'fit/spatial/deployment_map.pt' for s in protocol['scenes']])
    for key in ['selection_core','selection_matching','quality_completion']:
        sub={'selection_core':'core','selection_matching':'matching','quality_completion':'equal_size_quality'}[key]
        paths=[]
        for scene in protocol['scenes']:
            control=read(REPO/'paper/AnyGSLoc_Transactions_v7/data/selector_controls'/f'{scene}.json')
            paths.append(Path(control['evaluated_maps'][sub]['path']))
        inventory(key,paths)
    suite=POOL/'anygsloc_cambridge_suite_v2_20260914/maps'
    for key in ['mean_descriptor','single_descriptor']:
        inventory(key,[suite/('ablation_'+s+'__'+key)/'deployment_map.pt' for s in protocol['scenes']])
    prior_cells=read(POOL/'anygsloc_release_priors_20260913/registry.json')['cells']
    for key in [k for k in metrics if k.startswith('prior_')]:
        paths=[]
        for scene in protocol['scenes']:
            cell=next(c for c in prior_cells if c['key']=='CambridgePrior/'+scene+'/'+key.removeprefix('prior_'))
            path=Path(cell['deployment_map'])
            if not path.is_file():path=suite/('prior_'+scene+'_'+key.removeprefix('prior_'))/'deployment_map.pt'
            paths.append(path)
        inventory(key,paths)
    factorial=csv_read(EVIDENCE/'factorial_macro.csv')
    for r in factorial:
        key=f"factorial_v{int(r['V2'])}_f{int(r['fusion'])}_c{int(r['completion'])}_{r['selection']}"
        resources[key]=dict(points=int(r['total_anchors']),map_MB=r['total_map_MB'],
            mean_ms=r['core_mean_ms'],timing_source='20260914 factorial, three warmed rounds')
    for name,key in [('full_map','v1_f1_c1_full'),('no_screening','v0_f1_c1_compact'),
                     ('no_completion','v1_f1_c0_compact'),('observation_all_off','v0_f0_c0_compact')]:
        resources[name]=dict(resources['factorial_'+key])
    resources['single_descriptor'].update({k:resources['factorial_v1_f0_c1_compact'][k] for k in ['mean_ms','timing_source']})
    for r in read(CAMPAIGN/'macro_results.json')['results']:
        if r['stage']=='warmed_timing':
            key={'native_A':'complete','native_D':'native_D'}.get(r['method'],r['method'])
            if key in resources:
                resources[key].update(mean_ms=r['mean_ms'],p95_ms=r['p95_ms'],timing_source='20261009, three warmed rounds')
    for r in csv_read(EVIDENCE/'baseline_timing_macro.csv'):
        for stage,col in [('coarse','coarse_prefix_ms'),('fine','coarse_plus_fine_ms')]:
            key=f"{r['method']}_{r['mapping_mask']}_{stage}"
            resources[key]=dict(mean_ms=r[col],timing_source='20260915 native baseline timers')
    psnr=read(REPO/'artifacts/prior_psnr_reference_20261008/summary.json')
    for prior,value in psnr['five_scene_macro_psnr_db'].items():
        key='complete' if prior=='matcha' else 'prior_'+prior
        resources[key]['psnr_db']=value
    def select(keys,context=''):
        result=[]
        for key in keys:
            r=dict(metrics[key],**resources.get(key,{}))
            if key.startswith(('STDLoc_','ULFLoc_')):
                method,setting,stage=key.split('_'); label=method.replace('ULFLoc','ULF-Loc')
                r.update(label=label,mask='✓' if setting=='on' else '✗',
                         stage='单阶段（稀疏）' if stage=='coarse' else '两阶段（稀疏＋密集）')
            else:
                r.update(label=LABELS.get(key,key))
                if context=='系统':r.update(mask='✗†',stage='单阶段')
            if key=='complete':
                r['label']={'系统':'AnyGSLoc','融合':'本文多视图融合','RANSAC':'RANSAC A（原配置）',
                            '先验':'MAtCha','观测':'完整配置'}.get(context,LABELS[key])
            result.append(r)
        return result
    systems=[f'{m}_{setting}_{stage}' for m in ['STDLoc','ULFLoc'] for setting in ['on','off'] for stage in ['coarse','fine']]+['classic_spatial','complete']
    candidates=systems
    selection=['full_map','selection_core','selection_matching','selector_spatial','selector_quality','quality_completion','complete']
    book=xlsxwriter.Workbook(output/'实验指标候选总表.xlsx',{'strings_to_formulas':False,'strings_to_urls':False})
    header=book.add_format({'bold':True,'bg_color':'#173E56','font_color':'white','text_wrap':True,'valign':'vcenter'})
    number=book.add_format({'num_format':'0.000'}); wrapped=book.add_format({'text_wrap':True,'valign':'top'})
    sheets=[]
    def emit(name,rows,fields=None):
        if fields is None:
            fields=['label','condition']+[k for k in rows[0] if k not in ['label','condition']]
            for r in rows:
                fields += [k for k in r if k not in fields]
        sheet=book.add_worksheet(name); sheet.freeze_panes(1,min(2,len(fields)))
        sheet.set_row(0,42);sheet.set_column(0,0,33);sheet.set_column(1,1,32);sheet.set_column(2,len(fields)-1,19)
        for j,k in enumerate(fields):sheet.write(0,j,HEADERS.get(k,k),header)
        for i,r in enumerate(rows,1):
            for j,k in enumerate(fields):
                value=r.get(k)
                if value is None:continue
                if isinstance(value,(int,float,np.number)):
                    if math.isfinite(value):sheet.write_number(i,j,float(value),number)
                    else:sheet.write(i,j,'∞' if value>0 else 'NA')
                else:sheet.write(i,j,str(value))
        sheet.autofilter(0,0,len(rows),len(fields)-1)
        sheet.set_landscape();sheet.fit_to_pages(1,0);sheet.repeat_rows(0)
        with (output/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        sheets.append(dict(sheet=name,rows=len(rows),columns=len(fields)))
    notes=[
        ('用途','全部候选指标供作者选择；新增指标是对既有结果的事后描述性分析，不代表显著性检验。'),
        ('系统候选表','正文候选版含语义标记、单/两阶段、TE/RE、R5、mean latency；不放 R15。所有阈值仍在完整指标页。'),
        ('语义标记','✓/✗ 表示本地建图或转换阶段显式使用/不使用语义 mask。✗† 的转换输入是预建 MAtCha Gaussian，上游使用过语义。所有 query 关闭语义 mask。'),
        ('单/两阶段','单阶段 baseline 是仅稀疏初始化输出；两阶段是稀疏＋密集最终输出，时间包含稀疏前缀。'),
        ('选择表','完整 selector 就是“相同前缀＋位姿补点”，只列一次；相同前缀＋质量补点另列一行。'),
        ('预算','Full、核心、匹配前缀点数自然不同；空间、全局 Top-K、质量补点、完整 selector 每场景最终 K 相同。'),
        ('统计','Cambridge 是五场景等权平均：平均各场景中位数、均值、P90、P95；不是合并全部查询后再算这些指标。'),
        ('召回','RXX_pct 默认是 TE < XX cm 且 RE < 5 deg；另有明确命名的 1cm/1deg、2cm/2deg、200cm/5deg、500cm/10deg。均使用严格小于。'),
        ('失败','无有效解查询误差记为∞、召回记为失败；总失败数是五场景之和，失败比例是场景等权平均。有效解也可能误差很大。'),
        ('条件均值','successful_only_mean 仅对有效解统计，不能省略“仅有效解”或脱离失败率与全样本均值混比。'),
        ('时间','仅填已有预热计时，不用精度遍历/CPU solver replay时间冒充完整在线时间。空白表示本工作簿未汇集对应测量，不是零。'),
        ('存储','本轮 mask-on 基线没有匹配的部署资产审计，所以该行地图大小留空；不拼接 mask-off 的资产。MB=字节/1e6，共享网络权重另计。'),
        ('先验 PSNR','5,364 个参考视图，无 mask，全分辨率；是参考视图重建质量，不是测试视图泛化 PSNR。Long-LRM 未计算该项，留空。'),
        ('已有结果','不改原始实验、不按新指标重新选择算法参数；完整指标及原始来源哈希一并保留。')]
    emit('说明',[dict(项目=a,说明=b) for a,b in notes],['项目','说明'])
    book.get_worksheet_by_name('说明').set_column(1,1,110,wrapped)
    for i in range(1,len(notes)+1):book.get_worksheet_by_name('说明').set_row(i,45)
    emit('正文候选_系统',select(candidates,'系统'),['label','mask','stage','median_TE_cm','median_RE_deg','R5_pct','mean_ms'])
    emit('正文候选_选择',select(['full_map','selector_spatial','selector_quality','quality_completion','complete']),
        ['label','points','map_MB','median_TE_cm','mean_TE_cm','median_RE_deg','R5_pct'])
    groups=[('系统_全部指标',systems),('几何_全部指标',['geometry_center','geometry_depth','geometry_ray']),
        ('观测_全部指标',['observation_all_off','no_screening','single_descriptor','no_completion','complete']),
        ('融合_全部指标',['single_descriptor','mean_descriptor','complete']),('选择_全部指标',selection),
        ('RANSAC_全部指标',['complete','ransac_B','ransac_C','native_D']),
        ('先验_全部指标',['complete']+[k for k in metrics if k.startswith('prior_')]),
        ('组合_16配置',[k for k in metrics if k.startswith('factorial_')]),('全部宏平均',list(metrics))]
    for name,keys in groups:emit(name,select(keys,name.split('_')[0]))
    emit('Cambridge_逐场景',csv_read(CATALOG/'per_scene.csv'))
    primary=[]
    for path in sorted((POOL/'anygsloc_protocol_v2_primary_20260914/runs').glob('*/*/results.json')):
        rows=read(path)
        te=[r['translation_error_cm'] if r['success'] else np.inf for r in rows]
        re=[r['rotation_error_deg'] if r['success'] else np.inf for r in rows]
        primary.append(dict(dataset=path.parents[1].name,scene=path.parent.name,queries=len(rows),**error_metrics(te,re)))
    assert len(primary)==24 and sum(r['queries'] for r in primary)==24700
    emit('24场景_全部指标',primary,list(primary[0]))
    datasets=[]
    for dataset in ['Cambridge','7Scenes','12Scenes']:
        group=[r for r in primary if r['dataset']==dataset]
        datasets.append(dict(dataset=dataset,scenes=len(group),queries=sum(r['queries'] for r in group),
            total_failures=sum(r['failures'] for r in group),
            **{k:float(np.mean([r[k] for r in group])) for k in group[0] if k not in ['dataset','scene','queries','failures']}))
    emit('3数据集_汇总',datasets,list(datasets[0]))
    curves=read(CATALOG/'recall_curves.json')
    curve_rows=[dict(condition=k,threshold_cm=t,rotation_deg=5,recall_pct=y)
        for k,v in curves['macros'].items() for t,y in zip(curves['translation_thresholds_cm'],v)]
    emit('完整阈值曲线',curve_rows,list(curve_rows[0]))
    emit('指标字段',[dict(field=k,display=v) for k,v in HEADERS.items()],['field','display'])
    book.close()
    read(CATALOG/'provenance.json')
    (output/'manifest.json').write_text(json.dumps(dict(sheets=sheets,
        source_hashes=hashes,map_inventory=map_inventory,raw_metric_provenance=str(CATALOG/'provenance.json'),
        tool='XlsxWriter '+xlsxwriter.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2,ensure_ascii=False))
    print('Workbook:',output/'实验指标候选总表.xlsx')
    print('Sheets:',len(sheets),'Cambridge conditions:',len(metrics),'per-scene rows:',len(csv_read(CATALOG/'per_scene.csv')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
