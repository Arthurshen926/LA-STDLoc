"""Read-only attestation of stages rejected by unrelated archive-script changes."""
import argparse
import json
from pathlib import Path
from common.hashing import sha256_file


UNRELATED={'scripts/execute_stdloc_cleanup_dry_run_v2.py',
           'scripts/generate_lafgs_archive_full_dry_run_20260909.py',
           'scripts/execute_lafgs_archive_full_dry_run_20260909.py'}


def audit(command_file, snapshot):
    old=json.loads(command_file.read_text())['input_contract']
    mismatches=[]
    for relative,digest in old['code'].items():
        if relative in UNRELATED:continue
        p=snapshot/relative
        if not p.is_file() or sha256_file(p)!=digest:mismatches.append(relative)
    for name,digest in old['inputs'].items():
        p=Path(name)
        if not p.is_file() or sha256_file(p)!=digest:mismatches.append(name)
    if mismatches:raise ValueError(f'actual stage dependencies changed: {mismatches}')
    return {'source_command':str(command_file),'command_sha256':sha256_file(command_file),
            'all_recorded_inputs_match':True,'all_original_algorithm_modules_match':True,
            'excluded_nonalgorithm_scripts':sorted(UNRELATED),
            'caveat':'Current content equivalence, not continuous historical file monitoring.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--command',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--map-report',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=audit(a.command,a.snapshot)
    if a.map_report:
        report=json.loads(a.map_report.read_text());out=report['output']
        for key in ['map','metric','association_graph']:
            if sha256_file(Path(out[key]))!=out[key+'_sha256']:raise ValueError('map artifact changed')
        r['map_report_sha256']=sha256_file(a.map_report)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2));print('PASS',a.command,flush=True)


if __name__=='__main__':main()
