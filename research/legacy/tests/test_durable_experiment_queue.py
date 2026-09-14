import json
import os
import sys
from scripts import durable_experiment_queue as q


def test_failed_job_does_not_block_independent_work(tmp_path,monkeypatch):
    monkeypatch.setattr(q.time,'sleep',lambda _:None)
    jobs=[{'id':'bad','command':[sys.executable,'-c','raise SystemExit(3)']},
          {'id':'dependent','command':[sys.executable,'-c','pass'],'deps':['bad']},
          {'id':'good','command':[sys.executable,'-c','pass']}]
    (tmp_path/'jobs.json').write_text(json.dumps(jobs));q.run(tmp_path)
    state=json.loads((tmp_path/'state.json').read_text())
    assert [state['jobs'][j['id']]['status'] for j in jobs]==['failed','blocked','complete']
    assert state['queue_finished'] and not state['all_complete']
    q.run(tmp_path)
    assert json.loads((tmp_path/'state.json').read_text())['jobs']['good']==state['jobs']['good']


def test_missing_declared_output_is_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(q.time,'sleep',lambda _:None)
    (tmp_path/'jobs.json').write_text(json.dumps([{'id':'missing','command':[sys.executable,'-c','pass'],
                                                'outputs':[str(tmp_path/'absent')]}]))
    q.run(tmp_path)
    assert json.loads((tmp_path/'state.json').read_text())['jobs']['missing']['status']=='failed'


def test_stale_running_is_interrupted_not_silently_reexecuted(tmp_path):
    (tmp_path/'jobs.json').write_text(json.dumps([{'id':'old','command':[sys.executable,'-c','pass']}]))
    (tmp_path/'state.json').write_text(json.dumps({'jobs':{'old':{'status':'running','pid':999999999,'process_identity':'0'}}}))
    q.run(tmp_path)
    assert json.loads((tmp_path/'state.json').read_text())['jobs']['old']['status']=='interrupted'
    assert q.identity(os.getpid()) is not None
