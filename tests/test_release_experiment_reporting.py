import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('release_reporting',
    Path(__file__).resolve().parents[1]/'tools/report_release_validation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('names,denominator,valid', [
    (['a','b'],2,True), (['a'],1,False), (['a','a'],2,False),
    (['b','a'],2,False), (['a','b'],1,False)])
def test_complete_means_full_ordered_query_coverage(tmp_path, names, denominator, valid):
    (tmp_path/'manifest.json').write_text(json.dumps({'queries':['a','b']}))
    (tmp_path/'results.json').write_text(json.dumps([{'image_name':n} for n in names]))
    (tmp_path/'summary.json').write_text(json.dumps({'query_count':denominator}))
    if valid:
        assert module.checked_run(tmp_path)['query_count']==2
    else:
        with pytest.raises(ValueError):
            module.checked_run(tmp_path)
