import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_cambridge_v14_f0_scene.sh"
EVALUATOR = ROOT / "scripts" / "evaluate_shopfacade_mapfim_field_ablation.sh"


def test_v14_f0_runner_freezes_the_formal_cross_scene_protocol():
    text = RUNNER.read_text(encoding="utf-8")

    assert "detector_covsoft_fixlineage_30000" in text
    assert "SHOP_MAPFIM_DUSTBIN_WEIGHT=0.25" in text
    assert "SHOP_MAPFIM_MAP_CLEANLINESS_WEIGHT=0.5" in text
    assert "SHOP_MAPFIM_MAP_BIAS_WEIGHT=0.75" in text
    assert 'F0 "$GPU" 2000' in text
    assert '2000 2000 "$EVAL_SUBSET"' in text
    assert "strict2dgs-lafgs-v14f0-$SCENE" in text


def test_mapfim_evaluator_uses_scene_specific_result_names():
    text = EVALUATOR.read_text(encoding="utf-8")

    assert 'SCENE="${SHOP_MAPFIM_SCENE:-ShopFacade}"' in text
    assert 'PREFIX="mapfim-baseline-$SCENE"' in text
    assert "RESULT_PREFIX_OVERRIDE" in text


def test_v14_f0_shell_scripts_parse():
    for path in (RUNNER, EVALUATOR):
        subprocess.run(["bash", "-n", str(path)], check=True)
