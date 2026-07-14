from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
from backend.scripts import run_again_dense_2hz_phase6_original_three_checkpoint_control_complete as run

def test_scope_and_controls_are_fixed():
    assert run.SEEDS==tuple(range(20260660,20260675));assert len(run.GROUPS)==5;assert all(len(g)==3 for g in run.GROUPS);assert run.EXPECTED_ROWS==140
    assert set(run.PRIMARY_CONTROLS)<=set(run.CONTROLS)

def test_dry_run_forbids_selection_and_search():
    c=subprocess.run([sys.executable,str(Path(run.__file__).resolve()),"--dry-run"],check=True,capture_output=True,text=True);p=json.loads(c.stdout);assert p["member_selection"] is False;assert p["weight_search"] is False;assert p["accelerator"]=="mlx"

def test_plan_is_control_complete_and_fail_closed():
    text=(run.REPO_ROOT/"docs/phase6_original_three_checkpoint_control_complete_plan.md").read_text();assert "exactly `140` rows" in text;assert "Failure stops before grouped" in text;assert "label-permutation" in text
