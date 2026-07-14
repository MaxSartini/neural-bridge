#!/usr/bin/env python3
"""Fresh control-complete blocked confirmation of original 3-checkpoint ensembles."""

from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from backend.scripts import run_again_dense_2hz_phase5_frozen_ar_residual as fr  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_learned_heads as base  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_binary_big_confirm as confirm  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase5_temporal_residual_blocked as temporal  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_fixed_blend_fresh5 as fixed  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_robust_multiseed_optuna as robust  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_blocked_15seed as blocked15  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_fresh_seed_validation as fresh  # noqa: E402
from backend.scripts import run_again_dense_2hz_phase6_trial4_three_checkpoint_fresh15 as ens  # noqa: E402

SCHEMA_VERSION = "again_dense_2hz_phase6_original_three_checkpoint_control_complete_v1"
SEEDS = tuple(range(20260660, 20260675))
GROUPS = tuple(tuple(SEEDS[i:i+3]) for i in range(0, 15, 3))
CONTROLS = blocked15.CONTROLS
PRIMARY_CONTROLS = confirm.PRIMARY_CONTROLS
EXPECTED_ROWS = 140

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--source-root",default=str(confirm.SOURCE_ROOT)); p.add_argument("--foldsafe-pca-root",default=str(confirm.FOLDSAFE_PCA_ROOT)); p.add_argument("--output-root",default=None); p.add_argument("--reports-dir",default="reports"); p.add_argument("--batch-size",type=int,default=8192); p.add_argument("--dry-run",action="store_true"); return p.parse_args()

def default_output_root() -> Path:
    return Path(f"outputs/again_dense_2hz_phase6_original_three_checkpoint_control_complete_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

def compute_verdict(rows: pd.DataFrame, audit_pass: bool) -> tuple[dict[str,Any],pd.DataFrame]:
    m=rows[rows.row_type.eq("member")]; e=rows[rows.row_type.eq("ensemble")]
    p=e.pivot(index="group",columns="lane",values="pr_auc").reset_index()
    primary=[f"{c}_ensemble" for c in PRIMARY_CONTROLS]
    p["best_control"]=p[primary].max(axis=1); p["real_minus_ar"]=p["real_residual_ensemble"]-p["frozen_ar_ensemble"]; p["real_minus_best_control"]=p["real_residual_ensemble"]-p["best_control"]
    member_real=m[m.lane.eq("real_residual_member")]; group_member=member_real.groupby("group").pr_auc.mean(); p["member_real_mean"]=p.group.map(group_member); p["real_minus_member_mean"]=p["real_residual_ensemble"]-p["member_real_mean"]
    means={c:float(p[f"{c}_ensemble"].mean()) for c in PRIMARY_CONTROLS}; best=max(means,key=means.get)
    real=float(p.real_residual_ensemble.mean()); ar=float(p.frozen_ar_ensemble.mean()); bestv=means[best]; label=float(p.label_permutation_residual_ensemble.mean())
    contribution=fixed.max_positive_contribution(p.real_minus_best_control)
    checks={
      "delta_vs_ar_at_least_0_005":real-ar>=.005,
      "delta_vs_best_control_at_least_0_005":real-bestv>=.005,
      "positive_vs_ar_5_of_5":int((p.real_minus_ar>0).sum())==5,
      "positive_vs_best_control_5_of_5":int((p.real_minus_best_control>0).sum())==5,
      "positive_medians":float(p.real_minus_ar.median())>0 and float(p.real_minus_best_control.median())>0,
      "ensemble_uplift_over_members_at_least_0_001":real-float(member_real.pr_auc.mean())>=.001,
      "ensemble_beats_group_member_mean_at_least_4_of_5":int((p.real_minus_member_mean>0).sum())>=4,
      "label_permutation_minus_ar_at_most_0_001":label-ar<=.001,
      "single_group_contribution_at_most_0_50":contribution<=.50,
      "exact_scope":len(rows)==EXPECTED_ROWS and set(m.seed)==set(SEEDS) and len(p)==5,
      "audit_pass":audit_pass,
    }
    failed=[k for k,v in checks.items() if not v]
    return {"schema_version":SCHEMA_VERSION,"rows_actual":int(len(rows)),"rows_expected":EXPECTED_ROWS,"real_ensemble_pr_auc":real,"ar_ensemble_pr_auc":ar,"best_control":best,"best_control_pr_auc":bestv,"real_minus_ar":real-ar,"real_minus_best_control":real-bestv,"real_minus_member_mean":real-float(member_real.pr_auc.mean()),"wins_vs_ar":int((p.real_minus_ar>0).sum()),"wins_vs_best_control":int((p.real_minus_best_control>0).sum()),"wins_vs_member_mean":int((p.real_minus_member_mean>0).sum()),"label_permutation_minus_ar":label-ar,"max_positive_group_contribution":contribution,"checks":checks,"failed_gates":failed,"blocked_control_complete_pass":not failed,"grouped_confirmation_authorized":not failed},p

def report_text(r:dict[str,Any],root:Path)->str:
    return f"""# Phase 6 Original Three-Checkpoint Control-Complete Confirmation

Output root: `{root}`

- rows: `{r['rows_actual']}/{EXPECTED_ROWS}`
- real / AR / best-control PR-AUC: `{r['real_ensemble_pr_auc']:.10f}` / `{r['ar_ensemble_pr_auc']:.10f}` / `{r['best_control_pr_auc']:.10f}`
- real minus AR / best control / member mean: `{r['real_minus_ar']:+.10f}` / `{r['real_minus_best_control']:+.10f}` / `{r['real_minus_member_mean']:+.10f}`
- wins vs AR / best control / member mean: `{r['wins_vs_ar']}/5` / `{r['wins_vs_best_control']}/5` / `{r['wins_vs_member_mean']}/5`
- blocked control-complete pass: `{r['blocked_control_complete_pass']}`
- failed gates: `{r['failed_gates']}`
"""

def main()->int:
    a=parse_args(); dry={"schema_version":SCHEMA_VERSION,"seeds":list(SEEDS),"groups":[list(g) for g in GROUPS],"controls":list(CONTROLS),"rows":EXPECTED_ROWS,"params":robust.ORIGINAL_PARAMS,"member_selection":False,"weight_search":False,"accelerator":"mlx"}; print(json.dumps(dry,indent=2,sort_keys=True))
    if a.dry_run:return 0
    base.require_mlx(); root=Path(a.output_root) if a.output_root else default_output_root()
    if root.exists() and any(root.iterdir()):raise FileExistsError(root)
    for s in ("ar_baseline_checkpoints","metrics","diagnostics","reports","manifests","models"):(root/s).mkdir(parents=True,exist_ok=True)
    started=time.time(); pca_root=Path(a.foldsafe_pca_root); blocks,df,dense_root,_=temporal.build_blocks(Path(a.source_root),pca_root); block=temporal.block_for_target(blocks,confirm.TARGET_NAME)
    rows=[];curves=[];audits=[];scores={}; group_of={seed:i+1 for i,g in enumerate(GROUPS) for seed in g}
    for seed in SEEDS:
      _,ac=fresh.train_ar_inner_only(block=block,seed=seed,output_root=root,batch_size=a.batch_size,max_epochs=80,patience=12);curves.extend({"model":"ar",**x} for x in ac);ar=blocked15.load_fresh_ar(root,block,seed,a.batch_size);scores[seed]={"frozen_ar":ar};am=temporal.metric_row_for_block(block,ar["train_score"],ar["test_score"],ar["test_reg"]);rows.append({"row_type":"member","group":group_of[seed],"seed":seed,"lane":"frozen_ar_member",**am})
      real_pack=temporal.feature_pack_for(df,dense_root,pca_root,block,confirm.ARCHITECTURE,"real_residual",seed)
      for control in CONTROLS:
        pack=real_pack if control=="real_residual" else temporal.feature_pack_for(df,dense_root,pca_root,block,confirm.ARCHITECTURE,control,seed)
        metrics,cs,audit=temporal.train_temporal_residual(architecture=confirm.ARCHITECTURE,control=control,pack=pack,block=block,ar=ar,seed=seed,output_root=root/"models",batch_size=a.batch_size,max_epochs=int(robust.ORIGINAL_PARAMS["max_epochs"]),patience=int(robust.ORIGINAL_PARAMS["patience"]),hyperparameters=robust.ORIGINAL_PARAMS)
        curves.extend({"model":control,**x} for x in cs);audits.append({"seed":seed,"control":control,**audit,"context":pack.context_audit});rows.append({"row_type":"member","group":group_of[seed],"seed":seed,"lane":f"{control}_member",**metrics});scores[seed][control]=fixed.restored_scores(audit=audit,params=robust.ORIGINAL_PARAMS,pack=pack,block=block,ar=ar,batch_size=a.batch_size)
      pd.DataFrame(rows).to_csv(root/"metrics/rows.partial.csv",index=False)
    for gid,g in enumerate(GROUPS,1):
      for key,lane in [("frozen_ar","frozen_ar_ensemble")]+[(c,f"{c}_ensemble") for c in CONTROLS]:
        avg=ens.average_scores([scores[s][key] for s in g]);metrics=temporal.metric_row_for_block(block,avg["train_score"],avg["test_score"],avg["test_reg"]);rows.append({"row_type":"ensemble","group":gid,"seed":0,"lane":lane,"member_seeds":",".join(map(str,g)),**metrics})
    frame=pd.DataFrame(rows); audit_pass=bool(len(audits)==90 and all(x["context"].get("temporal_context_causal_only") and x["context"].get("same_video_history_masking") and not x["context"].get("uses_centered_or_future_windows") and (x.get("checkpoint_restored") or x.get("residual_suppressed")) for x in audits)); result,gf=compute_verdict(frame,audit_pass);result.update({"duration_seconds":time.time()-started,"accelerator_detail":"Device(gpu, 0)"});frame.to_csv(root/"metrics/rows.csv",index=False);gf.to_csv(root/"metrics/group_deltas.csv",index=False);pd.DataFrame(curves).to_csv(root/"diagnostics/training_curves.csv",index=False);fr.write_json(root/"metrics/result.json",result);fr.write_json(root/"manifests/run_manifest.json",{**dry,"duration_seconds":result["duration_seconds"]});report=report_text(result,root);name=f"again_dense_2hz_phase6_original_three_checkpoint_control_complete_{root.name.rsplit('_',2)[-2]}_{root.name.rsplit('_',1)[-1]}.md";(root/"reports"/name).write_text(report,encoding="utf-8");rp=Path(a.reports_dir)/name;rp.parent.mkdir(parents=True,exist_ok=True);rp.write_text(report,encoding="utf-8");print(json.dumps({"run_completed":True,"output_root":str(root),"report":str(rp),**result},indent=2,sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
