#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(); p.add_argument('baseline',type=Path); p.add_argument('candidate',type=Path); p.add_argument('--out-dir',type=Path,default=ROOT/'runs'); args=p.parse_args()
    b=json.loads(args.baseline.read_text()); c=json.loads(args.candidate.read_text()); recipe=c['recipe']; speedup=None
    bt=b['summary'].get('median_completion_tokens_per_sec') or 0; ct=c['summary'].get('median_completion_tokens_per_sec') or 0
    if bt>0: speedup=round(ct/bt,3)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-comparison'; od=args.out_dir/run_id; od.mkdir(parents=True,exist_ok=True)
    auto_clean=c.get('auto_quality_clean') is True
    bundle={'schema':'benchloop.operator_result.v3','run_id':run_id,'created_at':datetime.now(timezone.utc).isoformat(),'recipe':recipe,'source_lane_results':{'baseline':str(args.baseline.resolve()),'candidate':str(args.candidate.resolve())},'summary':{'baseline':b['summary'],'candidate':{**c['summary'],'speedup_vs_baseline':speedup}},'results':{'baseline':b['results'],'candidate':c['results']},'manual_review':{'quality_pass':None,'auto_quality_clean':auto_clean,'reliability_pass':c['summary'].get('error_count')==0,'speed_pass':None if speedup is None else speedup>=1.5,'decision':'pending_manual_quality_review','notes':'Manual quality review required before BenchLoop import.'},'benchloop_record_fields':{'lane':recipe['id'],'hardware':recipe['hardware'],'runtime':recipe['runtime'],'model':recipe['model'],'candidate':recipe['candidate'],'baseline':recipe['baseline'],'quality_pass':None,'auto_quality_clean':auto_clean,'reliability_pass':c['summary'].get('error_count')==0,'speedup':speedup,'decision':'pending','raw_result_path':str((od/'operator-result.json').resolve())}}
    path=od/'operator-result.json'; path.write_text(json.dumps(bundle,indent=2),encoding='utf-8')
    ingest=args.out_dir/'benchloop-ingest.jsonl'; ingest.open('a',encoding='utf-8').write(json.dumps(bundle['benchloop_record_fields'])+'\n')
    print(path); print(json.dumps(bundle['summary'],indent=2))
    return 0 if auto_clean and speedup and speedup>=1.3 else 3
if __name__=='__main__': raise SystemExit(main())
