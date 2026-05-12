#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, json, re, statistics, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parent

def load_json(p:Path): return json.loads(p.read_text(encoding='utf-8'))
def req(method,url,payload=None,timeout=240):
    data=None if payload is None else json.dumps(payload).encode()
    r=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'},method=method)
    try:
        with urllib.request.urlopen(r,timeout=timeout) as resp:
            raw=resp.read().decode('utf-8','replace')
            return resp.status,json.loads(raw),raw
    except urllib.error.HTTPError as e:
        raw=e.read().decode('utf-8','replace')
        try: body=json.loads(raw)
        except Exception: body={'error':raw}
        return e.code,body,raw
    except Exception as e:
        return 0,{'error':str(e)},''
def probe(ep,timeout):
    s,b,raw=req('GET',ep.rstrip('/')+'/v1/models',timeout=timeout)
    return {'status':s,'ok':200<=s<300,'raw':raw[:4000]}
def chat(ep,model,prompt,temp,max_tokens,timeout):
    payload={'model':model,'temperature':temp,'max_tokens':max_tokens,'messages':[{'role':'user','content':prompt}]}
    t=time.perf_counter(); s,b,raw=req('POST',ep.rstrip()+'/v1/chat/completions',payload,timeout); dt=time.perf_counter()-t
    choice=(b.get('choices') or [{}])[0] if isinstance(b,dict) else {}; msg=choice.get('message') or {}; usage=b.get('usage') or {} if isinstance(b,dict) else {}
    content=msg.get('content') or choice.get('text') or ''; ct=int(usage.get('completion_tokens') or 0)
    return {'status':s,'ok':200<=s<300 and bool(content.strip()),'error':'' if 200<=s<300 else str(b.get('error',b))[:1000],'content':content,'raw_response':b,'latency_ms':round(dt*1000,2),'completion_tokens':ct,'prompt_tokens':int(usage.get('prompt_tokens') or 0),'completion_tokens_per_sec':round(ct/dt,2) if dt>0 and ct else 0.0}
def hard_hits(content,patterns):
    hits=[p for p in patterns if p and p in content]
    for pat in [r'from\s+\w+\s*=\s*\w+',r'@app\.get\([^\n]*:\s*$',r'<0x[0-9A-Fa-f]+>',r'(.)\1{24,}']:
        if re.search(pat,content,re.M): hits.append('regex:'+pat)
    return sorted(set(hits))
def auto(pid,c):
    s=c.strip(); out={'auto_pass':None,'checks':[]}
    try:
        if pid in {'json_object_only','csv_to_json','tool_json_call'}:
            p=json.loads(s); out['checks'].append('json_parse_ok')
            out['auto_pass']=(isinstance(p,dict) and isinstance(p.get('name'),str) and isinstance(p.get('age'),(int,float)) and isinstance(p.get('tags'),list)) if pid=='json_object_only' else ((isinstance(p,list) and len(p)==2) if pid=='csv_to_json' else (isinstance(p,dict) and p.get('tool')=='send_email' and p.get('arguments',{}).get('to')=='sam@example.com'))
        elif pid in {'fastapi_health','fib_with_tests'}:
            ast.parse(s); out['checks'].append('python_ast_ok'); out['auto_pass']='```' not in s
        elif pid=='small_reasoning': out['auto_pass']=s=='67'
        elif pid=='exact_five_bullets': out['auto_pass']=len([l for l in s.splitlines() if re.match(r'^\s*(-|\*|\d+[.)])\s+',l)])==5
        elif pid=='specdec_tokenizer_120_words':
            words=re.findall(r'\b\w+\b',s); out['auto_pass']=90<=len(words)<=150 and 'token' in s.lower()
    except Exception as e:
        out['checks'].append('auto_check_error:'+str(e)); out['auto_pass']=False
    return out
def summary(rs):
    ok=[r for r in rs if r.get('ok')]; tps=[r['completion_tokens_per_sec'] for r in ok if r.get('completion_tokens_per_sec',0)>0]; known=[r for r in rs if r.get('auto',{}).get('auto_pass') is not None]; ap=sum(1 for r in known if r['auto']['auto_pass'] is True)
    return {'requests':len(rs),'ok_requests':len(ok),'error_count':len(rs)-len(ok),'hard_fail_count':sum(1 for r in rs if r.get('hard_fail_hits')),'auto_checked':len(known),'auto_pass_count':ap,'auto_pass_rate':round(ap/len(known),3) if known else None,'median_completion_tokens_per_sec':round(statistics.median(tps),2) if tps else 0.0,'mean_completion_tokens_per_sec':round(sum(tps)/len(tps),2) if tps else 0.0,'total_completion_tokens':sum(int(r.get('completion_tokens') or 0) for r in rs)}
def main():
    a=argparse.ArgumentParser(); a.add_argument('--lane',required=True,choices=['baseline','candidate']); a.add_argument('--endpoint',required=True); a.add_argument('--model',required=True); a.add_argument('--recipe',type=Path,default=ROOT/'recipe.json'); a.add_argument('--prompts',type=Path,default=ROOT/'prompts.json'); a.add_argument('--out-dir',type=Path,default=ROOT/'runs'); a.add_argument('--repeats',type=int,default=None); a.add_argument('--max-tokens',type=int,default=None); a.add_argument('--temperature',type=float,default=None); a.add_argument('--timeout',type=int,default=240)
    args=a.parse_args(); recipe=load_json(args.recipe); prompts=load_json(args.prompts); shape=recipe['run_shape']; reps=args.repeats or int(shape['repeats']); mt=args.max_tokens or int(shape['max_tokens']); temp=args.temperature if args.temperature is not None else float(shape['temperature'])
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+args.lane; od=args.out_dir/rid; od.mkdir(parents=True,exist_ok=True)
    pr=probe(args.endpoint,args.timeout); print('probe',args.lane,pr['status'],pr['ok']);
    if not pr['ok']: raise SystemExit('endpoint unreachable')
    results=[]
    for p in prompts:
        for rep in range(1,reps+1):
            r=chat(args.endpoint,args.model,p['prompt'],temp,mt,args.timeout); r.update({'lane':args.lane,'endpoint':args.endpoint,'model':args.model,'prompt_id':p['id'],'prompt_category':p.get('category',''),'repeat':rep,'manual_pass_criteria':p.get('manual_pass_criteria',[]),'hard_fail_hits':hard_hits(r['content'],p.get('hard_fail_patterns',[])),'auto':auto(p['id'],r['content'])}); results.append(r)
            sig='ERR' if not r['ok'] else ('HARD_FAIL' if r['hard_fail_hits'] else ('AUTO_FAIL' if r['auto'].get('auto_pass') is False else 'OK'))
            print(f"{p['id']} #{rep}: {sig} {r['completion_tokens_per_sec']} tok/s")
    summ=summary(results); bundle={'schema':'benchloop.lane_result.v1','run_id':rid,'created_at':datetime.now(timezone.utc).isoformat(),'lane':args.lane,'recipe':recipe,'endpoint_probe':pr,'summary':summ,'results':results,'auto_quality_clean':summ['error_count']==0 and summ['hard_fail_count']==0 and summ.get('auto_pass_rate')==1.0}
    path=od/f'{args.lane}-result.json'; path.write_text(json.dumps(bundle,indent=2),encoding='utf-8'); print(path); print(json.dumps(summ,indent=2)); return 3 if not bundle['auto_quality_clean'] else 0
if __name__=='__main__': raise SystemExit(main())
