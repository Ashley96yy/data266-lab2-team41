"""Local independent review forms. Never uploads images or fills human labels."""
import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
import shutil


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
MODES=['with','without','red_team']


def export(run_dir,output):
    root=Path(run_dir);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if any((out/f'member_{m}.html').exists() for m in ['A','B']):
        raise ValueError('Use a new review folder; do not overwrite an existing form.')
    assets=out/'images';assets.mkdir(exist_ok=True)
    hashes={mode:sha(root/f'{mode}.json') for mode in MODES};esc=html.escape
    def picture(path,label):
        if not path or not Path(path).is_file():return '<p>No image was returned.</p>'
        suffix=Path(path).suffix.lower();name=sha(path)[:20]+suffix
        if not (assets/name).exists():shutil.copy2(path,assets/name)
        return f'<figure><img loading="lazy" src="images/{name}" alt="{esc(label)}"><figcaption>{esc(label)}</figcaption></figure>'
    cards=[]
    for mode in MODES:
        run=read(root/f'{mode}.json')
        if not run['execution_complete']:raise ValueError('Review only complete runs.')
        for record in run['records']:
            trace=record['trace'];case=record['case_id']
            # Real-person inputs rejected before Vision are never copied into the gallery.
            source=picture(trace['input_image'],'Source') if trace.get('vision') else '<p>Stopped before Vision; source preview withheld.</p>'
            vision=esc(json.dumps(trace.get('vision'),indent=2,ensure_ascii=False)) if trace.get('vision') else 'No Vision call.'
            attempt_html=[]
            for attempt in trace['attempts']:
                if 'critique' not in attempt:continue
                path=attempt.get('generation',{}).get('image_path')
                attempt_html.append(f'<div class="attempt" data-index="{attempt["index"]}"><h3>Generated attempt {attempt["index"]}</h3>{picture(path,"Candidate output")}<label>Your image judgment <select class="accept" required><option value="">Choose independently</option><option value="accept">Accept</option><option value="revise">Revise or reject</option></select></label><textarea placeholder="Optional image-review notes"></textarea></div>')
            cards.append(f'<section data-mode="{mode}" data-case="{case}"><h2>{esc(case)} · {mode}</h2><p><b>Request:</b> {esc(trace["request"])}</p><ul>{"".join("<li>"+esc(item)+"</li>" for item in record["rubric"])}</ul>{source}<p><b>System status:</b> {esc(trace["status"])}<br>{esc(trace["message"])}</p><details><summary>Observed caption and visual answers</summary><pre>{vision}</pre></details>{"".join(attempt_html)}<label>Does the complete case pass every rubric clause? <select class="case-pass" required><option value="">Choose independently</option><option value="pass">Pass</option><option value="fail">Fail</option></select></label><textarea class="case-note" placeholder="Reason for your case judgment"></textarea></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Part 3 independent human review</title><style>body{font:16px/1.5 system-ui;max-width:1050px;margin:30px auto;padding:0 18px;color:#172b4d;background:#f5f7fa}section{padding:22px;margin:24px 0;border:1px solid #cbd4df;border-radius:12px;background:white}h2{font-size:21px}h3{font-size:18px}figure{display:inline-block;margin:12px 8px;vertical-align:top}img{max-width:440px;max-height:350px;width:auto;height:auto}label{display:block;margin:14px 0}textarea{display:block;width:95%;min-height:65px}select,input,textarea,button{font:inherit;padding:8px}button{background:#184877;color:white;border:0;padding:12px 18px;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.attempt{border-top:1px solid #ccd5df;margin-top:18px}</style><h1>Part 3: member __MEMBER__</h1><p>Review all 40 case executions independently against their frozen rubrics. For each generated attempt, judge whether it should be accepted, revised or rejected before consulting the model critic. Critic predictions and CLIP values are omitted here. Do not consult the other member's labels.</p><p>No scores are prefilled. Some cases correctly refuse or request clarification; use the rubric, not the word completed, to decide. This gallery contains synthetic course inputs and local generated outputs. Rejected real-person portrait inputs are not copied into it. Keep the review folder local.</p><form id="review"><label>Your full name <input id="name" required></label>__CARDS__<button type="submit">Export completed independent review</button><p id="status"></p></form><script>
const hashes=__HASHES__,member=__LABEL__;
document.querySelector('#review').addEventListener('submit',e=>{e.preventDefault();const name=document.querySelector('#name').value.trim();if(!name)return;const records=[...document.querySelectorAll('section')].map(s=>({mode:s.dataset.mode,case_id:s.dataset.case,pass:s.querySelector('.case-pass').value==='pass',note:s.querySelector('.case-note').value.trim(),attempts:[...s.querySelectorAll('.attempt')].map(a=>({index:Number(a.dataset.index),human_accept:a.querySelector('.accept').value==='accept',note:a.querySelector('textarea').value.trim()}))}));const value={schema_version:1,raw_execution_hashes:hashes,member_label:member,reviewer_name:name,independent_human_review:true,records,exported_at:new Date().toISOString()};const b=new Blob([JSON.stringify(value,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='part3-member-'+member+'-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.querySelector('#status').textContent='Independent review exported. Keep this file for adjudication.';});</script></html>'''
    for member in ['A','B']:
        (out/f'member_{member}.html').write_text(page.replace('__MEMBER__',member).replace('__LABEL__',json.dumps(member)).replace('__HASHES__',json.dumps(hashes)).replace('__CARDS__',''.join(cards)),encoding='utf-8')
    write(out/'review_contract.json',dict(raw_execution_hashes=hashes,expected_cases=len(cards),human_labels_prefilled=False))
    print('Exported two blank local forms:',out)


def merge(run_dir,a_path,b_path,output):
    root=Path(run_dir);a,b=read(a_path),read(b_path)
    hashes={mode:sha(root/f'{mode}.json') for mode in MODES}
    for form in [a,b]:
        if form.get('raw_execution_hashes')!=hashes:raise ValueError('Review belongs to another run.')
        if form.get('independent_human_review') is not True or not form.get('reviewer_name','').strip():raise ValueError('A named independent human reviewer is required.')
    if a['reviewer_name'].strip().casefold()==b['reviewer_name'].strip().casefold():raise ValueError('Two different reviewers are required.')
    lookup=[]
    expected=[(mode,r['case_id']) for mode in MODES for r in read(root/f'{mode}.json')['records']]
    for form in [a,b]:
        ids=[(r['mode'],r['case_id']) for r in form['records']]
        if ids!=expected:raise ValueError('Missing, duplicated or reordered case reviews.')
        lookup.append(dict(zip(ids,form['records'])))
    out=Path(output);out.mkdir(parents=True,exist_ok=False);unresolved=[]
    for mode in MODES:
        run=deepcopy(read(root/f'{mode}.json'));run['raw_execution_sha256']=hashes[mode]
        run.update(reviewer_a_name=a['reviewer_name'],reviewer_b_name=b['reviewer_name'])
        for record in run['records']:
            x,y=(l[(mode,record['case_id'])] for l in lookup)
            if type(x['pass']) is not bool or type(y['pass']) is not bool:raise ValueError('Pass labels must be boolean.')
            record.update(reviewer_a_pass=x['pass'],reviewer_b_pass=y['pass'],reviewer_a_note=x.get('note',''),reviewer_b_note=y.get('note',''),adjudicated_pass=x['pass'] if x['pass']==y['pass'] else None,resolution_note='')
            if record['adjudicated_pass'] is None:unresolved.append(dict(mode=mode,case_id=record['case_id'],kind='case'))
            reviewed_attempts=[z for z in record['trace']['attempts'] if 'critique' in z]
            indices=[z['index'] for z in reviewed_attempts]
            if [z['index'] for z in x['attempts']]!=indices or [z['index'] for z in y['attempts']]!=indices:raise ValueError('Wrong generated-attempt identities.')
            for attempt,ax,ay in zip(reviewed_attempts,x['attempts'],y['attempts']):
                if type(ax['human_accept']) is not bool or type(ay['human_accept']) is not bool:raise ValueError('Attempt labels must be boolean.')
                attempt.update(reviewer_a_accept=ax['human_accept'],reviewer_b_accept=ay['human_accept'],human_accept=ax['human_accept'] if ax['human_accept']==ay['human_accept'] else None,human_accept_resolution_note='')
                if attempt['human_accept'] is None:unresolved.append(dict(mode=mode,case_id=record['case_id'],kind='attempt',index=attempt['index']))
        write(out/f'{mode}.json',run)
    write(out/'adjudication_needed.json',unresolved)
    manifest=read(root/'manifest.json')
    manifest['review_derivation']=dict(raw_run_dir=str(root.resolve()),raw_execution_hashes=hashes,
        note='Real independent labels imported; resolve disagreements in the new copies. Original inference records are unchanged.')
    write(out/'manifest.json',manifest)
    print('Merged actual labels; unresolved disagreements remain blank:',len(unresolved))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    e=s.add_parser('export');e.add_argument('--run-dir',required=True);e.add_argument('--output',required=True)
    i=s.add_parser('import');i.add_argument('--run-dir',required=True);i.add_argument('--member-a',required=True);i.add_argument('--member-b',required=True);i.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='export':export(args.run_dir,args.output)
    else:merge(args.run_dir,args.member_a,args.member_b,args.output)
