'''Export independent human rating forms and import completed reviews.'''
import argparse
import html
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p1_evaluation import load_specification, make_review_template, summarize

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def export(run_path,spec_path,out,input_dir):
    run,spec=read(run_path),load_specification(spec_path)
    summarize(run,spec)
    if run['mode']!='final':raise ValueError('Human groundedness forms are for the final system.')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    template=make_review_template(run,spec)
    write(out/'final.review.blank.json',template)
    pdf_names={'DOC1':'Document.pdf','DOC2':'Document2_Supplement_A_Outturns_2011-2013.pdf','DOC3':'Document3_Briefing_B_Labour_Prices_Policy.pdf'}
    wanted=set()
    for row,case in zip(template['records'],spec['cases']):wanted.update(row['candidate']['citations']+case['citations'])
    assets=out/'source_pages';assets.mkdir(exist_ok=True)
    import fitz
    for citation in sorted(wanted):
        match=re.fullmatch(r'(DOC[123]):p(\d+)',citation)
        if not match:raise ValueError(f'Unexpected page citation: {citation}')
        doc,page=match[1],int(match[2])
        with fitz.open(Path(input_dir)/pdf_names[doc]) as pdf:
            pdf[page-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(assets/f'{doc}_p{page}.png')
    sections=[]
    esc=html.escape
    for row,case in zip(template['records'],spec['cases']):
        a=row['candidate'];qid=row['question_id']
        sources=[]
        for c in sorted(set(a['citations']+case['citations'])):
            d,p=c.split(':p');sources.append(f'<details><summary>{esc(c)}</summary><img loading="lazy" src="source_pages/{d}_p{p}.png" alt="Original course source {esc(c)}"></details>')
        sections.append(f'<section data-qid="{qid}"><h2>{qid} · conversation {esc(case["conversation_id"])} · turn {case["turn"]}</h2><p><b>Question:</b> {esc(row["question"])}</p><blockquote>{esc(a["answer"])}</blockquote><p>Status: {esc(a["status"])}<br>Citations: {esc(", ".join(a["citations"]) or "NONE")}<br>Figure references: {esc(", ".join(a["figure_references"]) or "NONE")}</p><label>Groundedness <select required><option value="">Choose independently</option>{"".join(f"<option value={v}>{v}</option>" for v in range(1,6))}</select></label><label>Notes <textarea rows="2"></textarea></label><details><summary>Reference and source pages</summary><p>{esc(row["reference"])}</p><p>{esc(row["scoring_notes"])}</p>{"".join(sources)}</details></section>')
    for member in ['A','B']:
        page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Part 1 independent groundedness review</title><style>body{font:16px/1.5 system-ui;max-width:980px;margin:32px auto;padding:0 20px;background:#f8fafc;color:#14213d}section{background:white;border:1px solid #ccd5df;border-radius:10px;margin:22px 0;padding:22px}h2{font-size:19px}label{display:block;margin:16px 0}textarea{display:block;width:95%}select,input,button,textarea{font:inherit;padding:8px}button{background:#173e75;color:white;border:0;border-radius:6px;cursor:pointer}blockquote{border-left:4px solid #7c9cc4;margin:12px 0;padding:12px;background:#f0f5fa}img{max-width:100%;height:auto}summary{cursor:pointer;margin:10px 0}</style><h1>Part 1 groundedness: member __MEMBER__</h1><p>Rate all 25 final answers independently. Do not consult the other member's scores. No score is prefilled. Use the cited original page images below and check every factual claim. These course documents contain constructed teaching values, not official statistics.</p><p><b>Rubric:</b> 1 = central claims unsupported or contradicted; 2 = major unsupported claims; 3 = mixed support with important gaps; 4 = mostly supported with a minor gap; 5 = every factual claim supported, with appropriate abstention where evidence is absent. Record concerns even when an answer sounds plausible.</p><form id="review"><label>Your full name <input id="name" required autocomplete="off"></label>__SECTIONS__<button type="submit">Export completed ratings</button><p id="status" role="status"></p></form><script>
const runHash=__HASH__, member=__LABEL__;
document.querySelector('#review').addEventListener('submit',e=>{e.preventDefault();const name=document.querySelector('#name').value.trim();if(!name){return;}const records=[...document.querySelectorAll('section')].map(s=>({question_id:s.dataset.qid,groundedness:Number(s.querySelector('select').value),note:s.querySelector('textarea').value.trim()}));if(records.length!==25||records.some(r=>!Number.isInteger(r.groundedness)||r.groundedness<1||r.groundedness>5)){alert('Complete all 25 ratings.');return;}const value={schema_version:1,run_sha256:runHash,mode:'final',member_label:member,reviewer_name:name,independent_human_review:true,exported_at:new Date().toISOString(),records};const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='part1-member-'+member+'-ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.querySelector('#status').textContent='Ratings exported. Keep the downloaded file and send it to the person assembling the report.';});</script></html>'''
        page=page.replace('__MEMBER__',member).replace('__LABEL__',json.dumps(member)).replace('__HASH__',json.dumps(run['run_sha256'])).replace('__SECTIONS__',''.join(sections))
        (out/f'member_{member}.html').write_text(page,encoding='utf-8')
    print('Exported two independent blank forms and original source page images:',out)

def merge(run_path,spec_path,review_path,a_path,b_path,out):
    run,spec,review=read(run_path),load_specification(spec_path),read(review_path)
    summarize(run,spec,review)
    expected=[r['question_id'] for r in review['records']]
    raters=[]
    for label,path in [('a',a_path),('b',b_path)]:
        data=read(path)
        if data.get('run_sha256')!=run['run_sha256'] or data.get('mode')!='final':raise ValueError('Rating form belongs to another run.')
        if data.get('independent_human_review') is not True:raise ValueError('An independent human review declaration is required.')
        name=data.get('reviewer_name','').strip()
        if not name:raise ValueError('Missing reviewer name.')
        if [r['question_id'] for r in data['records']]!=expected:raise ValueError('Rating rows are missing, duplicated or reordered.')
        review[f'reviewer_{label}_name']=name;raters.append(name.casefold())
        for dst,src in zip(review['records'],data['records']):
            value=src['groundedness']
            if type(value) is not int or not 1<=value<=5:raise ValueError('Each rating must be an integer from 1 to 5.')
            dst[f'reviewer_{label}_groundedness']=value
            dst[f'reviewer_{label}_note']=src.get('note','')
    if raters[0]==raters[1]:raise ValueError('The same person cannot count as both team members.')
    summarize(run,spec,review)
    write(out,review)
    print('Imported both independent reviews. Recompute the notebook metrics from',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    e=s.add_parser('export');e.add_argument('--run',required=True);e.add_argument('--spec',default=str(ROOT/'eval/p1_golden_v2.json'));e.add_argument('--output',required=True);e.add_argument('--input-dir',default=str(ROOT/'data/part1'))
    i=s.add_parser('import');i.add_argument('--run',required=True);i.add_argument('--spec',default=str(ROOT/'eval/p1_golden_v2.json'));i.add_argument('--review',required=True);i.add_argument('--member-a',required=True);i.add_argument('--member-b',required=True);i.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='export':export(a.run,a.spec,a.output,a.input_dir)
    else:merge(a.run,a.spec,a.review,a.member_a,a.member_b,a.output)
