"""Measured operational tables; missing human outcomes stay missing."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean,median
from src.p3_evaluation import summarize


def measured_rows(run_dir):
    rows=[];case_rows=[]
    for mode in ['with','without','red_team']:
        run=json.loads((Path(run_dir)/f'{mode}.json').read_text(encoding='utf-8'))
        if not run.get('execution_complete'):raise ValueError('A complete execution is required.')
        records=run['records'];expected=15 if mode!='red_team' else 10
        if len(records)!=expected:raise ValueError('Incorrect case count.')
        attempts=[a for r in records for a in r['trace']['attempts']]
        clips=[a['critique']['clip_cosine_similarity'] for a in attempts if 'critique' in a]
        first=[r['trace']['attempts'][0]['critique']['clip_cosine_similarity'] for r in records if r['trace']['attempts'] and 'critique' in r['trace']['attempts'][0]]
        last=[r['trace']['attempts'][-1]['critique']['clip_cosine_similarity'] for r in records if r['trace']['attempts'] and 'critique' in r['trace']['attempts'][-1]]
        row=dict(mode=mode,cases=len(records),execution_errors=sum(r['trace']['status']=='error' for r in records),
            statuses=dict(Counter(r['trace']['status'] for r in records)),
            median_seconds_per_request=median(r['trace']['elapsed_seconds'] for r in records),
            generation_attempts=len(attempts),clip_scored_attempts=len(clips),
            mean_first_clip_cosine=mean(first) if first else None,mean_final_clip_cosine=mean(last) if last else None,
            task_success_rate=None,passes=None,attack_successes=None,attack_success_rate=None,
            reviewer_agreement=None,critic_agreement=None,critic_agreement_denominator=len(clips),
            human_review_complete=False)
        complete=all(type(r.get(key)) is bool for r in records for key in ['reviewer_a_pass','reviewer_b_pass','adjudicated_pass'])
        if complete:
            row.update(summarize(run),human_review_complete=True)
        rows.append(row)
        for r in records:
            t=r['trace'];gens=[a['generation'] for a in t['attempts'] if 'generation' in a]
            case_rows.append(dict(mode=mode,case_id=r['case_id'],status=t['status'],
                generation_attempts=len(t['attempts']),elapsed_seconds=t['elapsed_seconds'],
                vision_questions=len((t.get('vision') or {}).get('answers',[])),
                backends=','.join(sorted({g['backend'] for g in gens})),
                clip_cosines=[a['critique']['clip_cosine_similarity'] for a in t['attempts'] if 'critique' in a],
                prompt_omissions=[g['prompt_components_omitted'] for g in gens],
                trace_path=t['trace_path'],human_pass=None))
    return rows,case_rows


def measured_markdown(rows,manifest):
    def number(value):return 'Pending' if value is None else f'{value:.4f}' if isinstance(value,float) else str(value)
    fields=[('Cases','cases'),('Execution errors','execution_errors'),('Generated attempts','generation_attempts'),
            ('CLIP-scored attempts','clip_scored_attempts'),('Mean first-output CLIP cosine','mean_first_clip_cosine'),
            ('Mean final-output CLIP cosine','mean_final_clip_cosine'),('Median seconds/request','median_seconds_per_request'),
            ('Human rubric passes','passes'),('Human task success rate','task_success_rate'),
            ('Human attack successes','attack_successes'),('Independent reviewer agreement','reviewer_agreement'),
            ('Critic/human agreement','critic_agreement')]
    text='# Part 3 measured development rerun\n\n'
    text+=f"Source commit: `{manifest['source_commit']}`. Host: {manifest['gpu']}.\n\n"
    text+='This is a development rerun after documented implementation fixes, using the unchanged frozen cases. Flow completion and CLIP similarity are not human pass/fail labels. Both members must independently review the rubric outcomes; blank human metrics remain pending.\n\n'
    text+='| Measurement | With quality loop | Without quality loop | Red team |\n| --- | ---: | ---: | ---: |\n'
    for label,key in fields:text+='| '+label+' | '+' | '.join(number(r[key]) for r in rows)+' |\n'
    text+='\nOperational statuses, not human success rates:\n\n'
    for row in rows:text+=f"- {row['mode']}: {json.dumps(row['statuses'],sort_keys=True)}\n"
    if manifest.get('status')=='completed' and manifest.get('actual_t4_verified'):
        hardware_note='Actual Tesla T4 hardware verified by this completed run.'
    elif manifest.get('status')=='completed' and manifest.get('actual_p100_verified'):
        hardware_note='Actual Tesla P100 hardware verified by this completed run.'
    else:
        hardware_note='Actual T4/P100 hardware remains unverified for this run.'
    text+=f"\nObserved peak PyTorch allocation: {manifest['peak_allocated_gib']:.3f} GiB; peak reserved: {manifest['peak_reserved_gib']:.3f} GiB, under a {manifest['allocator_budget_gib']:.1f} GiB allocator budget. {hardware_note} Total diagnostic and evaluation runtime: {manifest['elapsed_seconds']:.2f} seconds.\n"
    text+='\nThe no-quality-loop mode still runs the same output safety review and records the same diagnostic CLIP score. Its quality decision cannot trigger a revision. This is a revision-loop ablation, not removal of all safety checks. First-generation requests, prompts, negative prompts, model revisions, backends and settings were checked equal across the paired modes. Later prompts/photometric revision strengths may change within the fixed two-revision bound.\n'
    text+='\nEnhancement is explicitly a geometry-preserving Pillow photometric backend with zero diffusion steps, followed by the existing SD safety checker and VLM review. Style and domain routes still use real SD img2img; domain routes use the pinned team LoRA. A low CLIP score alone never becomes an unsafe label, and a high score does not establish correct objects, text or safety. Critic agreement is withheld until every generated attempt has its human accept/nonaccept label.\n'
    text+='\nEvery experimental number in this report is calculated from saved records. The independent human metrics are not filled by Codex. Inspect images and full traces before interpreting quality or safety, and retain any revision-limit or blocked outcomes.\n'
    return text
