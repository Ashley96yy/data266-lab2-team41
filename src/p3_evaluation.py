"""Unscored case execution and independent-human-review aggregation."""
import hashlib
import json
from pathlib import Path
from statistics import median

def load_cases(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def run_cases(orchestrator, specification, *, root='.', critique_enabled=True,
              checkpoint_path=None):
    from src.p3_generation import GenerationSettings
    root=Path(root).resolve()
    # Verify the entire frozen fixture bundle before any model call.
    import subprocess, sys
    subprocess.run([sys.executable,str(root/'eval/verify_p3_fixture_manifest.py'),'--require-complete'],cwd=root,check=True)
    records=[]
    result=dict(set_id=specification['set_id'],critique_enabled=critique_enabled,
                specification_sha256=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest(),
                records=records, scored=False, execution_complete=False)
    for case in specification['cases']:
        trace=orchestrator.run(case['request'],root/case['fixture'],
                    questions=case.get('vision_questions'),critique_enabled=critique_enabled,
                    settings=GenerationSettings(seed=specification['seed']))
        records.append(dict(case_id=case['id'],trace=trace,human_pass=None,
                            rubric=case['pass_if']))
        print(f"{case['id']}: {trace['status']} ({trace['elapsed_seconds']:.1f}s)",flush=True)
        if checkpoint_path:
            target=Path(checkpoint_path)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    result['execution_complete']=True
    if checkpoint_path:
        Path(checkpoint_path).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result

def summarize(reviewed):
    records=reviewed['records']
    expected=15 if reviewed['set_id']=='p3_primary_15' else 10
    ids=[r['case_id'] for r in records]
    prefix='P3-S' if expected==15 else 'P3-RT'
    if set(ids)!={f'{prefix}{i:02d}' for i in range(1,expected+1)} or len(ids)!=expected:
        raise ValueError('Incomplete or duplicate evaluation cases.')
    names=[reviewed.get('reviewer_a_name'),reviewed.get('reviewer_b_name')]
    if any(not isinstance(name,str) or not name.strip() for name in names) or names[0].strip().casefold()==names[1].strip().casefold():
        raise ValueError('Two distinct named human reviewers are required.')
    for record in records:
        a,b,resolved=(record.get(k) for k in ('reviewer_a_pass','reviewer_b_pass','adjudicated_pass'))
        if type(a) is not bool or type(b) is not bool or type(resolved) is not bool:
            raise ValueError('Two independent reviews and adjudication are required.')
        if a!=b and not record.get('resolution_note'):
            raise ValueError('Document disagreement resolution.')
        if a==b and resolved!=a:
            raise ValueError('Adjudication conflicts with agreed reviews.')
        if record['trace']['status']=='error':
            raise ValueError('Resolve execution errors before scored aggregation.')
    passed=sum(r['adjudicated_pass'] for r in records)
    comparisons=[]
    for r in records:
        for attempt in r['trace']['attempts']:
            if 'critique' not in attempt:continue
            human=attempt.get('human_accept')
            if type(human) is bool:
                comparisons.append(human == (attempt['critique']['decision']=='accept'))
    total_attempts=sum('critique' in a for r in records for a in r['trace']['attempts'])
    complete_critic_review=len(comparisons)==total_attempts and total_attempts>0
    return dict(denominator=expected,passes=passed,task_success_rate=passed/expected if expected==15 else None,
                attack_successes=expected-passed if expected==10 else None,
                attack_success_rate=(expected-passed)/expected if expected==10 else None,
                reviewer_agreement=sum(r['reviewer_a_pass']==r['reviewer_b_pass'] for r in records)/expected,
                review_disagreements=sum(r['reviewer_a_pass']!=r['reviewer_b_pass'] for r in records),
                median_seconds=median(r['trace']['elapsed_seconds'] for r in records),
                critic_agreement=sum(comparisons)/len(comparisons) if complete_critic_review else None,
                critic_agreement_denominator=total_attempts,critic_reviewed_attempts=len(comparisons))


def main():
    import argparse
    from functools import partial
    from src.p3_vision import load_qwen_vision, run_qwen_vision
    from src.p3_critique import VisionCritic
    from src.p3_generation import DiffusionGenerator
    from src.p3_orchestration import AgentOrchestrator
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',default='eval/p3_scenarios.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--without-critique',action='store_true')
    parser.add_argument('--vision-revision',required=True)
    parser.add_argument('--base-revision',required=True)
    parser.add_argument('--lora-revision',required=True)
    args=parser.parse_args()
    processor,model=load_qwen_vision(revision=args.vision_revision)
    pipeline=AgentOrchestrator(partial(run_qwen_vision,processor=processor,model=model),
        DiffusionGenerator(base_revision=args.base_revision,lora_revision=args.lora_revision),
        VisionCritic(processor,model))
    from src.p3_prompt_rewrite import QwenPromptEngineer
    pipeline.prompt_engineer=QwenPromptEngineer(processor,model,pipeline.critic.similarity.get_tokenizer)
    result=run_cases(pipeline,load_cases(args.spec),critique_enabled=not args.without_critique,
                     checkpoint_path=args.output)
    target=Path(args.output); target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__ == '__main__': main()
