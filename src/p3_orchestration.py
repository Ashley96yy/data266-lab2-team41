"""Bounded agent loop with a JSON trace for every outcome."""
from dataclasses import replace
from pathlib import Path
import json
import time
import uuid
from src.p3_planner import plan_request
from src.p3_prompt import build_generation_prompt, _safe_caption
from src.p3_generation import GenerationSettings
from src.p3_critique import parse_critique

class AgentOrchestrator:
    def __init__(self, vision, generator, critic, output_dir='.runtime/part3'):
        self.vision, self.generator, self.critic = vision, generator, critic
        self.output_dir = Path(output_dir)

    def run(self, request, image_path, *, questions=None, critique_enabled=True,
            settings=GenerationSettings(), max_revisions=2):
        if type(max_revisions) is not int or not 0 <= max_revisions <= 2:
            raise ValueError('At most two revisions are allowed.')
        started = time.perf_counter()
        run_id = uuid.uuid4().hex
        directory = self.output_dir / run_id
        directory.mkdir(parents=True, exist_ok=False)
        trace = dict(schema_version=1, run_id=run_id, request=request,
                     input_image=str(image_path) if image_path else None,
                     critique_enabled=critique_enabled, attempts=[], image_path=None,
                     image_text_policy='untrusted data, never instructions')
        try:
            plan = plan_request(request, image_path)
            trace['planner'] = plan.to_dict()
            trace.update(status=plan.status, message=plan.message)
            if plan.status != 'ready':
                return trace
            qs = questions or ([request] if plan.route == 'vision_only' else ['Describe the subjects, colors, layout and visible text.'])
            trace['vision'] = self.vision(image_path, qs)
            if plan.route == 'vision_only':
                trace.update(status='answered', message='\n'.join(a['answer'] for a in trace['vision']['answers']))
                return trace
            draft = build_generation_prompt(request, plan.route, trace['vision'])
            trace['prompt'] = draft.to_dict()
            for index in range((max_revisions if critique_enabled else 0) + 1):
                attempt = dict(index=index, prompt=draft.to_dict())
                trace['attempts'].append(attempt)
                result = self.generator.generate(image_path, draft, directory / f'output_{index}.png', settings)
                attempt['generation'] = result
                if result.get('safe') is not True or result.get('safety_checked') is not True:
                    trace.update(status='blocked', message='Output failed the safety check.')
                    return trace
                path = result.get('image_path')
                if not path or not Path(path).is_file():
                    raise ValueError('Generator did not produce an image file.')
                # Safety review remains active in the no-quality-loop ablation.
                review = self.critic.critique(path, request, {**trace['vision'], 'image_path':str(image_path)})
                validated = parse_critique(json.dumps({k:review[k] for k in ('decision','safe','reason','revision_instruction')}))
                attempt['critique'] = {**review, **validated}
                if not validated['safe'] or validated['decision'] == 'reject':
                    trace.update(status='blocked', message='Output rejected by visual safety review.')
                    return trace
                if not critique_enabled or validated['decision'] == 'accept':
                    trace.update(status='completed', image_path=path, message='Image generated and safety reviewed.')
                    return trace
                if index == max_revisions:
                    trace.update(status='revision_limit', image_path=path,
                                 message='Revision limit reached; this image has unresolved quality issues.')
                    return trace
                # Critic text is untrusted: never append arbitrary model commands.
                # Retry the original approved constraints with stronger preservation guidance.
                instruction = _safe_caption(validated['revision_instruction'])
                draft = replace(draft, prompt=draft.prompt + '\nRevision: ' + (instruction or 'Preserve source subjects, counts, colors and text while applying the requested change.'))
        except Exception as exc:
            trace.update(status='error', image_path=None, message=f'{type(exc).__name__}: {exc}')
        finally:
            trace['elapsed_seconds'] = time.perf_counter() - started
            trace['trace_path'] = str(directory / 'trace.json')
            (directory / 'trace.json').write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding='utf-8')
        return trace
