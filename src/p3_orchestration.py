"""Five typed workflow stages, bounded revisions, and complete auditable traces."""
from dataclasses import replace
from pathlib import Path
import json
import time
import uuid
from src.p3_planner import plan_request, _UNSAFE_CONTENT, _DECEPTIVE_CLAIM
from src.p3_prompt import build_generation_prompt, _safe_caption
from src.p3_generation import GenerationSettings
from src.p3_critique import parse_critique
from src.p3_state import AgentState


def safe_revision_feedback(value):
    cleaned=_safe_caption(value)
    if _UNSAFE_CONTENT.search(cleaned) or _DECEPTIVE_CLAIM.search(cleaned):
        return ''
    return cleaned


class AgentOrchestrator:
    def __init__(self,vision,generator,critic,output_dir='.runtime/part3',prompt_engineer=build_generation_prompt):
        self.vision,self.generator,self.critic=vision,generator,critic
        self.output_dir=Path(output_dir)
        self.prompt_engineer=prompt_engineer

    def _event(self,state,agent,inputs,call):
        started=time.perf_counter()
        event=dict(agent=agent,input=inputs,status='running')
        state.agent_events.append(event)
        try:
            output=call()
            event.update(status='completed',output=output.to_dict() if hasattr(output,'to_dict') else output)
            return output
        except Exception as exc:
            event.update(status='error',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            event['elapsed_seconds']=time.perf_counter()-started

    def _plan(self,state):
        plan=self._event(state,'planner',dict(request=state.request,image_path=state.input_image),
                         lambda:plan_request(state.request,state.input_image))
        state.planner=plan.to_dict();state.status=plan.status;state.message=plan.message
        return plan

    def _observe(self,state,questions):
        state.vision=self._event(state,'vision',dict(image_path=state.input_image,questions=questions),
                                lambda:self.vision(state.input_image,questions))

    def _engineer(self,state,route):
        draft=self._event(state,'prompt_engineering',dict(request=state.request,route=route,vision=state.vision),
                          lambda:self.prompt_engineer(state.request,route,state.vision))
        state.prompt=draft.to_dict()
        return draft

    def _generate(self,state,draft,target,settings):
        return self._event(state,'generation',dict(source=state.input_image,prompt=draft.to_dict(),settings=settings.__dict__,target=str(target)),
            lambda:self.generator.generate(state.input_image,draft,target,settings))

    def _revise_prompt(self,state,draft,instruction,index):
        revised=self._event(state,'prompt_engineering',dict(approved_request=state.request,
            previous_prompt=draft.to_dict(),sanitized_feedback=instruction,revision_index=index),
            lambda:replace(draft,prompt=draft.prompt+'\nRevision: '+instruction,
                           revision_instruction=instruction,revision_index=index))
        state.prompt=revised.to_dict()
        return revised

    def _review(self,state,path,result):
        review=self._event(state,'critique',dict(source=state.input_image,output=path,request=state.request),
            lambda:self.critic.critique(path,state.request,{**state.vision,'image_path':state.input_image,'generation':result}))
        validated=parse_critique(json.dumps({k:review[k] for k in ('decision','safe','reason','revision_instruction')}))
        return {**review,**validated}

    def _run_stages(self,state,questions,settings,max_revisions,directory):
        plan=self._plan(state)
        if plan.status!='ready':return
        qs=list(questions) if questions else ([state.request] if plan.route=='vision_only' else ['Describe the subjects, colors and spatial layout.'])
        if len(qs)<2:qs.append('What readable text and uncertain small details are visible?')
        self._observe(state,qs)
        if plan.route=='vision_only':
            import re
            caption_only=bool(re.search(r'\b(?:caption|describe)\b',state.request,re.I) and not re.search(r'\b(?:and|text|word|how many|what)\b',state.request,re.I))
            answers=state.vision['answers']
            exact=next((a['answer'] for a in answers if a['question']==state.request),None)
            state.status='answered'
            state.message=state.vision['caption'] if caption_only else exact or '\n'.join(a['answer'] for a in answers)
            return
        draft=self._engineer(state,plan.route)
        for index in range((max_revisions if state.critique_enabled else 0)+1):
            attempt=dict(index=index,prompt=draft.to_dict());state.attempts.append(attempt)
            result=self._generate(state,draft,directory/f'output_{index}.png',settings)
            attempt['generation']=result
            if result.get('safe') is not True or result.get('safety_checked') is not True:
                state.status='blocked';state.message='Output failed the safety check.';return
            path=result.get('image_path')
            if not path or not Path(path).is_file():raise ValueError('Generator did not produce an image file.')
            review=self._review(state,path,result);attempt['critique']=review
            if not review['safe']:
                state.status='blocked';state.message='Output rejected by visual safety review.';return
            if not state.critique_enabled or review['decision']=='accept':
                state.status='completed';state.image_path=path
                state.message='Image enhanced and safety reviewed.' if plan.route=='enhancement' else 'Image generated and safety reviewed.'
                return
            if index==max_revisions:
                state.status='revision_limit';state.image_path=path
                state.message='Revision limit reached; this image has unresolved quality issues.';return
            instruction=safe_revision_feedback(review['revision_instruction'])
            attempt['revision_feedback_used']=instruction
            attempt['revision_feedback_discarded']=not bool(instruction)
            instruction=instruction or 'Preserve the original subjects, counts, colors and text while applying the requested change.'
            draft=self._revise_prompt(state,draft,instruction,index+1)

    def run(self,request,image_path,*,questions=None,critique_enabled=True,settings=GenerationSettings(),max_revisions=2):
        if type(max_revisions) is not int or not 0<=max_revisions<=2:
            raise ValueError('At most two revisions are allowed.')
        started=time.perf_counter();run_id=uuid.uuid4().hex
        directory=self.output_dir/run_id;directory.mkdir(parents=True,exist_ok=False)
        state=AgentState(run_id=run_id,request=request,input_image=str(image_path) if image_path else None,
                         critique_enabled=critique_enabled)
        try:
            self._run_stages(state,questions,settings,max_revisions,directory)
        except Exception as exc:
            state.status='error';state.image_path=None
            if 'out of memory' in str(exc).casefold():
                state.message='GPU memory is full. Close other GPU applications and retry.'
            elif 'text limit' in str(exc).casefold():
                state.message='The request is too long for image generation. Please shorten it.'
            elif hasattr(exc,'model_attempts'):
                state.message='The model could not return a valid review. No result was accepted; please retry.'
            else:
                state.message='The model could not finish this request. The diagnostic trace contains details.'
            if hasattr(exc,'model_attempts'):state.model_format_failures=exc.model_attempts
            state.agent_events.append(dict(agent='workflow',status='error',error=f'{type(exc).__name__}: {exc}'))
        finally:
            state.elapsed_seconds=time.perf_counter()-started;state.trace_path=str(directory/'trace.json')
            trace=state.to_dict();temporary=directory/'trace.json.tmp'
            temporary.write_text(json.dumps(trace,indent=2,ensure_ascii=False),encoding='utf-8')
            temporary.replace(state.trace_path)
        return trace
