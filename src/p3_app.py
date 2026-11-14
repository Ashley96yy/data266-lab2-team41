"""Local Gradio entry point: python -m src.p3_app."""
from functools import partial
from threading import Lock
from src.p3_generation import DiffusionGenerator
from src.p3_orchestration import AgentOrchestrator
from src.p3_critique import VisionCritic
from src.p3_vision import load_qwen_vision, run_qwen_vision

def english_ui_javascript():
    """Select English through Gradio's locale API after frontend initialization."""
    from pathlib import Path
    import gradio as gr
    assets = Path(gr.__file__).parent / 'templates' / 'frontend' / 'assets'
    for asset in sorted(assets.glob('index-*.js')):
        exports = asset.read_text(encoding='utf-8')
        if 'changeLocale' in exports and 'setupi18n' in exports:
            return (
                '(async () => { '
                f'const {{ changeLocale }} = await import("/assets/{asset.name}"); '
                'changeLocale("en"); })();'
            )
    raise RuntimeError('Cannot locate the Gradio locale API for the English interface.')

def build_app(orchestrator):
    import gradio as gr
    lock = Lock()
    def submit(image, request, critique):
        with lock:
            trace = orchestrator.run(request, image, critique_enabled=critique)
        return trace.get('image_path'), trace['message'], trace, trace['trace_path']
    with gr.Blocks(title='Team 41 Image Agents') as app:
        gr.Markdown('# Image Understanding and Editing\nUpload an image, then ask a question or request a style change or enhancement.')
        image=gr.Image(type='filepath',label='Source image')
        request=gr.Textbox(label='Request',max_lines=6)
        critique=gr.Checkbox(value=True,label='Enable quality critique and up to two revisions')
        button=gr.Button('Run')
        output=gr.Image(label='Result')
        message=gr.Textbox(label='Status')
        trace=gr.JSON(label='Agent trace')
        download=gr.File(label='Download trace')
        button.click(submit,[image,request,critique],[output,message,trace,download],
                     concurrency_limit=1,api_name='run')
    return app

def main():
    import argparse
    import json
    import os
    from pathlib import Path
    from src.p3_vision import VISION_MODEL_ID
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revisions',default='.runtime/model-revisions.json')
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    revisions_path=Path(args.revisions)
    revisions=json.loads(revisions_path.read_text(encoding='utf-8')) if revisions_path.is_file() else {}
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
    cached=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots'/revisions.get('vision','missing')
    processor,model=load_qwen_vision(model_id=str(cached) if cached.is_dir() else VISION_MODEL_ID,
                                    revision=revisions.get('vision'))
    pipeline=AgentOrchestrator(partial(run_qwen_vision,processor=processor,model=model),
                               DiffusionGenerator(base_revision=revisions.get('base'),
                                                  lora_revision=revisions.get('lora')),
                               VisionCritic(processor,model))
    from src.p3_prompt_rewrite import QwenPromptEngineer
    pipeline.prompt_engineer=QwenPromptEngineer(processor,model,pipeline.critic.similarity.get_tokenizer)
    build_app(pipeline).queue().launch(server_name='127.0.0.1',server_port=args.port,
                                      share=False,js=english_ui_javascript())

if __name__ == '__main__': main()
