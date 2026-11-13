"""Local Gradio entry point: python -m src.p3_app."""
from functools import partial
from threading import Lock
from src.p3_generation import DiffusionGenerator
from src.p3_orchestration import AgentOrchestrator
from src.p3_critique import VisionCritic
from src.p3_vision import load_qwen_vision, run_qwen_vision

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
        button.click(submit,[image,request,critique],[output,message,trace,download],concurrency_limit=1)
    return app

def main():
    processor,model=load_qwen_vision()
    pipeline=AgentOrchestrator(partial(run_qwen_vision,processor=processor,model=model),
                               DiffusionGenerator(),VisionCritic(processor,model))
    build_app(pipeline).queue().launch(server_name='127.0.0.1',share=False)

if __name__ == '__main__': main()
