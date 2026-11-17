def run_qwen_generation(messages, processor, model):
    from unittest.mock import patch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers.integrations import sdpa_attention as hf_sdpa
    import gc
    inputs=None; output_ids=None; generated_ids=None
    trace=[]
    def observe_decode(module,args,kwargs):
        ids=kwargs.get('input_ids')
        length=int(ids.shape[-1]) if ids is not None else None
        trace.append(length)
        if len(trace)>1 and length is not None and length>1:
            raise RuntimeError('Cached decoding unexpectedly reused multiple prompt tokens. Preserve generation_diagnostics.jsonl before changing the runtime.')
    hook=model.register_forward_pre_hook(observe_decode,with_kwargs=True)
    started=time.perf_counter()
    try:
        template=processor.chat_template or processor.tokenizer.chat_template
        if not template:raise ValueError('The VLM processor does not provide a chat template.')
        inputs=processor.apply_chat_template(messages,chat_template=template,
            add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt').to(VLM_DEVICE)
        input_length=int(inputs['input_ids'].shape[1])
        config=copy.deepcopy(model.generation_config)
        config.do_sample=False;config.num_beams=1;config.num_return_sequences=1
        config.use_cache=True;config.cache_implementation='dynamic'
        config.max_new_tokens=VLM_MAX_NEW_TOKENS
        config.output_hidden_states=False;config.output_attentions=False
        config.output_scores=False;config.output_logits=False;config.return_dict_in_generate=False
        config.temperature=None;config.top_p=None;config.top_k=None
        # Preserve grouped-query semantics using Transformers' repeat_kv path.
        with torch.inference_mode(), patch.object(hf_sdpa,'use_gqa_in_sdpa',new=lambda *args, **kwargs: False), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output_ids=model.generate(**inputs,generation_config=config,logits_to_keep=1)
        generated_ids=output_ids[:,input_length:]
        raw=processor.batch_decode(generated_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        result={'raw_output':raw,'input_tokens':input_length,'generated_tokens':int(generated_ids.shape[1]),
                'elapsed_seconds':round(time.perf_counter()-started,3)}
        diagnostic={'status':'completed','input_tokens':input_length,
            'generated_tokens':result['generated_tokens'],'forward_input_lengths':trace,
            'peak_allocated_gib':torch.cuda.max_memory_allocated(0)/1024**3}
        with (RUNTIME_OUTPUT_ROOT/'generation_diagnostics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(diagnostic)+'\n')
        return result
    except Exception as error:
        diagnostic={'status':'failed','error_type':type(error).__name__,
            'forward_input_lengths':trace,'allocated_gib':torch.cuda.memory_allocated(0)/1024**3,
            'reserved_gib':torch.cuda.memory_reserved(0)/1024**3}
        with (RUNTIME_OUTPUT_ROOT/'generation_diagnostics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(diagnostic)+'\n')
        raise
    finally:
        hook.remove()
        inputs=None;output_ids=None;generated_ids=None
        gc.collect();torch.cuda.empty_cache()
