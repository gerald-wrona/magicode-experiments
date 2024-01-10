Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
C:\Users\wrona\anaconda3\envs\rag\Lib\site-packages\transformers\generation\configuration_utils.py:389: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
Setting `pad_token_id` to `eos_token_id`:32014 for open-end generation.
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[1], line 21
     14 prompt = MAGICODER_PROMPT.format(instruction=instruction)
     15 generator = pipeline(
     16     model="ise-uiuc/Magicoder-S-DS-6.7B",
     17     task="text-generation",
     18     torch_dtype=torch.bfloat16,
     19     device_map="auto",
     20 )
---> 21 result = generator(prompt, max_length=2048, num_return_sequences=1, temperature=0.0)
     22 print(result[0]["generated_text"])

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\pipelines\text_generation.py:208, in TextGenerationPipeline.__call__(self, text_inputs, **kwargs)
    167 def __call__(self, text_inputs, **kwargs):
    168     """
    169     Complete the prompt(s) given as inputs.
    170 
   (...)
    206           ids of the generated text.
    207     """
--> 208     return super().__call__(text_inputs, **kwargs)

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\pipelines\base.py:1140, in Pipeline.__call__(self, inputs, num_workers, batch_size, *args, **kwargs)
   1132     return next(
   1133         iter(
   1134             self.get_iterator(
   (...)
   1137         )
   1138     )
   1139 else:
-> 1140     return self.run_single(inputs, preprocess_params, forward_params, postprocess_params)

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\pipelines\base.py:1147, in Pipeline.run_single(self, inputs, preprocess_params, forward_params, postprocess_params)
   1145 def run_single(self, inputs, preprocess_params, forward_params, postprocess_params):
   1146     model_inputs = self.preprocess(inputs, **preprocess_params)
-> 1147     model_outputs = self.forward(model_inputs, **forward_params)
   1148     outputs = self.postprocess(model_outputs, **postprocess_params)
   1149     return outputs

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\pipelines\base.py:1046, in Pipeline.forward(self, model_inputs, **forward_params)
   1044     with inference_context():
   1045         model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
-> 1046         model_outputs = self._forward(model_inputs, **forward_params)
   1047         model_outputs = self._ensure_tensor_on_device(model_outputs, device=torch.device("cpu"))
   1048 else:

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\pipelines\text_generation.py:271, in TextGenerationPipeline._forward(self, model_inputs, **generate_kwargs)
    268         generate_kwargs["min_length"] += prefix_length
    270 # BS x SL
--> 271 generated_sequence = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
    272 out_b = generated_sequence.shape[0]
    273 if self.framework == "pt":

File ~\anaconda3\envs\rag\Lib\site-packages\torch\utils\_contextlib.py:115, in context_decorator.<locals>.decorate_context(*args, **kwargs)
    112 @functools.wraps(func)
    113 def decorate_context(*args, **kwargs):
    114     with ctx_factory():
--> 115         return func(*args, **kwargs)

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\generation\utils.py:1718, in GenerationMixin.generate(self, inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, **kwargs)
   1701     return self.assisted_decoding(
   1702         input_ids,
   1703         assistant_model=assistant_model,
   (...)
   1714         **model_kwargs,
   1715     )
   1716 if generation_mode == GenerationMode.GREEDY_SEARCH:
   1717     # 11. run greedy search
-> 1718     return self.greedy_search(
   1719         input_ids,
   1720         logits_processor=logits_processor,
   1721         stopping_criteria=stopping_criteria,
   1722         pad_token_id=generation_config.pad_token_id,
   1723         eos_token_id=generation_config.eos_token_id,
   1724         output_scores=generation_config.output_scores,
   1725         return_dict_in_generate=generation_config.return_dict_in_generate,
   1726         synced_gpus=synced_gpus,
   1727         streamer=streamer,
   1728         **model_kwargs,
   1729     )
   1731 elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
   1732     if not model_kwargs["use_cache"]:

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\generation\utils.py:2579, in GenerationMixin.greedy_search(self, input_ids, logits_processor, stopping_criteria, max_length, pad_token_id, eos_token_id, output_attentions, output_hidden_states, output_scores, return_dict_in_generate, synced_gpus, streamer, **model_kwargs)
   2576 model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
   2578 # forward pass to get next token
-> 2579 outputs = self(
   2580     **model_inputs,
   2581     return_dict=True,
   2582     output_attentions=output_attentions,
   2583     output_hidden_states=output_hidden_states,
   2584 )
   2586 if synced_gpus and this_peer_finished:
   2587     continue  # don't waste resources running the code we don't need

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
   1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1517 else:
-> 1518     return self._call_impl(*args, **kwargs)

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1527, in Module._call_impl(self, *args, **kwargs)
   1522 # If we don't have any hooks, we want to skip the rest of the logic in
   1523 # this function, and just call forward.
   1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1525         or _global_backward_pre_hooks or _global_backward_hooks
   1526         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1527     return forward_call(*args, **kwargs)
   1529 try:
   1530     result = None

File ~\anaconda3\envs\rag\Lib\site-packages\accelerate\hooks.py:165, in add_hook_to_module.<locals>.new_forward(module, *args, **kwargs)
    163         output = module._old_forward(*args, **kwargs)
    164 else:
--> 165     output = module._old_forward(*args, **kwargs)
    166 return module._hf_hook.post_forward(module, output)

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\models\llama\modeling_llama.py:1181, in LlamaForCausalLM.forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
   1178 return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1180 # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
-> 1181 outputs = self.model(
   1182     input_ids=input_ids,
   1183     attention_mask=attention_mask,
   1184     position_ids=position_ids,
   1185     past_key_values=past_key_values,
   1186     inputs_embeds=inputs_embeds,
   1187     use_cache=use_cache,
   1188     output_attentions=output_attentions,
   1189     output_hidden_states=output_hidden_states,
   1190     return_dict=return_dict,
   1191 )
   1193 hidden_states = outputs[0]
   1194 if self.config.pretraining_tp > 1:

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
   1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1517 else:
-> 1518     return self._call_impl(*args, **kwargs)

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1527, in Module._call_impl(self, *args, **kwargs)
   1522 # If we don't have any hooks, we want to skip the rest of the logic in
   1523 # this function, and just call forward.
   1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1525         or _global_backward_pre_hooks or _global_backward_hooks
   1526         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1527     return forward_call(*args, **kwargs)
   1529 try:
   1530     result = None

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\models\llama\modeling_llama.py:1068, in LlamaModel.forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict)
   1058     layer_outputs = self._gradient_checkpointing_func(
   1059         decoder_layer.__call__,
   1060         hidden_states,
   (...)
   1065         use_cache,
   1066     )
   1067 else:
-> 1068     layer_outputs = decoder_layer(
   1069         hidden_states,
   1070         attention_mask=attention_mask,
   1071         position_ids=position_ids,
   1072         past_key_value=past_key_values,
   1073         output_attentions=output_attentions,
   1074         use_cache=use_cache,
   1075     )
   1077 hidden_states = layer_outputs[0]
   1079 if use_cache:

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
   1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1517 else:
-> 1518     return self._call_impl(*args, **kwargs)

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1527, in Module._call_impl(self, *args, **kwargs)
   1522 # If we don't have any hooks, we want to skip the rest of the logic in
   1523 # this function, and just call forward.
   1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1525         or _global_backward_pre_hooks or _global_backward_hooks
   1526         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1527     return forward_call(*args, **kwargs)
   1529 try:
   1530     result = None

File ~\anaconda3\envs\rag\Lib\site-packages\accelerate\hooks.py:165, in add_hook_to_module.<locals>.new_forward(module, *args, **kwargs)
    163         output = module._old_forward(*args, **kwargs)
    164 else:
--> 165     output = module._old_forward(*args, **kwargs)
    166 return module._hf_hook.post_forward(module, output)

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\models\llama\modeling_llama.py:796, in LlamaDecoderLayer.forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
    793 hidden_states = self.input_layernorm(hidden_states)
    795 # Self Attention
--> 796 hidden_states, self_attn_weights, present_key_value = self.self_attn(
    797     hidden_states=hidden_states,
    798     attention_mask=attention_mask,
    799     position_ids=position_ids,
    800     past_key_value=past_key_value,
    801     output_attentions=output_attentions,
    802     use_cache=use_cache,
    803     **kwargs,
    804 )
    805 hidden_states = residual + hidden_states
    807 # Fully Connected

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1518, in Module._wrapped_call_impl(self, *args, **kwargs)
   1516     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1517 else:
-> 1518     return self._call_impl(*args, **kwargs)

File ~\anaconda3\envs\rag\Lib\site-packages\torch\nn\modules\module.py:1527, in Module._call_impl(self, *args, **kwargs)
   1522 # If we don't have any hooks, we want to skip the rest of the logic in
   1523 # this function, and just call forward.
   1524 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1525         or _global_backward_pre_hooks or _global_backward_hooks
   1526         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1527     return forward_call(*args, **kwargs)
   1529 try:
   1530     result = None

File ~\anaconda3\envs\rag\Lib\site-packages\accelerate\hooks.py:165, in add_hook_to_module.<locals>.new_forward(module, *args, **kwargs)
    163         output = module._old_forward(*args, **kwargs)
    164 else:
--> 165     output = module._old_forward(*args, **kwargs)
    166 return module._hf_hook.post_forward(module, output)

File ~\anaconda3\envs\rag\Lib\site-packages\transformers\models\llama\modeling_llama.py:726, in LlamaSdpaAttention.forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)
    723     key_states = key_states.contiguous()
    724     value_states = value_states.contiguous()
--> 726 attn_output = torch.nn.functional.scaled_dot_product_attention(
    727     query_states,
    728     key_states,
    729     value_states,
    730     attn_mask=attention_mask,
    731     dropout_p=self.attention_dropout if self.training else 0.0,
    732     # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    733     is_causal=self.is_causal and attention_mask is None and q_len > 1,
    734 )
    736 attn_output = attn_output.transpose(1, 2).contiguous()
    737 attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

RuntimeError: cutlassF: no kernel found to launch!