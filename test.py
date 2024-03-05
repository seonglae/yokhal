import torch
import fire

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def test(device, model_id='hannam-md', push=None, tag=None):
  # Load the pretrained model and tokenizer.
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model: AutoModelForCausalLM
  try:
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto" if device is None else device, 
                                                 attn_implementation="flash_attention_2")
  except Exception:
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', 
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto" if device is None else device, 
                                                 attn_implementation="flash_attention_2")
    model = PeftModel.load(model, model_id)
    model.merge_and_unload()
  sys_prompt = 'You are offensive AI who responses to user message. 한국어로 대답해'
  texts = ['안녕', '서울은 오늘 어때']

  # Inference
  chats = list(map(lambda t: [{'role': 'user', 'content': f'{sys_prompt}\n{t}'}], texts))
  prompts = list(map(lambda p: tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True), chats))
  input_ids = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda" if device is None else device)
  outputs = model.generate(**input_ids, max_new_tokens=100, repetition_penalty=1.05)
  for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))
    print()

  if push:
    model.name_or_path = push
    model.push_to_hub(push, commit_message=model_id, revision='main' if tag is None else tag)
    tokenizer.push_to_hub(push, commit_message=model_id)

if __name__ == '__main__':
  fire.Fire(test)
