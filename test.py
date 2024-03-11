"""
python test.py generate --device cuda:0
python test.py ppl --device cuda:0
"""
import logging
import platform

import torch
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from peft import PeftModel

default = 'cuda'
if platform.system() == 'Darwin': default = 'mps'

class YokhalTester:
  model: PreTrainedModel
  tokenizer: AutoTokenizer

  def load(self, model_id, device=None):
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.model: AutoModelForCausalLM
    try:
      import flash_attn
      if flash_attn:
        flash = "flash_attention_2"
    except ModuleNotFoundError:
      flash = "sdpa" if hasattr(
          torch.nn.functional, 'scaled_dot_product_attention') else None
    try:
      dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except AssertionError:
      dtype = torch.float16

    try:
      self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                   torch_dtype=dtype,
                                                   device_map="auto" if device is None else device,
                                                   attn_implementation=flash)
    except Exception as e:
      logging.error(e)
      logging.warn('Fallback into PEFT model loading...')
      self.model = AutoModelForCausalLM.from_pretrained('google/gemma-2b',
                                                   torch_dtype=torch.bfloat16,
                                                   device_map="auto" if device is None else device,
                                                   attn_implementation=flash)
      self.model = PeftModel.load(self.model, model_id)
      self.model.merge_and_unload()

  def generate(self, device=None, model_id='seonglae/yokhal-md', push=None, tag=None):
    """ Generate sampled tokens from the model with inference
    Args:
        device (str, optional): torch device. Defaults to None.
        model_id (str, optional): local path or model id. Defaults to 'seonglae/yokhal-md'.
        push (str, optional): If you want to push, pass id string to push like `seonglae/yokhal-md`. Defaults to None.
        tag (str, optional): If you want to push with tag revision to hub repository. Defaults to None.
    """
    self.load(model_id, device)
    sys_prompt = 'You are offensive AI who responses to user message. 한국어로 대답해'
    texts = ['안녕', '서울은 오늘 어때']

    # Inference
    chats = list(
        map(lambda t: [{'role': 'user', 'content': f'{sys_prompt}\n{t}'}], texts))
    prompts = list(map(lambda p: self.tokenizer.apply_chat_template(
        p, tokenize=False, add_generation_prompt=True), chats))
    input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
        default if device is None else device)
    outputs = self.model.generate(
        **input_ids, max_new_tokens=100, repetition_penalty=1.0)
    for output in outputs:
      print(self.tokenizer.decode(
          output, skip_special_tokens=True), end='\n\n')
    if push:
      self.push()

  def ppl(self, device=None, model_id='seonglae/yokhal-md', ds='kor_hate', split='test', col='comments'):
    """calculate inference perplexity

    Args:
        device (str, optional): torch device. Defaults to None.
        model_id (str, optional): local path or model id. Defaults to 'seonglae/yokhal-md'.
        ds (str, optional): Dataset id. Defaults to 'kor_hate'.
        split (str, optional): Dataset split. Defaults to 'test'.
        col (str, optional): Text column of the dataset. Defaults to 'comments'.
    """
    self.load(model_id, device)
    from evaluate import load 
    from datasets import load_dataset

    perplexity = load("yokhal/perplexity.py", module_type="metric")
    texts = load_dataset(ds, split=split)[col]
    self.load(model_id)
    results = perplexity.compute(model=self.model, tokenizer=self.tokenizer,
                                 add_start_token=True,
                                 predictions=texts,
                                 device=default)
    print(f'Mean Perplexity: {round(results["mean_perplexity"], 2)}')

  def push(self, push_to, message, tag=None):
    self.model.name_or_path = push_to
    self.model.push_to_hub(push_to, commit_message=message,
                           revision='main' if tag is None else tag)
    self.tokenizer.push_to_hub(push_to, commit_message=message)


if __name__ == '__main__':
  fire.Fire(YokhalTester)
