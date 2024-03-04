from dataclasses import dataclass, field
from typing import Optional
import re

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)


def make_format_func(model_id):
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  def formatting_func(row):
    text = row['text']
    text = text.replace('<|user|>', '')
    text = text.replace('<|bot|>', '')
    pattern = re.compile(
        r'(<sys>|<usr>|<bot>)(.*?)(?=<sys>|<usr>|<bot>|$)', re.DOTALL)
    matches = pattern.findall(text)
    chat = []
    roles = {"<sys>": "assistant", "<usr>": "user", "<bot>": "assistant"}
    for token, content in matches:
      chat.append({"role": roles[token], "content": content.strip()})
    if (len(chat) == 0):
      return {"text": ""}
    if (len(chat) == 1):
      return {"text": ''}
    if (chat[0]['role'] == 'assistant'):
      chat[1]['content'] = f"{chat[0]['content']}\n{chat[1]['content']}"
      chat = chat[1:]
    try:
      prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    except Exception:
      prompt = ""
    return {"text": prompt}
  return formatting_func


@dataclass
class ScriptArguments:
  """
  These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
  """
  gradient_accumulation_steps: Optional[int] = field(default=4)
  learning_rate: Optional[float] = field(default=2e-4)
  max_grad_norm: Optional[float] = field(default=0.3)
  weight_decay: Optional[int] = field(default=0.001)
  lora_alpha: Optional[int] = field(default=16)
  lora_dropout: Optional[float] = field(default=0.1)
  lora_r: Optional[int] = field(default=8)
  max_seq_length: Optional[int] = field(default=1024)
  model_name: Optional[str] = field(
      default=None,
      metadata={
          "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
      }
  )
  dataset_name: Optional[str] = field(
      default="heegyu/open-korean-instructions",
      metadata={"help": "The preference dataset to use."},
  )
  fp16: Optional[bool] = field(
      default=True,
      metadata={"help": "Enables fp16 training."},
  )
  bf16: Optional[bool] = field(
      default=False,
      metadata={"help": "Enables bf16 training."},
  )
  packing: Optional[bool] = field(
      default=True,
      metadata={"help": "Use packing dataset creating."},
  )
  gradient_checkpointing: Optional[bool] = field(
      default=True,
      metadata={"help": "Enables gradient checkpointing."},
  )
  use_flash_attention_2: Optional[bool] = field(
      default=False,
      metadata={"help": "Enables Flash Attention 2."},
  )
  optim: Optional[str] = field(
      default="paged_adamw_32bit",
      metadata={"help": "The optimizer to use."},
  )
  lr_scheduler_type: str = field(
      default="constant",
      metadata={
          "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
  )
  max_steps: int = field(default=100000, metadata={
                         "help": "How many optimizer update steps to take"})
  warmup_ratio: float = field(default=0.03, metadata={
                              "help": "Fraction of steps to do a warmup for"})
  save_steps: int = field(default=100, metadata={
                          "help": "Save checkpoint every X updates steps."})
  logging_steps: int = field(
      default=10, metadata={"help": "Log every X updates steps."})
