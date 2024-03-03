from dataclasses import dataclass, field
from typing import Optional
import re
import os

import torch
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=2)
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
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=100000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


def formatting_func(row):
    text = row['text']
    text = text.replace('<|user|>', '')
    text = text.replace('<|bot|>', '')
    pattern = re.compile(r'(<sys>|<usr>|<bot>)(.*?)(?=<sys>|<usr>|<bot>|$)', re.DOTALL)
    matches = pattern.findall(text)
    chat = []
    roles = {"<sys>": "assistant", "<usr>": "user", "<bot>": "assistant"}
    for token, content in matches:
      chat.append({"role": roles[token], "content": content.strip()})
    if (len(chat) == 0): return {"text": ""}
    if (len(chat) == 1): return {"text": ''}
    if (chat[0]['role'] == 'assistant'):
      chat[1]['content'] = f"{chat[0]['content']}\n{chat[1]['content']}"
      chat = chat[1:]
    try:
      prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    except Exception:
      prompt = ""
    return {"text": prompt}

# Load the GG model - this is the local one, update it to the one on the Hub
model_id = "./hannam-2b/checkpoint-27000"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=quantization_config, 
    torch_dtype=torch.float16,
    attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=script_args.lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout
)

train_dataset = load_dataset(script_args.dataset_name, split="train")
train_dataset = train_dataset.map(
    formatting_func,
    num_proc=os.cpu_count() // 2,
)
train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 0)
print(train_dataset[0])


YOUR_HF_USERNAME = 'seonglae'
output_dir = f"{YOUR_HF_USERNAME}/hannam-chat"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    lr_scheduler_type=script_args.lr_scheduler_type,
    gradient_checkpointing=script_args.gradient_checkpointing,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    packing=script_args.packing,
    tokenizer=tokenizer,
    max_seq_length=script_args.max_seq_length,
)

trainer.train()
trainer.save_model(output_dir)
model.save_pretrained('hannam-2b')
tokenizer.save_pretrained('hannam-2b')
tokenizer.save_vocabulary('hannam-2b')
