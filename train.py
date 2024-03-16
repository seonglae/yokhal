import os
import logging

import torch
import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, PreTrainedModel, HfArgumentParser
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

from yokhal.dataset import get_dataset
from yokhal.adapt import ScriptArguments, make_format_func, quantization_config


torch.manual_seed(0)


class yokhalTrainer():
  model: PreTrainedModel
  tokenizer: AutoTokenizer

  def finetune(self, base='seonglae/yokhal-md', save_local=True, push=False, resume=False,
               epoch=1, batch=3, output="./yokhal-md", target="wiki", max_length=1024, device=None,
               log_steps=10, save_steps=100, eval_steps=100, lr=1e-5, optim='adafactor'):

    # Load the dataset and format it for training.
    train_ds, eval_ds = get_dataset(target)
    logging.info(f"Train data: {len(train_ds)} Eval data: {len(eval_ds)}")

    # Load the pretrained model and tokenizer.
    flash = "sdpa" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None
    self.tokenizer = AutoTokenizer.from_pretrained(base)
    self.model = AutoModelForCausalLM.from_pretrained(base, attn_implementation=flash,
                                                      torch_dtype=torch.bfloat16,
                                                      device_map="auto" if device is None else device)
    logging.info(f'Special tokens: {self.tokenizer.all_special_tokens}')

    # Finally, set up the trainer and train the model.
    trainer = SFTTrainer(
        model=self.model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=TrainingArguments(
            per_device_train_batch_size=batch,
            per_device_eval_batch_size=batch,
            num_train_epochs=epoch,
            output_dir=output,
            optim=optim,
            logging_steps=log_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            learning_rate=lr,
            evaluation_strategy='steps'
        ),
        dataset_text_field="text",
        max_seq_length=max_length,
        packing=True,
    )
    trainer.train(resume_from_checkpoint=resume)
    self.push(output, save_local=save_local, push=push)

  def adapt(self, model_id='seonglae/yokhal-md', save_local=True, push=False,
            epoch=1, batch=3, output="./yokhal-md"):
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 quantization_config=quantization_config,
                                                 torch_dtype=torch.float16,
                                                 attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2"
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    lora_config = LoraConfig(
        r=script_args.lora_r,
        target_modules=["q_proj", "o_proj", "k_proj",
                        "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout
    )
    formatting_func = make_format_func(model_id)
    train_dataset = load_dataset(script_args.dataset_name, split="train")
    train_dataset = train_dataset.map(
        formatting_func,
        num_proc=os.cpu_count() // 2,
    )
    train_dataset = train_dataset.filter(lambda x: len(x["text"]) > 0)
    logging.info(train_dataset[0])

    training_arguments = TrainingArguments(
        output_dir=output,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        num_train_epochs=epoch,
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
    trainer.train(resume_from_checkpoint=True)
    self.push(model, tokenizer, output, save_local=save_local, push=push)

  def push(self, output, save_local, push):
    if save_local:
      self.model.save_pretrained(output)
      self.tokenizer.save_pretrained(output)
    if push:
      self.model.name_or_path = push
      self.model.push_to_hub(push)
      self.tokenizer.push_to_hub(push)


if __name__ == '__main__':
  fire.Fire(yokhalTrainer)
