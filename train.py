import torch
import fire

from hannam.dataset import get_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, PreTrainedModel

from trl import SFTTrainer


torch.manual_seed(0)
def finetune(base='seonglae/hannam-2b', save_local=True, push=False,
          epoch=2, batch=1, output="./hannam-2b", target="wiki"):

  # Load the dataset and format it for training.
  train_ds, eval_ds = get_dataset(target)
  print(f"Train data: {len(train_ds)} Eval data: {len(eval_ds)}")
  
  # Load the pretrained model and tokenizer.
  max_seq_length = 4096
  tokenizer = AutoTokenizer.from_pretrained(base)
  model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(base,
                                               torch_dtype=torch.bfloat16,
                                               device_map="auto")
  print(f'Special tokens: {tokenizer.all_special_tokens}')

  # Finally, set up the trainer and train the model.
  trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=TrainingArguments(
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epoch,
        output_dir=output,
        optim="adafactor",
        logging_steps=10,
        save_steps=1000,
        eval_steps=100,
        learning_rate=1e-5,
        evaluation_strategy='steps'
    ),
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
  )
  trainer.train()

  if save_local:
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    tokenizer.save_vocabulary(output)
  if push:
    model.push_to_hub(push)
    tokenizer.push_to_hub(push)


if __name__ == '__main__':
  fire.Fire()
