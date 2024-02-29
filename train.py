import torch
import fire

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from trl import SFTTrainer

dataset_infos = [
  {'id': 'kor_hate', 'text_field': 'comments', 'train': ['train'], 'eval': ['test']},
  {'id': 'jeanlee/kmhas_korean_hate_speech', 'text_field': 'text', 'train': ['train', 'test'], 'eval': ['validation']},
  {'id': 'SJ-Donald/kor-hate-sentence', 'text_field': '문장', 'train': ['train'], 'eval': ['validation']}
]

torch.manual_seed(0)
def prediction(base='google/gemma-2b-it', save_local=True, push=False,
          epoch=5, batch=3, output="./hannam-2b"):

  # Load the dataset and format it for training.
  train_ds = Dataset.from_list([])
  eval_ds = Dataset.from_list([])
  for dataset_info in dataset_infos:
    def map_row(row):
      text = row[dataset_info['text_field']]
      if text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
      newrow = {'text': text}
      return newrow
    formated = load_dataset(dataset_info['id']).map(map_row)
    for split in dataset_info['train']:
      for col in formated[split].column_names:
        if col != 'text':
          formated[split] = formated[split].remove_columns(col)
      train_ds = concatenate_datasets([formated[split], train_ds])
    for split in dataset_info['eval']:
      for col in formated[split].column_names:
        if col != 'text':
          formated[split] = formated[split].remove_columns(col)
      eval_ds = concatenate_datasets([formated[split], eval_ds])
  print(f"Train data: {len(train_ds)} Eval data: {len(eval_ds)}")
  
  # Load the pretrained model and tokenizer.
  max_seq_length = 1024
  tokenizer = AutoTokenizer.from_pretrained(base)
  model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(base,
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
        save_steps=100,
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
