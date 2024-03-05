from datasets import load_dataset, Dataset, concatenate_datasets

comment_dataset = [
  {'id': 'kor_hate', 'text_field': 'comments', 'train': ['train'], 'eval': ['test']},
  {'id': 'jeanlee/kmhas_korean_hate_speech', 'text_field': 'text', 'train': ['train', 'test'], 'eval': ['validation']},
  {'id': 'SJ-Donald/kor-hate-sentence', 'text_field': '문장', 'train': ['train'], 'eval': ['validation']}
]

wiki_dataset = [
  {'id': 'heegyu/namuwiki-extracted', 'text_field': 'text', 'train': ['train'], 'eval': ['test'], 'ratio': 0.0001}
]

def get_dataset(target: str):
  # Load the dataset and format it for training.
  train_ds = Dataset.from_list([])
  eval_ds = Dataset.from_list([])
  if target == 'comment': dataset_infos = comment_dataset
  if target == 'wiki': dataset_infos = wiki_dataset

  for dataset_info in dataset_infos:
    def map_row(row):
      text = row[dataset_info['text_field']]
      if text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
      newrow = {'text': text}
      return newrow
    formated = load_dataset(dataset_info['id']).map(map_row)
    if 'ratio' in dataset_info:
      formated = formated['train'].train_test_split(dataset_info['ratio'])
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
  return train_ds, eval_ds
