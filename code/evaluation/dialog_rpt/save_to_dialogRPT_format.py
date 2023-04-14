import os
from tqdm.auto import tqdm
import pickle
from datasets import Dataset

train_dataset = Dataset.from_json(f'data{os.sep}raw{os.sep}train_with-reference.jsonl')

dialogues = []
for i in tqdm(range(len(train_dataset))):
  context = ''
  for dialogue_line in train_dataset['utterances'][i]:
      context += "'" + dialogue_line['speaker'] + "': " + dialogue_line['text'] + ' <|endoftext|> '
  dialogue_line = train_dataset['response'][i]
  response = "'" + dialogue_line['speaker'] + "': " + dialogue_line['text'] + ' <|endoftext|> '  
  dialogues.append((context, response))

with open(f'data{os.sep}train_with-reference{os.sep}dialogue-RPT.pickle', 'wb') as f:
  pickle.dump(dialogues, f)