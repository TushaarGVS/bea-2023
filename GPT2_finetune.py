from transformers import AutoTokenizer, default_data_collator, GPT2LMHeadModel
from datasets import Dataset
import os
import torch
import torch.utils.data.dataloader
from tqdm.auto import tqdm
import torch.optim
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', 
  type=str, 
  choices=[
    "microsoft/DialoGPT-small",
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-large"
  ],
  default="microsoft/DialoGPT-small"
)
args = parser.parse_args()


# Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = GPT2LMHeadModel.from_pretrained(args.model).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

eos_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

trainset_indices = torch.load(
    f'data{os.sep}train_with-reference{os.sep}train-dialoGPT-compatible-tokens-trainset-indices.pt', 
    map_location=torch.device('cuda')
)
valset_indices = torch.load(
    f'data{os.sep}train_with-reference{os.sep}train-dialoGPT-compatible-tokens-valset-indices.pt', 
    map_location=torch.device('cuda')
)
raw_dataset = torch.load(
    f'data{os.sep}train_with-reference{os.sep}train-dialoGPT-compatible-tokens.pt', 
    map_location=torch.device('cuda')
)


class BEADataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, indices) -> None:
        self.dataset = raw_dataset
        self.indices = indices

    def __getitem__(self, i):
        ret = { k: v[self.indices[i]] for k, v in self.dataset.items() }
        ret['labels'] = ret['input_ids'].clone()
        return ret
        # context = self.train_dataset['utterances'][i]
        # response = self.train_dataset['response'][i]
        # dialogue = (' ' + tokenizer.eos_token +
        #             ' ').join(d['text'] for d in context) + ' ' + tokenizer.eos_token + ' ' + response['text']
        # return dialogue

    def __len__(self):
        return len(self.indices)

# dataset = BEADataset()
# # train_dataset = Dataset.from_json(f'data{os.sep}raw{os.sep}train_with-reference.jsonl')
# # dialogues = []
# # for i in tqdm(range(len(train_dataset))):
# #     dialogues.append(dataset[i])

# dialogues = torch.load(f'data{os.sep}train_with-reference{os.sep}train-dialoGPT-compatible-strings.pt')


train_dataset = BEADataset(raw_dataset, trainset_indices)
val_dataset = BEADataset(raw_dataset, valset_indices)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, collate_fn=default_data_collator)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=default_data_collator)
counter = tqdm(range(10 * len(train_loader)))

for _ in range(10):
    train_loss = 0
    for i, data_dict in enumerate(train_loader):
        outputs = model(**data_dict)
        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()
        if i % 2 == 0 and i > 0:
            optimizer.step()
            optimizer.zero_grad()
        else:
            for p in model.parameters():
                p.grad.div_(2)
        
        if i % 100 and i > 0:
            print(f'train loss: {train_loss:.3f}')
            train_loss = 0
        counter.update(1)

    with torch.no_grad():
        total_val_loss = 0
        for data_dict in val_loader:
            outputs = model(**data_dict)
            total_val_loss += (len(data_dict['input_ids'])) * outputs.loss
        total_val_loss /= len(val_dataset)
        print(f'validation ppl: {torch.exp(total_val_loss).item():.3f}')

torch.save(model, f'model{os.sep}{args.model.replace(os.sep, "-")}-finetuned.pt')