from transformers import AutoTokenizer, default_data_collator, GPT2LMHeadModel
from datasets import Dataset
import os
import torch
import torch.utils.data.dataloader
from tqdm.auto import tqdm
import torch.optim
import numpy as np
import argparse
import warnings
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings('ignore')

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
model = torch.load(f'model{os.sep}{args.model.replace(os.sep, "-")}-finetuned.pt', map_location=torch.device('cuda'))
dialogues = torch.load(f'data{os.sep}train_with-reference{os.sep}train-dialoGPT-compatible-strings.pt')

tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

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
        idx = self.indices[i]
        context = self.dataset['utterances'][idx]
        response = self.dataset['response'][idx]
        context = (' ' + tokenizer.eos_token + ' ').join(d['text'] for d in context) + tokenizer.eos_token 
        context_tokenized = tokenizer(context, padding='do_not_pad', return_tensors='pt')
        combined_tokenized = tokenizer(context + ' ' + response['text'], padding='do_not_pad', return_tensors='pt')
        combined_tokenized['labels'] = combined_tokenized['input_ids'].clone()
        return context_tokenized, combined_tokenized, context, response['text']

    def __len__(self):
        return len(self.indices)


train_dataset = Dataset.from_json(
    f'data{os.sep}raw{os.sep}train_with-reference.jsonl')
val_dataset = BEADataset(train_dataset, valset_indices)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
counter = tqdm(range(len(val_dataset)))

generations = []
with torch.no_grad():
    model.eval()
    total_val_loss = 0
    normalization_factor = 0
    for i in range(len(val_dataset)):
        gen_dict, ppl_dict, context_text, response_text = val_dataset[i]
        input_ids = gen_dict['input_ids'].squeeze().cuda()
        attention_mask = gen_dict['attention_mask'].squeeze().cuda()
        input_length = len(input_ids)
        outputs = model.generate(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            max_length=300, 
            do_sample=True,
            top_p=0.9, 
            top_k=10,
            num_return_sequences=1, 
            early_stopping=True
        )[0]
        valid_gen_length = torch.nonzero(outputs[input_length:] == tokenizer.eos_token_id).item()
        valid_gen = outputs[input_length:input_length + valid_gen_length]
        valid_gen = tokenizer.decode(valid_gen).strip()
        generations.append((context_text, valid_gen, response_text))
        
        ce_loss = model(**{k: v.cuda() for k, v in ppl_dict.items()}).loss
        total_val_loss += ce_loss * ppl_dict['input_ids'].shape[1]
        normalization_factor += ppl_dict['input_ids'].shape[1]

        counter.update(1)

    print(f'validation ppl: {torch.exp(total_val_loss / normalization_factor).item():.3f}')

torch.save(generations,
           f'data{os.sep}train_with-reference{os.sep}{args.model.replace(os.sep, "-")}-generation-on-valset.pt')
