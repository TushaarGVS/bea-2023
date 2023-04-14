from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
import os
import pickle
import numpy as np
from tqdm.auto import tqdm
import scipy.stats


parser = argparse.ArgumentParser()
parser.add_argument('--model', 
  type=str, 
  choices=[
    "microsoft/DialogRPT-updown",
    "microsoft/DialogRPT-human-vs-rand",
    "microsoft/DialogRPT-human-vs-machine"
  ],
  default="microsoft/DialogRPT-updown"
)
args = parser.parse_args()

model = args.model  # you can try other model_card listed in the table above
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)
model.eval()

@torch.no_grad()
def score(ctx, hyp):
  model_input = tokenizer.encode(ctx + hyp, return_tensors="pt")
  result = model(model_input, return_dict=True)
  return torch.sigmoid(result.logits)

with open(f'data{os.sep}train_with-reference{os.sep}dialogue-RPT.pickle', 'rb') as f:
  dialogues = pickle.load(f)

scores = []
for ctx, response in tqdm(dialogues):
  scores.append(score(ctx, response))

scores = torch.as_tensor(scores).cpu().numpy()
print(scipy.stats.describe(scores))
np.save(f'data{os.sep}train_with-reference{os.sep}dialogue-RPT-{args.model.replace(os.sep, "-")}.npy', scores)
