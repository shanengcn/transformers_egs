# -*-coding:utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from datasets import ClassLabel, load_metric
from tqdm.auto import tqdm
from transformers import get_scheduler, AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding


raw_datasets = load_dataset('./dataset/examples')
model_path = '../model/bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize_function(example):
    return tokenizer(example['sentence'], max_length=32, truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(
    ['labelid', 'labelname', 'sentence'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

print(tokenized_datasets['train'].column_names)
train_dataloader = DataLoader(
    tokenized_datasets['train'], shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(
    tokenized_datasets['validation'], batch_size=32, collate_fn=data_collator)
test_dataloader = DataLoader(
    tokenized_datasets['test'], batch_size=16, collate_fn=data_collator)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
print(num_training_steps)
lr_scheduler = get_scheduler(
    'Linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)


def compute_metrics(model, eval_dataloader, device):
    metric = load_metric('accuracy')
    model.eval()
    loss = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k,  v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        loss.append(outputs, loss)
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
    loss = np.mean(np.array(loss))
    print('loss:', loss)
    print(metric.compute())


progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    print('epoch:', epoch)
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        compute_metrics(model, eval_dataloader, device)
        compute_metrics(model, test_dataloader, device)
model.save_pretrained('./checkpoint')
