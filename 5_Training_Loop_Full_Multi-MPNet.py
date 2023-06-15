# 5/23/23: Code compiled and modified by Seth Kyler
# Based off code by CDR Milton Mendieta, Ecuadorian Navy:
# https://github.com/m2im/violence_prediction/tree/main/Scripts
# 6/7/23: Prediction threshold added by T.C. Warren, Naval Postgraduate School

from datasets import load_from_disk, Value
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler
from transformers import AutoModelForSequenceClassification, AdamW
from accelerate import Accelerator
from argparse import Namespace
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import roc_curve
import evaluate
import time
import torch
import wandb
import os
import numpy as np
import pandas as pd

# empty unused memory before each run
torch.cuda.empty_cache()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login()
run = wandb.init(project="TrainingLoop_v20")

# Data file name
FN_DATA = '/data2/skyler2/articles_dataset_ru_v3'

# Load the dataset
dataset = load_from_disk(FN_DATA)

# Convert label column to float, necessary for calculating loss
new_features = dataset['train'].features.copy()
new_features['match70'] = Value('float32')
dataset = dataset.cast(new_features)
dataset['train'].features

# Header with hyperparameters and other variables
config = {
    'model_ckpt': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'batch_size': 32,
    'num_labels': 1,
    'init_lr': 5e-5,
    'num_epochs': 1,
    'num_warmup_steps': 0,
    'lr_scheduler_type': 'constant',
    'weight_decay': 0.1,
    'seed': 42
}

args = Namespace(**config)

accelerator = Accelerator()

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)


# Tokenizing the whole dataset
def tokenize_function(examples):
    return tokenizer(examples['Text'], padding=True, truncation=True)


print('Max length of tokenizer:', tokenizer.model_max_length)

# Time tokenization of dataset
start_time_tokenize = time.time()

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print('\nTokenizing Time elapsed:')
print(time.strftime(
    '%H hr : %M min : %S sec',
    time.gmtime(time.time()-start_time_tokenize)))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare for training by removing all columns except "input_ids",
# which is the column on which the model will train,
# "match70", which is the column that will be used as labels,
# "token_type_ids", and "attention_mask"
# "match70" must be changed to "labels"
# because that is the name the model expects
tokenized_datasets = tokenized_datasets.remove_columns(
    ['GlobalEventID', 'SOURCEURL', 'Title', 'Text', 'match50', 'match30'])
tokenized_datasets = tokenized_datasets.rename_column('match70', 'labels')

# Define dataloaders used for transfering data to GPU
train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True,
                              batch_size=args.batch_size,
                              collate_fn=data_collator)
val_dataloader = DataLoader(tokenized_datasets['validation'],
                            batch_size=args.batch_size,
                            collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets['test'],
                             batch_size=args.batch_size,
                             collate_fn=data_collator)

# Instantiate the model
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_ckpt, num_labels=args.num_labels)

# Inspect a batch to check that there are no mistakes
# for batch in train_dataloader:
#     break
# {k: v.shape for k, v in batch.items()}

# A test to make sure everything is working properly
# outputs = model(**batch)
# print('\nLoss:', outputs.loss, '\nLogits:', outputs.logits,
#       '\nLogits Shape:', outputs.logits.shape)

# Defining the optimizer
optimizer = AdamW(model.parameters(), lr=args.init_lr)

# Define the learning rate scheduler
num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=num_training_steps
)
print('Number of training steps:', num_training_steps)

# Prepare components for use on GPUs
train_dataloader, val_dataloader, test_dataloader, model, optimizer, lr_scheduler =\
    accelerator.prepare(train_dataloader, val_dataloader, test_dataloader,
                        model, optimizer, lr_scheduler)

# Training loop
print('Training...')
progress_bar = tqdm(range(num_training_steps))

wandb.init(settings=wandb.Settings(start_method="fork"))
wandb.init()

# Load required metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
roc_auc_metric = evaluate.load("roc_auc")

# Training time of dataset
start_time_training = time.time()

training_loss = 0.0
train_batches = 0
validation_loss = 0.0
val_batches = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        model.train()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        training_loss += loss.item()
        train_batches += 1

        # Log training loss
        if (train_batches + 1) % 100 == 0:
            training_loss /= train_batches
            wandb.log({'Training Loss': training_loss})

            # Evaluate the model on the validation set
            model.eval()
            for batch in val_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    validation_loss += loss.item()
                    val_batches += 1

            # Log validation loss
            validation_loss /= val_batches
            wandb.log({"Validation Loss": validation_loss})

            training_loss = 0.0
            train_batches = 0
            validation_loss = 0.0
            val_batches = 0


print("\nTraining Time elapsed:")
print(time.strftime(
    "%H hr : %M min : %S sec",
    time.gmtime(time.time()-start_time_training)))


# Evaluation loop
print('Finding optimal threshold...')
start_time_eval = time.time()

# Calculate optimal threshold using validation data
probs = []
labs = []
cnt = 0
tot = len(val_dataloader)
model.eval()
for batch in val_dataloader:
    cnt += 1
    # print('Batch', cnt, 'of', tot)
    with torch.no_grad():
        logits = model(**batch).logits
    prob = torch.sigmoid(logits).squeeze()
    probs += prob.tolist()
    labs += batch['labels'].tolist()

fpr, tpr, thresholds = roc_curve(labs, probs, drop_intermediate=False)
# opt_thresh = thresholds[np.argmin(np.abs(fpr + tpr - 1))]
opt_thresh = thresholds[np.argmax(tpr - fpr)]
# opt_thresh = thresholds[np.argmax(np.sqrt(tpr * (1 - fpr)))]

# Calculate final accuracy metrics using test data
print('Conducting final tests ...')
probs = []
preds = []
labs = []
cnt = 0
tot = len(test_dataloader)
model.eval()
for batch in test_dataloader:
    cnt += 1
    # print('Batch', cnt, 'of', tot)
    with torch.no_grad():
        logits = model(**batch).logits
    prob = torch.sigmoid(logits).squeeze()
    pred = (prob >= opt_thresh).to(torch.int)
    lab = batch['labels']

    # Save results
    probs += prob.tolist()
    preds += pred.tolist()
    labs += lab.tolist()

    # Store results for metrics
    accuracy_metric.add_batch(predictions=pred, references=lab)
    precision_metric.add_batch(predictions=pred, references=lab)
    recall_metric.add_batch(predictions=pred, references=lab)
    f1_metric.add_batch(predictions=pred, references=lab)
    roc_auc_metric.add_batch(prediction_scores=prob, references=lab)

print("\nEvaluation Time elapsed:")
print(time.strftime(
    "%H hr : %M min : %S sec",
    time.gmtime(time.time()-start_time_eval)))

# Compute the metric values
accuracy = accuracy_metric.compute()
precision = precision_metric.compute()
recall = recall_metric.compute()
f1 = f1_metric.compute()
roc_auc = roc_auc_metric.compute()

# Log the final metrics
wandb.log({"Final Accuracy": accuracy})
wandb.log({"Final Precision": precision})
wandb.log({"Final Recall": recall})
wandb.log({"Final F1": f1})
wandb.log({"Final ROC_AUC": roc_auc})

print('Threshold:', opt_thresh)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1:', f1)
print('ROC_AUC:', roc_auc)

# Saves output values to CSV
DF_OUT = '/home/skyler/Accuracy_Logs/'
FN_OUT_ACC = 'accuracy_v37_{}.csv'
FN_OUT_PRED = 'predict_v37_{}.csv'

# Create output file names
time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
fn_pred = FN_OUT_PRED.format(time_str)
fn_pred = os.path.join(DF_OUT, fn_pred)
fn_acc = FN_OUT_ACC.format(time_str)
fn_acc = os.path.join(DF_OUT, fn_acc)

# Write values to CSV
print("Writing file to csv:\n{}".format(fn_pred))
df = pd.DataFrame({'probs': probs, 'preds': preds, 'labs': labs})
df.to_csv(fn_pred, index=False)

print("Writing file to csv:\n{}".format(fn_acc))
acc = {'opt_thresh': opt_thresh}
acc.update(accuracy)
acc.update(precision)
acc.update(recall)
acc.update(f1)
acc.update(roc_auc)
df = pd.DataFrame([acc])
df.to_csv(fn_acc)