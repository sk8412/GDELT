from datasets import load_from_disk, Value
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler
from transformers import AutoModelForSequenceClassification, AdamW
from accelerate import Accelerator
from argparse import Namespace
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate, time, torch, wandb
import os

# empty unused memory before each run
torch.cuda.empty_cache()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#wandb.login()

# Data file name
FN_DATA = 'articles_dataset'

# Load the dataset
dataset = load_from_disk(FN_DATA)

# Convert label column to float, necessary for calculating loss
new_features = dataset["train"].features.copy()
new_features["match30"] = Value("float32")
dataset = dataset.cast(new_features)
dataset["train"].features

# Header with hyperparameters and other variables
config = {
    "model_ckpt": "setu4993/smaller-LaBSE", # "roberta-base",
    "batch_size": 32,
    "num_labels": 1,
    "init_lr": 5e-5,
    "num_epochs": 5,
    "num_warmup_steps": 0,
    "lr_scheduler_type": "linear", # cosine
    "weight_decay": 0.1,
    "seed": 42
}

args = Namespace(**config)

accelerator = Accelerator()

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)

# Tokenizing the whole dataset
def tokenize_function(examples):
    return tokenizer(examples["Text"], padding=True, truncation=True)

print("Max length of tokenizer:", tokenizer.model_max_length)

# Time tokenization of dataset
start_time_tokenize = time.time()

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("\nTokenizing Time elapsed:")
print(time.strftime(
    "%H hr : %M min : %S sec",
    time.gmtime(time.time()-start_time_tokenize)))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Prepare for training by removing all columns except "input_ids", which is the column on which the model will train,
# "match30", which is the column that will be used as labels, "token_type_ids", and "attention_mask"
# "match30" must be changed to "labels" because that is the name the model expects
tokenized_datasets = tokenized_datasets.remove_columns(["GlobalEventID", "SOURCEURL", "Title", "Text", "match50", "match70"])
tokenized_datasets = tokenized_datasets.rename_column("match30", "labels")

# Define dataloaders used for transfering data to GPU
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=args.batch_size, collate_fn=data_collator)

# Instantiate the model
model = AutoModelForSequenceClassification.from_pretrained(args.model_ckpt, num_labels=args.num_labels)

# Inspect a batch to check that there are no mistakes
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# A test to make sure everything is working properly
outputs = model(**batch)
print("\nLoss:", outputs.loss, "\nLogits:", outputs.logits, "\nLogits Shape:", outputs.logits.shape)

# Defining the optimizer
optimizer = AdamW(model.parameters(), lr=args.init_lr)

# Define the learning rate scheduler
num_epochs = args.num_epochs
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name = args.lr_scheduler_type,
    optimizer = optimizer,
    num_warmup_steps = args.num_warmup_steps,
    num_training_steps = num_training_steps
)
print("Number of training steps:", num_training_steps)

# Prepare components for use on GPUs
print("Preparing the components for use on GPUs. This may take a couple hours...")
train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer, lr_scheduler)

# Training loop
progress_bar = tqdm(range(num_training_steps))

#wandb.init(settings=wandb.Settings(start_method="fork"))
#wandb.init()

# Load required metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

# Training time of dataset
start_time_training = time.time()
step = 0

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        #wandb.log({'epoch': epoch, 'train_loss': loss.item()})

    # Evaluate the model on the validation set
    if accelerator.is_main_process and ((step + 1) % (num_training_steps // num_epochs) == 0 or (step + 1) == num_training_steps):
        evaluate.eval(model, eval_dataloader, tokenizer, accuracy_metric, precision_metric, recall_metric, f1_metric)

print("\nTraining Time elapsed:")
print(time.strftime(
    "%H hr : %M min : %S sec",
    time.gmtime(time.time()-start_time_training)))

# Evaluation loop
start_time_eval = time.time()

model.eval()
for batch in eval_dataloader:
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    accuracy_metric.add_batch(predictions=accelerator.gather(predictions), references=batch["labels"])
    precision_metric.add_batch(predictions=accelerator.gather(predictions), references=batch["labels"])
    recall_metric.add_batch(predictions=accelerator.gather(predictions), references=batch["labels"])
    f1_metric.add_batch(predictions=accelerator.gather(predictions), references=batch["labels"])
    
print("\nEvaluation Time elapsed:")
print(time.strftime(
    "%H hr : %M min : %S sec",
    time.gmtime(time.time()-start_time_eval)))

print("Accuracy:", accuracy_metric.compute())
print("Precision:", precision_metric.compute())
print("Recall:", recall_metric.compute())
print("F1:", f1_metric.compute())