# 5/23/23: Code compiled and modified by Seth Kyler
# Based off code by CDR Milton Mendieta, Ecuadorian Navy:
# https://github.com/m2im/violence_prediction/tree/main/Scripts

import wandb, time, torch, gc
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_from_disk
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from accelerate import Accelerator

PRETRAINED_MODEL = 'sentence-transformers/LaBSE'

# Pretrained languag models
# sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# sentence-transformers/paraphrase-xlm-r-multilingual-v1
# sentence-transformers/LaBSE
# setu4993/smaller-LaBSE

torch.cuda.empty_cache()
gc.collect()

# Uncomment to record run in W&B
wandb.login()

run = wandb.init(
    project='TrainingLoop_SetFit_v2')


# Data file name
FN_DATA = '/data2/skyler2/articles_dataset_ru_v3'

# Load the dataset
dataset = load_from_disk(FN_DATA)

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Tokenizing the whole dataset
def tokenize_function(examples):
    return tokenizer(examples['Text'], padding=True, truncation=True)

print('Max length of tokenizer:', tokenizer.model_max_length)

# Mapping the tokenized dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Adding padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Remaining the columns because SetFit excepts 'text' and 'label'
tokenized_dataset = dataset.rename_column('Text', 'text')
tokenized_dataset = tokenized_dataset.rename_column('match70', 'label')
#print(tokenized_dataset['train'].features)

# Defining the datasets
train_dataset = sample_dataset(tokenized_dataset['train'], label_column='label', num_samples=8)
test_dataset = tokenized_dataset['test']

# Load SetFit model from Hub
model = SetFitModel.from_pretrained(PRETRAINED_MODEL)

# Initialize the accelerator
accelerator = Accelerator()

# Prepare the model and datasets
model, train_dataset, eval_dataset = accelerator.prepare(
    model, train_dataset, test_dataset
)

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_class=CosineSimilarityLoss,
    metric="f1",
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=3, #5 #15 #25 # Number of epochs to use for contrastive learning
)

# Start timer
start_time = time.time()

# Train and evaluate
trainer = accelerator.prepare(trainer)
trainer.train()
print('\nTraining Time elapsed:')
print(time.strftime(
    '%H hr : %M min : %S sec',
    time.gmtime(time.time()-start_time)))

metrics = trainer.evaluate()

print(metrics)
wandb.log(metrics)