import wandb, timeit
from datasets import load_from_disk
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from accelerate import Accelerator


# Uncomment to record run in W&B
wandb.login()

run = wandb.init(
    project="TrainingLoop_SetFit")


# Main data folder
#DF_MAIN = 'C:\\Users\\2bowb\\Documents\\Thesis\\'

# Data file name
FN_DATA = 'articles_dataset'

# Load the dataset
dataset = load_from_disk(FN_DATA)

dataset = dataset.rename_column("Text", "text")
dataset = dataset.rename_column("match30", "label")

train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
eval_dataset = dataset["validation"]

# Load SetFit model from Hub
# model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# ("setu4993/smaller-LaBSE")

# Initialize the accelerator
accelerator = Accelerator()

# Prepare the model and datasets
model, train_dataset, eval_dataset = accelerator.prepare(
    model, train_dataset, eval_dataset
)

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss_class=CosineSimilarityLoss,
    batch_size=32,
    num_iterations=5, # Number of text pairs to generate for contrastive learning
    num_epochs=3, # Number of epochs to use for contrastive learning
)

# Train and evaluate
start = timeit.default_timer()
trainer = accelerator.prepare(trainer)
trainer.train()
end = timeit.default_timer()
print("Time to train:", {end - start}, "seconds")

metrics = trainer.evaluate()

print(metrics)
wandb.log(metrics)