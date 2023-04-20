from datasets import load_dataset
import timeit
import html
import re

# Main data folder
#DF_MAIN = 'C:\\Users\\2bowb\\Documents\\Thesis\\'

# Data file name
FN_DATA = 'articles_labeled.csv'

# Output file name
FN_OUT = 'articles_dataset'

# read the csv file and convert it into a dataset object named 'dataset'
start = timeit.default_timer()

dataset = load_dataset('csv', data_files=FN_DATA, sep=",")

end = timeit.default_timer()
print("Conversion to Dataset:", {end - start}, "seconds")

# Removing duplicates
dsf = dataset.filter(lambda x: x["Text"] is not None)

print("Dataset features:", dsf)

# Clean the data
def clean_text(example):
    text = example['Text']
    text = text.replace("\n", " ") # cleaning newline "\n" from the text
    text = html.unescape(text) # decode html characters
    text = re.sub("@[A-Za-z0-9_:]+","",text) # remove mentions
    text = re.sub(r'http\S+', '', text) # remove URLs
    text = re.sub('RT ', '', text) # remove mentions
    return {'Text': text.strip()} # strip white space

dataset_clean = dsf.map(clean_text)

# Removing duplicates
dsf_clean = dataset_clean.filter(lambda x: x["Text"] is not None)

# Create splits: 90% train - 10% test, from train set split (80% train - 20% validation)
# Create train-test split
dataset_new = dsf_clean["train"].train_test_split(train_size=0.9, seed=42)
print("Split Dataset features:", dataset_new)

# From train split, create the train - validation split
dataset_splits = dataset_new["train"].train_test_split(train_size=0.8, seed=42)

# Rename the default "test" split to "validation"
dataset_splits["validation"] = dataset_splits.pop("test")

# Add the "test" set to our `DatasetDict`
dataset_splits["test"] = dataset_new["test"]
print("Features of Dataset to be saved:", dataset_splits)

# Write results to dataset
out_fn = FN_OUT
dataset_splits.save_to_disk(out_fn)
print("\nWriting results to: {}".format(out_fn))