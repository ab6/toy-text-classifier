import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb

# capture a dictionary of hyperparameters with config
config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = DistilBertTokenizer.from_pretrained("./saved_models/distilbert_orig")
model = DistilBertForSequenceClassification.from_pretrained("./saved_models/distilbert_orig")

# tokenizer = AutoTokenizer.from_pretrained("./saved_models/distilbert_orig")
# model = AutoModelForSequenceClassification.from_pretrained(
#     "./saved_models/distilbert_orig",
#     device_map="auto",
#     attn_implementation="sdpa"
# )
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def preprocess_function(examples):
    # Tokenize text with truncation and padding to DistilBERT's max limit
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128,
        return_tensors="pt"
    )

tokenized_datasets = ds['train'].map(tokenize_function)
unset = ds['train'].map(preprocess_function)

tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="wandb"
)
# 4. Create the PyTorch DataLoader
train_loader = DataLoader(
    dataset=tokenized_datasets,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=data_collator
)

# 6. Iterate exactly like a normal PyTorch loop
# for batch in train_loader:
#     # Elements are automatically converted to torch.Tensor
#     input_ids = batch["input_ids"] 
#     labels = batch["labels"]
#     break

# for idx, feature in enumerate(train_loader):
#     print(f"Batch {idx+1}:", feature)

        
# start a new experiment
with wandb.init(project="new-dbert-model", config=config) as run:

    # set up model and data
    model, dataloader = model, train_loader

    # optional: track gradients
    run.watch(model)
    batch_ct = 0
    example_ct = 0
    for batch in dataloader:
        optimizer.zero_grad()
    
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        # Forward pass automatically computes the loss if labels are provided
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    
        loss.backward()
        optimizer.step()

        example_ct +=  16
        batch_ct += 1

        # Report metrics every 25th batch
        if ((batch_ct + 1) % 25) == 0:
            run.log({"loss": loss}, step=example_ct)
            print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")

