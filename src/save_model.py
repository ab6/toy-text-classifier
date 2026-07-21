import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

model_name = "distilbert"
save_directory = "./saved_models/distilbert_orig"

# Save to your local folder
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)