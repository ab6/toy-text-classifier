import pytest
from datasets import Dataset
from transformers import AutoTokenizer

# 1. Define your preprocessing function
def preprocess_function(examples, tokenizer):
    # Truncate and pad to a specific max length for testing
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=16
    )

# 2. Define your pytest fixture for the tokenizer
@pytest.fixture(scope="module")
def tokenizer():
    # Use a small, standard model for fast testing (e.g., DistilBERT)
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 3. Define the dummy dataset to test against
@pytest.fixture(scope="module")
def raw_dataset():
    return Dataset.from_dict({
        "id": ["0", "1"],
        "text": [
            "Hugging Face is awesome!", 
            "Pytest is great for testing ML pipelines."
        ],
        "label": [1, 1]
    })

# 4. The actual test functions
def test_preprocessing_output_keys(raw_dataset, tokenizer):
    # Apply preprocessing
    processed_dataset = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True
    )
    
    # Assert that required Hugging Face tensor keys exist
    expected_keys = {"input_ids", "attention_mask"}
    for example in processed_dataset:
        assert set(example.keys()).issuperset(expected_keys)

def test_preprocessing_output_shapes(raw_dataset, tokenizer):
    processed_dataset = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True
    )
    
    # Assert that all tokenized sequences are padded to max_length
    for input_ids in processed_dataset["input_ids"]:
        assert len(input_ids) == 16

def test_preprocessing_special_tokens(raw_dataset, tokenizer):
    processed_dataset = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True
    )
    
    # Assert that special tokens (e.g., [CLS] and [SEP]) are properly injected
    for input_ids in processed_dataset["input_ids"]:
        assert input_ids[0] == tokenizer.cls_token_id
        assert tokenizer.sep_token_id in input_ids
