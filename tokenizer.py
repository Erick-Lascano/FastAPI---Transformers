from transformers import AutoTokenizer

def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize(text, tokenizer, max_length=128):
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )