# Fine-Tuned DistilBERT Model

## Hugging Face Hub Location
The fine-tuned model is hosted on Hugging Face Hub:
- **Model ID**: `ARG33/DistilBERT-finetuned`
- **Hub URL**: https://huggingface.co/ARG33/DistilBERT-finetuned

## Loading the Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
check_point = "ARG33/DistilBERT-finetuned"

model = AutoModelForSequenceClassification.from_pretrained(
    check_point
)
tokenizer = AutoTokenizer.from_pretrained(
    check_point
)
```

## Training Details
See `config.json` for training hyperparameters and performance metrics.

