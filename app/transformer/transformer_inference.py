import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class TransformerInference:
    def __init__(self, model_path="app/transformer/weights/checkpoint-180", model_name="aubmindlab/araelectra-base-discriminator"):
        # model_path is the directory containing the checkpoint (e.g., .../weights/checkpoint-210)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def vectorize_text(self, text, feature_type=None):
        # For transformers, vectorization is handled by the tokenizer
        return self.tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")

    def predict_author(self, text, feature_type=None):
        with torch.no_grad():
            encodings = self.vectorize_text(text, feature_type)
            outputs = self.model(**encodings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        # You may want to map pred to author names using a label map
        # For now, just return the predicted label (int)
        return (int(preds[0]))
