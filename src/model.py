import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification


class Classifier(nn.Module):
    """
    A classifier that fine-tunes a pretrained BERT model.
    Assumes multiclass classification.
    """

    def __init__(self, pretrained_model_name="bert-base-uncased", num_labels=3, dropout=0.3):

        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)  # <--- add this
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, text: list[str]) -> torch.Tensor:
        # Tokenize the input sentences
        encoding = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.bert.device)

        # BERT forward pass
        outputs = self.bert(**encoding)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        # Apply dropout
        dropped = self.dropout(pooled_output)

        # Classification head
        logits = self.classifier(dropped)  # [batch_size, num_labels]
        return logits


def create_hf_model(pretrained_model_name: str, num_labels: int):
    """Create a HuggingFace model for sequence classification."""
    return AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, 
        num_labels=num_labels
    )


if __name__ == "__main__":
    # Dummy input sentence
    sentences = ["This movie was great!"]

    # Model
    model = Classifier()
    model.eval()  # Inference mode

    # Forward pass
    with torch.no_grad():
        logits = model(sentences)

    print(f"Output shape of model: {logits.shape}")  # e.g., [1, 3] for 3-class classification
