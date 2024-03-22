from transformers import DistilBertModel
import torch.nn as nn


class SummarizationModel(nn.Module):
    def __init__(self):
        super(SummarizationModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.distilbert.config.dim, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(1)
        logits = self.classifier(pooled_output)
        return logits
