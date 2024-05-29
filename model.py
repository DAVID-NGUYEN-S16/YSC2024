
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging

class ClaimVerification(nn.Module):
    def __init__(self, config):
        super(ClaimVerification, self).__init__()
        self.bert = AutoModel.from_pretrained(config.model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.4)
        self.fc = nn.Linear(self.bert.config.hidden_size, config.n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x