import torch
import torch.nn as nn
from transformers import AutoModel

class BaseModel(nn.Module):

    def __init__(self, params, num_classes=1):

        super(BaseModel, self).__init__()

        self.model = AutoModel.from_pretrained(params['transformer'])
        
        self.linear = nn.Linear(768, 128)
        self.output = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
         
    def forward(self, input_id, mask):

        _, pooled_output = self.model(input_ids= input_id, attention_mask=mask,return_dict=False)[:2]
        x = self.linear(pooled_output)
        x = self.relu(x)
        x = self.output(x)
        output = self.sigmoid(x)

        return output