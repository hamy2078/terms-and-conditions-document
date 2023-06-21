from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BertDataset(Dataset):
      
    def __init__(self, data,params):
        self.dataset = data
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained(params['transformer'])  # BertTokenizer
    def __len__(self):
        return len(self.dataset)
  
    def __getitem__(self, idx):
        text = self.dataset['summary'][idx]
        inputs = self.tokenizer(text, padding='max_length', max_length = self.params['max_length'], truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids'][0]
        # token_type_ids = inputs['token_type_ids'][0]
        attention_mask = inputs['attention_mask'][0]
    
        y = self.dataset['ad_label'][idx]
        return input_ids, attention_mask, y
    
    