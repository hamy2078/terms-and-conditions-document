import torch
from torch.utils.data import Dataset
import scipy

class DNNDataset(Dataset):
      
    def __init__(self, df, tfidf):
        self.df = df
        self.tfidf = tfidf
        
        self.texts = self.df['sent_jo']
        self.labels = self.df['label']
        
        self.texts = self.tfidf.transform(self.texts).astype('float32')

    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, idx):
        return {'text':torch.tensor(scipy.sparse.csr_matrix.todense(self.texts[idx])).float(), 
                'label':self.labels[idx]}