#%%
import json  
dir1 = '/home/jisoo/VScode/MemSum/data/custom_data/train__CUSTOM_labelled.jsonl'
# %%
train_corpus = [ json.loads(line) for line in open(dir1) ]

# %%
for i in range(len(train_corpus)):
    if len(train_corpus[i]['indices'])<1:
        print(i)
# %%
train_corpus[0]['indices']
# %%
from tqdm import  tqdm
corpus = []
with open( dir1, "r" ) as f:
    for line in tqdm(f):
        data = json.loads(line) #수정
        # if len(data["text"]) == 0 or len(data["summary"]) == 0:
        #     continue
        # if is_training:
        #     if len( data["indices"] ) == 0 or len( data["score"] ) == 0:
        #         continue

        corpus.append( data )

# %%
corpus
# %%
corpus[0]['indices']
# %%
for i in range(len(corpus)):
    if corpus[i]['indices']==[[]]:
        print(i)
# %%
