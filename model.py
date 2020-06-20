#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cornac
import pandas as pd
from cornac.eval_methods import BaseMethod
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import RandomSearch
import random, math
from custom_metric import HarmonicMean
import pickle

# In[21]:
GLOBAL_DIR = "./vaecf-final/"
LOAD_DIR = "./vaecf-hm1-ae20/"

random.seed(123)
data = cornac.data.Reader(bin_threshold=1.0, min_user_freq=10).read(fpath='./cs608_ip_stu_v2.csv', sep=",", fmt='UIR', skip_lines=1)

ratio_split = cornac.eval_methods.RatioSplit(data=data, test_size=0.1, rating_threshold=0.5, seed=123, verbose=True)

cv = cornac.eval_methods.cross_validation.CrossValidation(
    data, n_folds=3, rating_threshold=0.5, seed=123, exclude_unknowns=True, verbose=True)


# In[24]:


mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=50)
recall25 = cornac.metrics.Recall(k=25)
recall50 = cornac.metrics.Recall(k=50)
ndcg = cornac.metrics.NDCG(k=50)
ncrr = cornac.metrics.NCRR(k=50)
auc = cornac.metrics.AUC()
mAP = cornac.metrics.MAP()
f1 = cornac.metrics.FMeasure(k=50)
hm = HarmonicMean(k=50)


# In[25]:

best_params = pickle.load( open( LOAD_DIR + "best_params.pkl", "rb" ) )
vaecf = cornac.models.vaecf.recom_vaecf.VAECF(
    name='VAECF', 
    autoencoder_structure=[20], 
    n_epochs=best_params.get('n_epochs'), 
    k=best_params.get('k'),
    act_fn=best_params.get('act_fn'),
    beta=best_params.get('beta'),
    learning_rate=best_params.get('learning_rate'),
    batch_size=100, 
    trainable=True, 
    verbose=False, 
    seed=123, 
    use_gpu=True
)
mp = cornac.models.most_pop.recom_most_pop.MostPop(name='MostPop')

# In[ ]:


cornac.Experiment(
  eval_method=ratio_split,
  models=[vaecf, mp],
  metrics=[mae, rmse, recall25, recall50, ndcg, ncrr, auc, mAP, f1, hm],
  user_based=True,
  #save_dir=GLOBAL_DIR,
).run()


# In[ ]:

saved_path = vaecf.save(GLOBAL_DIR)
item_idx2id = list(vaecf.train_set.item_ids)
item_id2idx = vaecf.train_set.uid_map
mp_item_idx2id = list(mp.train_set.item_ids)

mapping = {
    'vaecf': {
        'item_idx2id': item_idx2id,
        'item_id2idx': item_id2idx,
    },
    'mp': {
        'item_idx2id': item_idx2id,
    }
}

pickle.dump( mapping, open( GLOBAL_DIR + "mapping.pkl", "wb" ) )
pickle.dump( vaecf, open( GLOBAL_DIR + "vaecf.pkl", "wb" ) )
pickle.dump( mp, open( GLOBAL_DIR + "mp.pkl", "wb" ) )


# In[18]:


with open(GLOBAL_DIR + "submission.txt", "w") as f:
    for i in range(9402):
        try:
            user = item_id2idx[str(i+1)]
            f.write(" ".join([str(item_idx2id[rec]) for rec in vaecf.rank(user)[0][0:50]]) + "\n")
        except:
            f.write(" ".join([str(mp_item_idx2id[rec]) for rec in mp.rank(1)[0][0:50]]) + "\n")

with open(GLOBAL_DIR + "submission-250.txt", "w") as f:
    for i in range(9402):
        try:
            user = item_id2idx[str(i+1)]
            f.write(" ".join([str(item_idx2id[rec]) for rec in vaecf.rank(user)[0][0:250]]) + "\n")
        except:
            f.write(" ".join([str(mp_item_idx2id[rec]) for rec in mp.rank(1)[0][0:250]]) + "\n")

