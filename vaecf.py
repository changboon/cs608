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


# In[21]:
GLOBAL_DIR = "./vaecf/"

random.seed(123)
data = cornac.data.Reader().read(fpath='./cs608_ip_train_v2.csv', sep=",", fmt='UIR', skip_lines=1)
random.shuffle(data)

train = data[math.ceil(0.2*len(data)):]
test = data[:math.ceil(0.2*len(data))]

holdout = cornac.data.Reader().read(fpath='./cs608_ip_probe_v2.csv', sep=",", fmt='UIR', skip_lines=1)

ratio_split = cornac.eval_methods.RatioSplit(data=train, test_size=0.2, rating_threshold=1.0, seed=123)

eval_method = BaseMethod.from_splits(
    train_data=train,  test_data=test, val_data=holdout, exclude_unknowns=True, verbose=True, seed=123
)

cv = cornac.eval_methods.cross_validation.CrossValidation(
    data, n_folds=3, rating_threshold=1.0, seed=123, exclude_unknowns=True, verbose=True)


# In[24]:


mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=50)
recall = cornac.metrics.Recall(k=50)
ndcg = cornac.metrics.NDCG(k=50)
ncrr = cornac.metrics.NCRR(k=50)
auc = cornac.metrics.AUC()
mAP = cornac.metrics.MAP()
f1 = cornac.metrics.FMeasure(k=50)
hm = HarmonicMean(k=50)


# In[25]:

vaecf = cornac.models.vaecf.recom_vaecf.VAECF(name='VAECF', autoencoder_structure=[20], n_epochs=100, batch_size=100, trainable=True, verbose=False, seed=123, use_gpu=True)

rs = RandomSearch(
    model=vaecf,
    space=[
        Discrete('k', [8, 10, 15]),
        Discrete('act_fn', ['tanh','relu','relu6']),
        Discrete('likelihood', ['mult','bern','gaus']),
        Continuous("beta", low=0.8, high=1.3),
        Continuous("learning_rate", low=1e-4, high=1e-2),
    ],
    metric=hm,
    eval_method=eval_method,
    n_trails=60,
)


# In[ ]:


cornac.Experiment(
  eval_method=eval_method,
  models=[rs],
  metrics=[mae, rmse, recall, ndcg, ncrr, auc, mAP, f1, hm],
  user_based=True,
  save_dir=GLOBAL_DIR,
).run()


# In[ ]:


import pickle
saved_path = rs.best_model.save(GLOBAL_DIR)
item_idx2id = list(rs.best_model.train_set.item_ids)
item_id2idx = rs.best_model.train_set.uid_map
mapping = {
    'item_idx2id': item_idx2id,
    'item_id2idx': item_id2idx,
}

pickle.dump( mapping, open( GLOBAL_DIR + "mapping.pkl", "wb" ) )


# In[18]:


with open(GLOBAL_DIR + "submission.txt", "w") as f:
    item_idx2id = list(rs.best_model.train_set.item_ids)
    item_id2idx = rs.best_model.train_set.uid_map
    last_ok = item_id2idx["1"]
    for i in range(9402):
        try:
            user = item_id2idx[str(i+1)]
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs.best_model.rank(user)[0][0:50]]) + "\n")
            last_ok = user
        except:
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs.best_model.rank(last_ok)[0][0:50]]) + "\n")

with open(GLOBAL_DIR + "-250.txt", "w") as f:
    item_idx2id = list(rs.best_model.train_set.item_ids)
    item_id2idx = rs.best_model.train_set.uid_map
    last_ok = item_id2idx["1"]
    for i in range(9402):
        try:
            user = item_id2idx[str(i+1)]
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs.best_model.rank(user)[0][0:250]]) + "\n")
            last_ok = user
        except:
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs.best_model.rank(last_ok)[0][0:250]]) + "\n")

