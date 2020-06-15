#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cornac
import pandas as pd
from cornac.eval_methods import BaseMethod
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import RandomSearch
import random, math


# In[21]:


random.seed(123)
data = cornac.data.Reader().read(fpath='./cs608_ip_train_v2.csv', sep=",", fmt='UIR', skip_lines=1)
random.shuffle(data)

train = data[:math.ceil(0.2*len(data))]
test = data[math.ceil(0.2*len(data)):]

holdout = cornac.data.Reader().read(fpath='./cs608_ip_probe_v2.csv', sep=",", fmt='UIR', skip_lines=1)

ratio_split = cornac.eval_methods.RatioSplit(data=train, test_size=0.2, rating_threshold=1.0, seed=123)

eval_method = BaseMethod.from_splits(
    train_data=train,  test_data=test, val_data=holdout, exclude_unknowns=False, verbose=True, seed=123
)

cv = cornac.eval_methods.cross_validation.CrossValidation(
    data, n_folds=3, rating_threshold=1.0, seed=123, exclude_unknowns=True, verbose=True)


# In[3]:


mf = cornac.models.MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123)
pmf = cornac.models.PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123)
bpr = cornac.models.BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)


# In[24]:


mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=50)
recall = cornac.metrics.Recall(k=50)
ndcg = cornac.metrics.NDCG(k=50)
auc = cornac.metrics.AUC()
mAP = cornac.metrics.MAP()
f1 = cornac.metrics.FMeasure(k=50)


# In[25]:


wmf = cornac.models.WMF(max_iter=100, verbose=False, seed=123)

rs_wmf = RandomSearch(
    model=wmf,
    space=[
        Discrete("k", [10, 15, 30, 50]),
        Continuous("lambda_u", low=1e-4, high=1e-1),
        Continuous("lambda_v", low=1e-4, high=1e-1),
        Continuous("learning_rate", low=1e-3, high=1e-1),
        Continuous('a', low=0.8, high=1.4),
        Continuous('b', low=0.0, high=2e-1),
    ],
    metric=f1,
    eval_method=cv,
    n_trails=30,
)


# In[ ]:


cornac.Experiment(
  eval_method=cv,
  models=[rs_wmf],
  metrics=[mae, rmse, recall, ndcg, auc, mAP, f1],
  user_based=True
).run()


# In[ ]:


import pickle
saved_path = rs_wmf.best_model.save('./wmf/')
item_idx2id = list(rs_wmf.best_model.train_set.item_ids)
item_id2idx = rs_wmf.best_model.train_set.uid_map
mapping = {
    'item_idx2id': item_idx2id,
    'item_id2idx': item_id2idx,
}

pickle.dump( mapping, open( "./wmf/mapping.pkl", "wb" ) )


# In[18]:


with open("submission.txt", "w") as f:
    item_idx2id = list(rs_wmf.best_model.train_set.item_ids)
    item_id2idx = rs_wmf.best_model.train_set.uid_map
    last_ok = item_id2idx["0"]
    for i in range(9402):
        try:
            user = item_id2idx[str(i+1)]
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_wmf.best_model.rank(user)[0][0:50]]) + "\n")
            last_ok = user
        except:
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_wmf.best_model.rank(last_ok)[0][0:50]]) + "\n")
