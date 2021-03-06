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
auc = cornac.metrics.AUC()
mAP = cornac.metrics.MAP()
f1 = cornac.metrics.FMeasure(k=50)


# In[25]:


nmf = cornac.models.ncf.recom_neumf.NeuMF(name='NeuMF', layers=(64, 32, 16, 8), act_fn='relu', reg_layers=(0.0, 0.0, 0.0, 0.0), num_epochs=40, batch_size=256, learner='adam', early_stopping={'min_delta': 0.001, 'patience': 5}, trainable=True, verbose=False, seed=123)

rs_nmf = RandomSearch(
    model=nmf,
    space=[
        Discrete('num_factors', [4, 8, 16]),
        Discrete('num_neg', [3,4,5]),
        Continuous("reg_mf", low=0, high=1e-1),
        Continuous("lr", low=1e-4, high=1e-2),
    ],
    metric=f1,
    eval_method=eval_method,
    n_trails=60,
)


# In[ ]:


cornac.Experiment(
  eval_method=eval_method,
  models=[rs_nmf],
  metrics=[mae, rmse, recall, ndcg, auc, mAP, f1],
  user_based=True
).run()


# In[ ]:


import pickle
saved_path = rs_nmf.best_model.save('./nmf/')
item_idx2id = list(rs_nmf.best_model.train_set.item_ids)
item_id2idx = rs_nmf.best_model.train_set.uid_map
mapping = {
    'item_idx2id': item_idx2id,
    'item_id2idx': item_id2idx,
}

pickle.dump( mapping, open( "./nmf/mapping.pkl", "wb" ) )


# In[18]:


with open("./nmf/submission.txt", "w") as f:
    item_idx2id = list(rs_nmf.best_model.train_set.item_ids)
    item_id2idx = rs_nmf.best_model.train_set.uid_map
    last_ok = item_id2idx["1"]
    for i in range(9402):
        try:
            user = item_id2idx[str(i+1)]
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_nmf.best_model.rank(user)[0][0:50]]) + "\n")
            last_ok = user
        except:
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_nmf.best_model.rank(last_ok)[0][0:50]]) + "\n")

with open("./nmf/all.txt", "w") as f:
    item_idx2id = list(rs_nmf.best_model.train_set.item_ids)
    item_id2idx = rs_nmf.best_model.train_set.uid_map
    last_ok = item_id2idx["1"]
    for i in range(9402):
        try:
            user = item_id2idx[str(i+1)]
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_nmf.best_model.rank(user)[0]]) + "\n")
            last_ok = user
        except:
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_nmf.best_model.rank(last_ok)[0]]) + "\n")

