#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install cornac')
#get_ipython().system('pip install pandas')


# In[4]:


import cornac
import pandas as pd
from cornac.eval_methods import BaseMethod
from cornac.hyperopt import Discrete, Continuous
from cornac.hyperopt import RandomSearch
import random, math
import io


# In[5]:


data = pd.read_csv('https://raw.githubusercontent.com/changboon/cs608/master/cs608_ip_train_v2.csv',  dtype=str)
data['tmp'] = list(zip(data.user_id, data.item_id, data.rating))
data = data.tmp.tolist()

holdout = pd.read_csv('https://raw.githubusercontent.com/changboon/cs608/master/cs608_ip_probe_v2.csv', dtype=str)
holdout['tmp'] = list(zip(holdout.user_id, holdout.item_id, holdout.rating))
holdout = holdout.tmp.tolist()


# In[6]:


random.seed(123)
random.shuffle(data)

train = data[:math.ceil(0.2*len(data))]
test = data[math.ceil(0.2*len(data)):]

#ratio_split = cornac.eval_methods.RatioSplit(data=data, test_size=0.1, rating_threshold=1.0, seed=123, verbose=True)
cv = cornac.eval_methods.cross_validation.CrossValidation(data, n_folds=5, rating_threshold=1.0, seed=123, exclude_unknowns=True, verbose=True)

#eval_method = BaseMethod.from_splits(
#    train_data=train,  test_data=test, val_data=holdout, exclude_unknowns=True, verbose=True, seed=123
#)


# In[7]:


mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=50)
recall = cornac.metrics.Recall(k=50)
ndcg = cornac.metrics.NDCG(k=50)
auc = cornac.metrics.AUC()
mAP = cornac.metrics.MAP()


# In[8]:


wmf = cornac.models.WMF(max_iter=100, verbose=False, seed=123)

rs_wmf = RandomSearch(
    model=wmf,
    space=[
        Discrete("k", [5, 10, 15, 30, 50]),
        Continuous("lambda_u", low=1e-4, high=1e-1),
        Continuous("lambda_v", low=1e-4, high=1e-1),
        Continuous("learning_rate", low=1e-3, high=1e-1),
        Discrete('batch_size', [128, 256]),
        Continuous('a', low=0.8, high=1.4),
        Continuous('b', low=0.0, high=2e-1),
    ],
    metric=auc,
    eval_method=cv,
    n_trails=30,
)


# In[9]:


cornac.Experiment(
  eval_method=cv,
  models=[rs_wmf],
  metrics=[mae, rmse, recall, ndcg, auc, mAP],
  user_based=True
).run()


# In[ ]:


with open("submission.txt", "w") as f:
    item_idx2id = list(rs_wmf.best_model.train_set.item_ids)
    last_ok = 0
    for i in range(9402):
        try:
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_wmf.best_model.rank(i)[0][0:50]]) + "\n")
            last_ok = i
        except:
            f.write(" ".join([str(item_idx2id[rec]) for rec in rs_wmf.best_model.rank(last_ok)[0][0:50]]) + "\n")


# In[ ]:


saved_path = rs_wmf.best_model.save('./')


# In[ ]:


with open("best_params.txt", "w") as f:
    for key in rs_wmf.best_params.keys():
        f.write(key + ": " + str(rs_wmf.best_params[key]) +"\n")

