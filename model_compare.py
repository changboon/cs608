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
GLOBAL_DIR = "./model_compare/"


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


c2pf = cornac.models.c2pf.recom_c2pf.C2PF(name='C2PF', seed=123)
mter = cornac.models.mter.recom_mter.MTER(name="MTER", seed=123)
pcrl = cornac.models.pcrl.recom_pcrl.PCRL(name="PCRL", seed=123)
vaecf = cornac.models.vaecf.recom_vaecf.VAECF(name="VAECF", seed=123)
cvae = cornac.models.cvae.recom_cvae.CVAE(name='CVAE', seed=123)
gmf = cornac.models.ncf.recom_gmf.GMF(name='GMF', seed=123)
ibpr = cornac.models.ibpr.recom_ibpr.IBPR(name="IBPR")
mcf = cornac.models.mcf.recom_mcf.MCF(name='MCF', seed=123)
mlp = cornac.models.ncf.recom_mlp.MLP(name='MLP', seed=123)
neumf = cornac.models.ncf.recom_neumf.NeuMF(name='NeuMF', seed=123)
oibpr = cornac.models.online_ibpr.recom_online_ibpr.OnlineIBPR(name="OnlineIBPR")
vmf = cornac.models.vmf.recom_vmf.VMF(name='VMF', seed=123)
cdr = cornac.models.cdr.recom_cdr.CDR(name='CDR', seed=123)
coe = cornac.models.coe.recom_coe.COE(name="COE")
convmf = cornac.models.conv_mf.recom_convmf.ConvMF(name='ConvMF', seed=123)
skmeans = cornac.models.skm.recom_skmeans.SKMeans(name="SKMeans")
vbpr = cornac.models.vbpr.recom_vbpr.VBPR(name='VBPR', seed=123)
cdl = cornac.models.cdl.recom_cdl.CDL(name='CDL', seed=123)
hpf = cornac.models.hpf.recom_hpf.HPF(name="HPF")
efm = cornac.models.efm.recom_efm.EFM(name='EFM', seed=123)
sbpr = cornac.models.sbpr.recom_sbpr.SBPR(name='SBPR', seed=123)
hft = cornac.models.hft.recom_hft.HFT(name='HFT', seed=123)
wbpr = cornac.models.bpr.recom_wbpr.WBPR(name='WBPR', seed=123)
ctr = cornac.models.ctr.recom_ctr.CTR(name='CTR', seed=123)
bpr = cornac.models.bpr.recom_bpr.BPR(name='BPR', seed=123)
ga = cornac.models.global_avg.recom_global_avg.GlobalAvg(name='GlobalAvg')
iknn = cornac.models.knn.recom_knn.ItemKNN(name='ItemKNN', seed=123)
mf = cornac.models.mf.recom_mf.MF(name='MF', seed=123)
mmmf = cornac.models.mmmf.recom_mmmf.MMMF(name='MMMF', seed=123)
mp = cornac.models.most_pop.recom_most_pop.MostPop(name='MostPop')
nmf = cornac.models.nmf.recom_nmf.NMF(name='NMF', seed=123)
pmf = cornac.models.pmf.recom_pmf.PMF(name='PMF', seed=123)
svd = cornac.models.svd.recom_svd.SVD(name='SVD', seed=123)
sorec = cornac.models.sorec.recom_sorec.SoRec(name='SoRec', seed=123)
uknn = cornac.models.knn.recom_knn.UserKNN(name='UserKNN', seed=123)
wmf = cornac.models.wmf.recom_wmf.WMF(name='WMF', seed=123)

# In[ ]:

models = [
    c2pf, mter, pcrl, vaecf, cvae, gmf, ibpr, mcf, mlp, neumf, oibpr, vmf, cdr, coe, convmf, skmeans,
    vbpr, cdl, hpf, efm, sbpr, hft, wbpr, ctr, bpr, ga, iknn, mf, mmmf, mp, nmf, pmf, svd, sorec, uknn, wmf 
]

cornac.Experiment(
  eval_method=eval_method,
  models=models,
  metrics=[mae, rmse, recall, ndcg, auc, mAP, f1],
  user_based=True,
  save_dir="./model_compare/"
).run()


# In[ ]:


import pickle

for model in models:
    LOCAL_DIR = GLOBAL_DIR + model.name + "/"

    saved_path = model.save(LOCAL_DIR)
    item_idx2id = list(model.train_set.item_ids)
    item_id2idx = model.train_set.uid_map
    mapping = {
        'item_idx2id': item_idx2id,
        'item_id2idx': item_id2idx,
    }

    pickle.dump( mapping, open( LOCAL_DIR + "mapping.pkl", "wb" ) )

    with open(LOCAL_DIR + "submission.txt", "w") as f:
        item_idx2id = list(model.train_set.item_ids)
        item_id2idx = model.train_set.uid_map
        last_ok = item_id2idx["1"]
        for i in range(9402):
            try:
                user = item_id2idx[str(i+1)]
                f.write(" ".join([str(item_idx2id[rec]) for rec in model.rank(user)[0][0:50]]) + "\n")
                last_ok = user
            except:
                f.write(" ".join([str(item_idx2id[rec]) for rec in model.rank(last_ok)[0][0:50]]) + "\n")

    with open(LOCAL_DIR + "submission-all.txt", "w") as f:
        item_idx2id = list(model.train_set.item_ids)
        item_id2idx = model.train_set.uid_map
        last_ok = item_id2idx["1"]
        for i in range(9402):
            try:
                user = item_id2idx[str(i+1)]
                f.write(" ".join([str(item_idx2id[rec]) for rec in model.rank(user)[0]]) + "\n")
                last_ok = user
            except:
                f.write(" ".join([str(item_idx2id[rec]) for rec in model.rank(last_ok)[0]]) + "\n")

