class HarmonicMean(recall, ndcg, ncrr):
    w_ndcg = 2.846685771
    w_ncrr = -0.992450325
    w_recall = -0.651643769

    return recall * w_recall + ndcg * w_ndcg + ndrr * w_ncrr
