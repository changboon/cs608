class HarmonicMean(RankingMetric):
    

    def __init__(self, k=-1):
        RankingMetric.__init__(self, name="HarmonicMean@{}".format(k), k=k)
        self.w_ndcg = 2.846685771
        self.w_ncrr = -0.992450325
        self.w_recall = -0.651643769


    @staticmethod
    def dcg_score(gt_pos, pd_rank, k=-1):
        """Compute Discounted Cumulative Gain score.

        Parameters
        ----------
        gt_pos: Numpy array
            Binary vector of positive items.

        pd_rank: Numpy array
            Item ranking prediction.

        k: int, optional, default: -1 (all)
            The number of items in the top@k list.
            If None, all items will be considered.

        Returns
        -------
        dcg: A scalar
            Discounted Cumulative Gain score.

        """
        if k > 0:
            truncated_pd_rank = pd_rank[:k]
        else:
            truncated_pd_rank = pd_rank

        ranked_scores = np.take(gt_pos, truncated_pd_rank)
        gain = 2 ** ranked_scores - 1
        discounts = np.log2(np.arange(len(ranked_scores)) + 2)

        return np.sum(gain / discounts)


    @staticmethod
    def measure_at_k(gt_pos, truncated_pd_rank):
        pred = np.zeros_like(gt_pos)
        pred[truncated_pd_rank] = 1

        tp = np.sum(pred * gt_pos)
        tp_fn = np.sum(gt_pos)
        tp_fp = np.sum(pred)

        return tp, tp_fn, tp_fp



    def compute(self, gt_pos, pd_rank, **kwargs):
        
        if self.k > 0:
            truncated_pd_rank = pd_rank[: self.k]
        else:
            truncated_pd_rank = pd_rank

        gt_pos_items = np.nonzero(gt_pos > 0)

        # Compute CRR
        rec_rank = np.where(np.in1d(truncated_pd_rank, gt_pos_items))[0]
        if len(rec_rank) == 0:
            return 0.0
        rec_rank = rec_rank + 1  # +1 because indices starts from 0 in python
        crr = np.sum(1.0 / rec_rank)

        # Compute Ideal CRR
        max_nb_pos = min(len(gt_pos_items[0]), len(truncated_pd_rank))
        ideal_rank = np.arange(max_nb_pos)
        ideal_rank = ideal_rank + 1  # +1 because indices starts from 0 in python
        icrr = np.sum(1.0 / ideal_rank)

        # Compute nDCG
        ncrr = crr / icrr

        dcg = self.dcg_score(gt_pos, pd_rank, self.k)
        idcg = self.dcg_score(gt_pos, np.argsort(gt_pos)[::-1], self.k)
        ndcg = dcg / idcg

        tp, tp_fn, _ = self.measure_at_k(gt_pos, truncated_pd_rank)
        recall = tp / tp_fn


        return recall * self.w_recall + ndcg * self.w_ndcg + ndrr * self.w_ncrr