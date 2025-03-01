import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())

#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        loss = loss_fn(descriptors, labels, miner_outputs)
        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss


class GeneralizedRecallloss(torch.nn.Module):
    def __init__(self,k_vals=[1], tmp1=0.01, tmp2=0.1,tmp3=0.1):
        super().__init__()
        self.k_vals = torch.tensor(k_vals)
        self.tmp1=tmp1
        self.tmp2=tmp2
        self.tmp3 = tmp3
    def forward(self, scores, args=None, triplets_global_indexes=None):
        batch, num_reference = scores.shape
        scores = scores.to(torch.float32)
        loss = torch.tensor(0, device="cuda", dtype=torch.float, requires_grad=True)
        gts = torch.zeros_like(scores)
        gts[:, :args.pos_num_per_query] = 1
        for score, gt, global_index in zip(scores, gts, triplets_global_indexes):
            pos_score = score[gt == 1]
            N_pos = len(pos_score)
            neg_score = score[gt == 0]
            target = gt.bool().unsqueeze(-1).repeat(1, N_pos)
            pos_diff_all = pos_score.unsqueeze(-1) - pos_score.unsqueeze(0).repeat(N_pos, 1)
            neg_diff_all = neg_score.unsqueeze(-1) - pos_score.unsqueeze(0).repeat(len(neg_score), 1) + self.neg_margin
            tensor_diff_all = torch.cat((pos_diff_all, neg_diff_all), dim=0)
            mask_1 = (tensor_diff_all >= 0).bool()
            tensor_diff_all[~target & ~mask_1] = torch.log(1 + torch.exp(tensor_diff_all[~target & ~mask_1] / self.tmp1))
            tensor_diff_all[~target & mask_1] = torch.log(1 + torch.exp(tensor_diff_all[~target & mask_1] / self.tmp2)) + 0.1
            tmp = tensor_diff_all[target]
            tmp[tmp >= 0] = 1
            tmp[tmp < 0] = 0
            tensor_diff_all[target] = torch.tensor(tmp, device=tensor_diff_all.device, dtype=tensor_diff_all.dtype)
            sim_sg = tensor_diff_all
            for i in range(N_pos):
                sim_sg[i, i] = 0
            sim_all_rk = (1 + torch.sum(sim_sg, dim=0)).unsqueeze(0)
            sim_all_rk = sim_all_rk.unsqueeze(-1).repeat(1, 1, len(self.k_vals))
            k_vals = self.k_vals.unsqueeze(0).unsqueeze(0).repeat(1, N_pos, 1).cuda()
            recall_pos = sim_all_rk - k_vals
            recall_pos = torch.log(1 + torch.exp(torch.clamp(recall_pos / self.tmp3, max=88))) / N_pos
            recall_pos = torch.sum(recall_pos) / len(k_vals)
            loss = loss + recall_pos
        return loss / batch
