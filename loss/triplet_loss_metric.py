import torch
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.distances import CosineSimilarity


class TripletLossMetricLearning(losses.TripletMarginLoss):
    def __init__(self):
        super().__init__()
        self.triplet_loss = losses.TripletMarginLoss(
            distance=CosineSimilarity(),
            reducer=ThresholdReducer(high=0.3),
            embedding_regularizer=LpRegularizer(),
        )
        self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")

    def get_loss(self, cmr_embeddings, map_embeddings):
        concate_embeddings = torch.cat((cmr_embeddings, map_embeddings), dim=0)
        labels = torch.cat(
            (
                torch.arange(cmr_embeddings.shape[0]),
                torch.arange(map_embeddings.shape[0]),
            ),
            dim=0,
        )
        hard_pairs = self.miner(concate_embeddings, labels)
        return self.triplet_loss(concate_embeddings, labels, hard_pairs)


if __name__ == "__main__":
    triplet_loss_metric_learning = TripletLossMetricLearning()
    cmr_embeddings = torch.randn(2, 64, requires_grad=True)
    map_embeddings = torch.randn(2, 64, requires_grad=True)
    output = triplet_loss_metric_learning.get_loss(cmr_embeddings, map_embeddings)
    output.backward()
    print(output.shape)
    print(output)
