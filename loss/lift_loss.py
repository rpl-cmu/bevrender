import torch
import numpy as np
from pytorch_metric_learning import losses

np.random.seed(15213)


class LiftedStructureLoss(losses.LiftedStructureLoss):
    def __init__(self, neg_margin=1, pos_margin=0, **kwargs):
        super().__init__(neg_margin, pos_margin, **kwargs)
        self.lift_loss = losses.LiftedStructureLoss(neg_margin=1, pos_margin=0)

    def get_loss(self, cmr_embeddings, map_embeddings):
        concate = torch.cat((cmr_embeddings, map_embeddings), dim=0)
        labels = torch.cat(
            (
                torch.arange(cmr_embeddings.shape[0]),
                torch.arange(cmr_embeddings.shape[0]),
            ),
            dim=0,
        )
        return self.lift_loss(concate, labels)


if __name__ == "__main__":
    lift_loss = LiftedStructureLoss()
    cmr_embeddings = torch.randn(2, 64, requires_grad=True)
    map_embeddings = torch.randn(2, 64, requires_grad=True)
    output = lift_loss.get_loss(cmr_embeddings, map_embeddings)
    output.backward()
    print(output.shape)
    print(output)
