import torch
from pytorch_metric_learning import losses


class ContrastiveLoss(losses.ContrastiveLoss):
    def __init__(self, pos_margin=0.0, neg_margin=1.0):
        super().__init__(pos_margin=pos_margin, neg_margin=neg_margin)
        self.contrastive_loss = losses.ContrastiveLoss()

    def get_loss(self, cmr_embeddings, map_embeddings):
        concate_embeddings = torch.cat((cmr_embeddings, map_embeddings), dim=0)
        labels = torch.cat(
            (
                torch.arange(cmr_embeddings.shape[0]),
                torch.arange(map_embeddings.shape[0]),
            ),
            dim=0,
        )
        return self.contrastive_loss(concate_embeddings, labels)


if __name__ == "__main__":
    contrastive_loss = ContrastiveLoss()
    cmr_embeddings = torch.randn(2, 64, requires_grad=True)
    map_embeddings = torch.randn(2, 64, requires_grad=True)
    output = contrastive_loss.get_loss(cmr_embeddings, map_embeddings)
    output.backward()
    print(output.shape)
    print(output)
