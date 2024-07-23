import torch
from torch import Tensor


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    ) -> None:
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )

    def get_loss(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)


if __name__ == "__main__":
    cross_entropy_loss = CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = cross_entropy_loss.get_loss(input, target)
    output.backward()
    print(output)
