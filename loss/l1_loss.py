import torch
from torch import Tensor


class L1Loss(torch.nn.L1Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def get_loss(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)


if __name__ == "__main__":
    l1loss = L1Loss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = l1loss.get_loss(input, target)
    output.backward()
    print(output)
