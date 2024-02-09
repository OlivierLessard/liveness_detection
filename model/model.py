from torch import nn, Tensor


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int, nb_classes: int):
        """

        :param input_size: size of the input square image (same height and width)
        :param nb_classes: number of output classes
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(28*28, 512),
            nn.Linear(3 * input_size * input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, nb_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: input tensor of size [B, C, H, W]
        :return: prediction scores of size [B, N], N classes
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
