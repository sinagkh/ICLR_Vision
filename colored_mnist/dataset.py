"""Colored MNIST (Arjovsky et al., 2019) with a noisy binary label and a label-correlated digit color.

Label y = 1 for even digits and 0 for odd digits, flipped for 25% of the images. Color c equals y with probability
`corr` and is flipped otherwise. c = 0 draws the digit strokes in the red channel on a green background;
c = 1 draws them in the green channel on a red background. Items are (image 3x28x28, y, c).
The label noise and the color draws are fixed by `seed`; MNIST is downloaded by torchvision to <root>/MNIST.
"""
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

BG_INTENSITY = 0.8  # background tint, in the channel opposite to the digit color
LABEL_NOISE = 0.25  # share of flipped labels


def paint_digit(digit_2d, color):
    """Return a 3x28x28 image with the strokes in the red (color 0) or green (color 1) channel only."""
    img = torch.zeros(3, 28, 28)
    bg = digit_2d == 0
    stroke_ch, bg_ch = (0, 1) if color == 0 else (1, 0)
    img[stroke_ch] = digit_2d
    img[bg_ch][bg] = BG_INTENSITY
    return img


class ColoredMNIST(Dataset):
    def __init__(self, root, train, corr, seed):
        super().__init__()
        base = MNIST(root, train=train, download=True, transform=transforms.ToTensor())
        rng = random.Random(seed)
        self.images, self.labels, self.colors = [], [], []
        for x_gray, digit in base:
            y = int(digit % 2 == 0)
            if rng.random() < LABEL_NOISE:
                y = 1 - y
            color = y ^ int(rng.random() > corr)
            self.images.append(paint_digit(x_gray.squeeze(0), color))
            self.labels.append(y)
            self.colors.append(color)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i], self.colors[i]
