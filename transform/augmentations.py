import random

from PIL import ImageFilter, ImageOps


class GaussianBlur:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class GaussianBlurSigmaRange:
    def __init__(self, sigma: list[float] = [0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img
