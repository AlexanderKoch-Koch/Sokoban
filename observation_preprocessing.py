import numpy as np


def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def normalize(img):
    return np.divide(img, 255)

def preprocess_atari(img):
    downsampled = downsample(img)
    grayscale = to_grayscale(downsampled)
    cropped = grayscale[16:96, :]
    normalized = normalize(cropped)
    return normalized
