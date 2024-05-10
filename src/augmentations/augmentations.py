from PIL import Image
import numpy as np

from src.benchmark.features import Features


def overlay_img(img: Image.Image, path: str = "data/octopus.png") -> Image.Image:
    img_ = Image.open(path).convert("RGBA")
    size_ = np.array(img_.size) // 8
    img_ = img_.resize(tuple(size_))
    pos = np.array(img.size) // 2 - size_ // 2
    print(pos)

    img.paste(img_, tuple(pos), img_)
    return img


def prune_corners(f: Features, p: float = 0.2) -> Features:
    mask = np.random.rand(f.board.shape[0]) > p
    print(mask.shape, f.corners.shape)
    f.board = f.board[mask]
    f.corners = f.corners[mask]
    return f
