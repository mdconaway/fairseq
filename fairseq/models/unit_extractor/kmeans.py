import numpy as np
import torch
from fairseq.typing import DataType, Device
from torch import Tensor, nn


class KmeansModel(nn.Module):
    def __init__(self, kmeans_path: str, device: Device, dtype: DataType):
        super().__init__()
        km_model: Tensor = np.load(kmeans_path)
        centroids_numpy = km_model.transpose()
        centroids = torch.from_numpy(centroids_numpy)
        self.centroids = centroids.to(device=device, dtype=dtype)
        self.centroid_norm = (self.centroids**2).sum(0, keepdims=True)

    def forward(self, x: Tensor) -> Tensor:
        dist: Tensor = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.centroids)
            + self.centroid_norm
        )
        return dist.argmin(dim=-1)
