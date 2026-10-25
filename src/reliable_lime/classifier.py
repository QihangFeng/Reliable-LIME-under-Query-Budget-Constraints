"""ResNet18 adapter for the notebook's HWC image classifier interface."""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models


class ResNet18Classifier:
    """Load the pretrained classifier when explicitly instantiated.

    Inputs are NHWC RGB arrays with floating-point values in [0, 1]. Outputs
    are NumPy arrays of ImageNet class probabilities. The first construction
    downloads the pretrained weights if they are not already cached.
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weights).to(self.device).eval()
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    @torch.no_grad()
    def __call__(self, images_np):
        imgs = []
        for img in images_np:
            img = img.astype(np.float32)
            tensor = self.preprocess(torch.from_numpy(img).permute(2, 0, 1))
            imgs.append(tensor)
        batch = torch.stack(imgs, dim=0).to(self.device)
        return F.softmax(self.model(batch), dim=1).detach().cpu().numpy()


def load_example_image():
    """Return the same built-in cat image used in the original notebook."""
    from skimage import data

    return data.cat().astype(np.float32) / 255.0
