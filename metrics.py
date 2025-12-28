import torch
import torch.nn.functional as F

class Metrics:
    @staticmethod
    def entropy(logits: torch.Tensor) -> torch.Tensor:
        """Calculates Shannon Entropy (Uncertainty) of predictions."""
        p = torch.softmax(logits, dim=1)
        return -(p * torch.log(p + 1e-8)).sum(dim=1).mean()

    @staticmethod
    def cosine_distance(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """Calculates Feature Drift. 0.0 = Identical, 1.0 = Orthogonal."""
        return 1.0 - F.cosine_similarity(feat1, feat2, dim=1).mean().item()