import torch

class DomainShifter:
    @staticmethod
    def nigeria_lab_shift(x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
        """Simulates purple cast and sensor noise."""
        x_shifted = x.clone()
        noise = torch.randn_like(x_shifted) * (0.1 * severity)
        x_shifted[:, 0, :, :] += (2.0 * severity)  # R
        x_shifted[:, 1, :, :] -= (2.5 * severity)  # G
        x_shifted[:, 2, :, :] += (2.0 * severity)  # B
        return x_shifted + noise