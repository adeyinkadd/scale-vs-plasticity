import torch
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from transformers import Dinov2Model
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm
import gc

# --- IMPORTS FROM MODULES ---
from domain_shift import DomainShifter
from metrics import Metrics

# --- CONFIGURATION ---
@dataclass
class ExperimentConfig:
    batch_size: int = 64
    batches: int = 100
    learning_rate: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class RobustnessBenchmark:
    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.domain_shifter = DomainShifter()
        self.anchor_embedding = None 
        self._setup_models()
        self._setup_optimization()
        
    def _setup_models(self):
        logger.info(f"Initializing models on {self.cfg.device}...")
        # Goliath: Frozen DINOv2
        self.goliath = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.cfg.device)
        self.goliath.eval() 
        
        # David: Adaptive ResNet
        self.david = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.cfg.device)
        # Only optimize BatchNorm parameters (TTA)
        for name, param in self.david.named_parameters():
            param.requires_grad = "bn" in name
        self.david.train() 

    def _setup_optimization(self):
        trainable_params = filter(lambda p: p.requires_grad, self.david.parameters())
        self.optimizer = optim.SGD(trainable_params, lr=self.cfg.learning_rate)

    def _get_goliath_feats(self, x: torch.Tensor) -> torch.Tensor:
        return self.goliath(x).last_hidden_state[:, 0, :]

    def establish_anchor(self):
        with torch.no_grad():
            x_clean = torch.randn(self.cfg.batch_size, 3, 224, 224).to(self.cfg.device)
            self.anchor_embedding = self._get_goliath_feats(x_clean).mean(dim=0, keepdim=True)

    def run(self) -> Tuple[List[float], List[float]]:
        gc.collect()
        torch.cuda.empty_cache()
        self.establish_anchor()
        
        history_goliath = [] 
        history_david = []   
        
        progress_bar = tqdm(range(self.cfg.batches), desc="Processing Stream")
        for _ in progress_bar:
            # 1. Generate Data
            x_clean = torch.randn(self.cfg.batch_size, 3, 224, 224).to(self.cfg.device)
            x_corrupt = self.domain_shifter.nigeria_lab_shift(x_clean)

            # 2. Goliath (Monitor Drift)
            with torch.no_grad():
                feats_corrupt = self._get_goliath_feats(x_corrupt)
                drift_g = Metrics.cosine_distance(feats_corrupt, self.anchor_embedding)
                history_goliath.append(drift_g)

            # 3. David (Adapt & Measure Entropy)
            logits_d = self.david(x_corrupt)
            loss_ent = Metrics.entropy(logits_d)
            history_david.append(loss_ent.item())

            # Test-Time Adaptation Step
            self.optimizer.zero_grad()
            loss_ent.backward()
            self.optimizer.step()
            
            progress_bar.set_postfix({"Drift(G)": f"{drift_g:.3f}", "Ent(D)": f"{loss_ent.item():.2f}"})

        return history_goliath, history_david

    def plot_results(self, goliath_scores: List[float], david_scores: List[float]):
        sns.set_theme(style="whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 7))

        df = pd.DataFrame({'David': david_scores, 'Goliath': goliath_scores})
        df_smooth = df.rolling(window=10).mean()

        # Plot David (Entropy)
        color_d = '#e74c3c' 
        ax1.set_xlabel('Inference Batches (t)', fontsize=12)
        ax1.set_ylabel('David: Entropy (Smoothed)', color=color_d, fontsize=12, fontweight='bold')
        ax1.plot(df['David'], color=color_d, alpha=0.2, linewidth=1)
        line1 = ax1.plot(df_smooth['David'], label='David (Trend)', color=color_d, linewidth=4)
        ax1.tick_params(axis='y', labelcolor=color_d)

        # Plot Goliath (Drift)
        ax2 = ax1.twinx() 
        color_g = '#2c3e50' 
        ax2.set_ylabel('Goliath: Feature Drift (Smoothed)', color=color_g, fontsize=12, fontweight='bold')
        ax2.plot(df['Goliath'], color=color_g, alpha=0.2, linewidth=1, linestyle='--')
        line2 = ax2.plot(df_smooth['Goliath'], label='Goliath (Trend)', color=color_g, linewidth=3, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color_g)

        # Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', frameon=True, fontsize=12)

        plt.title(f"Scale vs Plasticity: {self.cfg.batches} Batches", fontsize=16, pad=20)
        plt.tight_layout()
        
        # Safe Save
        os.makedirs("plots", exist_ok=True) # <--- Added safety check here
        save_path = "plots/result.png"
        plt.savefig(save_path)
        print(f"\n[SUCCESS] Plot saved to {save_path}")
        # plt.show() # Commented out for headless environments (like servers), uncomment if local

if __name__ == "__main__":
    try:
        torch.manual_seed(42)
        config = ExperimentConfig()
        benchmark = RobustnessBenchmark(config)
        g_scores, d_scores = benchmark.run()
        benchmark.plot_results(g_scores, d_scores)
    except KeyboardInterrupt:
        print("\n[STOP] Benchmark interrupted by user.")