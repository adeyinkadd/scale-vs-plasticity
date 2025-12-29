# Scale vs. Plasticity: Robustness Benchmark

![Benchmark Result](plots/result.png)  
*Figure 1: Comparing a frozen foundation model (DINOv2) and an adaptive ResNet-18 under synthetic domain shift.*

---

## What This Is

This repository studies how **large frozen models** compare to **small adaptive models** when the input distribution shifts.

We simulate a pathology scanner shift (color cast and sensor noise) and evaluate how each model responds over time:

- **Goliath (Scale):** Frozen **DINOv2 Vision Transformer**
- **David (Plasticity):** **ResNet-18** with **Test-Time Adaptation (TTA)**

Model behavior is evaluated using **predictive uncertainty (Shannon entropy)**.

---

## Key Findings

- **Frozen ViT (Goliath):** Uncertainty remains consistently high under domain shift. **Pretraining scale alone does not enable adaptation.**
- **Adaptive ResNet (David):** Entropy decreases over time as **BatchNorm statistics adapt** to the new input distribution.

**Takeaway:** Under non-stationary conditions, **lightweight test-time adaptation can outperform scale alone**.

---

## Why This Matters

- **Robustness is not guaranteed by scale alone:** Large frozen models cannot adapt to local domain shifts, even if pretrained on massive datasets.  
- **Relevant for low-resource medical imaging:** Nigerian labs and similar settings often have variable staining protocols, scanner calibration, and lighting conditions that introduce distribution shifts.  
- **Edge AI deployment:** Small adaptive models using Test-Time Adaptation (TTA) can recalibrate on-the-fly, requiring only limited compute and no labels, making them feasible for local devices.  
- **Practical advantage of plasticity:** TTA allows models to handle real-world variability where frozen foundation models may fail, highlighting the trade-off between pretraining scale (rigidity) and inference-time adaptation (plasticity).  

---

## Getting Started

### Installation(Local)

```bash
git clone https://github.com/adeyinkadd/scale-vs-plasticity.git
cd scale-vs-plasticity
pip install -r requirements.txt

Run the Benchmark
python benchmark.py
```
## This will:

Generate synthetic domain-shifted batches

Run both models through the stream

Record entropy and feature drift over time

Save comparison plots to plots/result.png

### Run on Google Colab
You can run this benchmark instantly in the cloud:

1. Open a new [Google Colab Notebook](https://colab.research.google.com/).
2. Create a code cell and run this block:
   ```python
   !git clone [https://github.com/adeyinkadd/scale-vs-plasticity.git](https://github.com/adeyinkadd/scale-vs-plasticity.git)
   %cd scale-vs-plasticity
   !pip install -r requirements.txt
   !python benchmark.py```


## Repository Structure
## Project Structure

```text
scale-vs-plasticity/
│
├─ plots/              # Saved visualization figures
│   └── result.png
├─ benchmark.py        # Core experiment engine
├─ domain_shift.py     # Synthetic domain shift generator
├─ metrics.py          # Entropy & Cosine distance calculations
├─ README.md           # Project documentation
└─ requirements.txt    # Python dependencies
```

## References

Wang et al., Tent: Fully Test-Time Adaptation by Entropy Minimization, ICLR 2021

Oquab et al., DINOv2: Learning Robust Visual Features without Supervision, 2023

<sub>Note: The adaptive model is referred to as “David” and the large frozen model as “Goliath,” reflecting both the contrast in model scale and the author’s name :).</sub>
