# PRISM: Position-as-Probability for Out-of-Distribution Sequence Generalization

Demo notebook for **"Position as Probability: Self-Supervised Transformers that Think Past Their Training for Length Extrapolation"** ([arXiv:2506.00920](https://arxiv.org/abs/2506.00920))

## **I. Environment & Configuration**
- CUDA optimization for A100 + torch.dynamo
- Model hyperparameters tuned for **Addition task** with drastically reduced number of attention heads/"cursors" for compute efficiency (significantly lower than paper reported values)
- Debug/logging configs
- WARNING: Code is provided "as-is" for reproducibility/research purposes only. As a disclaimer, code is in unmodified form from a colab debugging workflow. Significant refactoring is still needed.

## **II. Positional Encoding Implementations**
- Absolute and Relative Sinusoidal PE baselines
- **RSPE (Relative Sinusoidal PE)**: Core innovation with histogram update kernels
- Multiple kernel variants (v7, optimized backward passes, Triton-based)
- GRU with Frobenius gain for position probability scaling

## **III. Model Architecture**
- Transformer with RSE-enhanced Multi-Head Attention
- Position-as-probability mechanism integration

## **IV. Datasets**
12 synthetic reasoning tasks for length extrapolation testing:
- Copy, Odds-First, Stack Manipulation, Missing Duplicate, Bucket Sort
- Addition (with CoT), Reverse, Token Copy, Dynamic String Copy
- SCAN, Breadth-First Traversal, Multiplication

## **V. Training & Evaluation**
- **Progressive length curriculum** (trainlen 3→40+) for **compute efficiency only** — not strictly required for the method to work
- Train on in-distribution, evaluate on sequences exceeding training length
- Performance metrics: accuracy, loss convergence, extrapolation factors
- **Note:** Hyperparameters work well for Addition, OddsFirst, SCAN, and Stack Manipulation despite reduced model capacity

## **VI. Visualization & Analysis**
- Attention patterns and position probability distributions
- Alignment weights and scaling factors per layer
- Regularization effectiveness histograms
- Profiling and OOM debugging

**Result**: Demonstrates how PRISM enables transformers to generalize beyond training sequence lengths through probabilistic position encoding.

## Contact
For inquiries, please reach out to: **phil.hj.lee@gmail.com**
