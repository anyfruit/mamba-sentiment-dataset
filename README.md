# Mamba Sentiment Classification

**Authors**: Xiaoke Song, Ruoyun Yang, Kejing Yan  
**Course**: DSI 1470, Brown University

---

## Overview

This project re-implements the **Mamba** model (Gu et al., 2024) — a linear-time sequence model with selective state spaces — in **TensorFlow**, and evaluates its performance on a natural language sentiment classification task. We benchmark Mamba against **Transformer**, **LSTM**, and **structured state-space (SSM)** models using the **Sentiment140** dataset.

Our goal is to assess Mamba’s modeling capability on textual data while comparing its **accuracy** and **inference speed** to established baselines.

---

## Key Features
- **TensorFlow Reimplementation** of Mamba architecture
- **Synthetic scan operations** using `tf.einsum`, `tf.cumsum`, and custom tensor reshaping
- **Unified evaluation** across Mamba, Transformer, LSTM, and SSM models
- **Standardized hyperparameters** for fair comparison
- **Benchmarking** on classification accuracy and inference runtime

---

## Dataset
- **Sentiment140 Dataset** ([Kaggle Link](https://www.kaggle.com/datasets/kazanova/sentiment140))  
- 1.6 million tweets labeled as positive, negative, or neutral

---

## Model Setup
| Model        | Depth | Main Layers                  | Hidden Units     | Dropout |
|--------------|-------|-------------------------------|------------------|---------|
| Mamba        | 2     | HighwayBlock (Selective SSM)  | 64 internal       | 0.2     |
| Transformer  | 2     | MHA + Feedforward             | 64 embed, 128 FF  | 0.1     |
| LSTM         | 2     | LSTM                          | 128 units         | 0.2     |
| SSM          | 2     | SimpleSSMLayer                | 64                | 0.1     |

> **Note**: Mamba uses deeper layers for better representation learning, and LSTM is assigned a higher dropout rate to mitigate overfitting.

---

## Model Repository Structure
- `config.py`: Model hyperparameter configurations
- `ssm.py`: Dynamic selective scan operations
- `mamba_block.py`: Core selective block design
- `residual_block.py`: Residual wrapper around selective blocks
- `model.py`: Full Mamba model construction
- `inference.py`: Simple inference function
- `run_test.py`: Script for model training and evaluation with synthetic/random data
- `README.md`: Project overview

---

## Quick Start
```bash
python run_test.py
