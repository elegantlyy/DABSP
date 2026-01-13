# DABSP: Dual-directional Attention Based Multimodal Data Fusion Framework for Pan-Cancer Survival Outcome Prediction

This repository contains the implementation of **DABSP**, a framework for integrating histologic and transcriptomic data for pan-cancer survival outcome prediction through a dual-directional attention based multimodal data fusion framework.

## Citation

If you use this code in your research, please cite:

```
DABSP: Integrating histologic and transcriptomic data for pan-cancer survival outcome prediction through a dual-directional attention based multimodal data fusion framework
```

## Overview

![Overview](figs/Overview.png)

DABSP is a multimodal deep learning framework that combines:
- **Histologic data**: Whole Slide Images (WSI) from histopathology
- **Transcriptomic data**: RNA-seq gene expression data organized by biological pathways

The framework uses a dual-directional attention mechanism with LoRA (Low-Rank Adaptation) to effectively fuse these modalities for improved survival prediction.

## Features

- Multimodal fusion of WSI and omics data
- Dual-directional attention mechanism with LoRA
- Support for multiple cancer types (pan-cancer)
- 5-fold cross-validation for robust evaluation
- Multiple pathway types: `xena`, `hallmarks`, `combine`
- Chebyshev KAN (Kolmogorov-Arnold Network) for pathway processing

## Requirements

- Python 3.x
- PyTorch
- CUDA (for GPU acceleration)


## Quick Start

### Example Usage

Run the example script for the COAD dataset:

```bash
bash scripts/run_coad.sh
```



## Supported Cancer Types

The framework supports multiple TCGA cancer types. Update the `STUDIES` variable in the script to run on different datasets:
- COAD (Colon Adenocarcinoma)
- BRCA (Breast Invasive Carcinoma)
- BLCA (Bladder Urothelial Carcinoma)
- HNSC (Head and Neck Squamous Cell Carcinoma)
- STAD (Stomach Adenocarcinoma)
- And more...


