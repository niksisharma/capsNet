# CapsNet Vision Transformer Hybrid

A comparative study of Capsule Networks, Vision Transformers, and hybrid architectures for CIFAR-100 image classification.

## Overview

This repository contains implementations and experiments comparing different deep learning architectures:

- **CNN Baseline**: Standard convolutional neural network baseline
- **Vision Transformer (ViT)**: Transformer-based image classification model
- **CapsNet**: Capsule Network implementation with dynamic routing
- **CapsViT Hybrid**: Novel hybrid architecture combining Capsule Networks with Vision Transformer features

## Project Structure

- `0. CNN_Baseline/` - Baseline CNN implementation
- `1. vit_run/` - Vision Transformer training code and results
- `2. CapsNet/` - Capsule Network baseline implementation
- `3. capsvit/` - Hybrid CapsNet-ViT architecture
- `data/` - CIFAR-100 dataset
- `Evaluation/` - Model evaluation scripts and metrics

## Requirements

- PyTorch
- torchvision
- einops
- GPUtil
- numpy

## Usage

Each model can be trained independently using the Python scripts in their respective directories. Models are trained on CIFAR-100 dataset with 100 classes.

## Results

Results and trained models are stored in the respective output directories for each architecture.
