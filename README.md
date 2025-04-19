# DA6401 Assignment 2

## Overview

This assignment implements two distinct approaches to image classification:

### Part A: Building a CNN from Scratch
- **Complete custom implementation** using PyTorch primitives
- **No pre-trained weights** - all layers initialized randomly
- **Handcrafted architecture** with configurable:
  - Convolutional layers
  - Dense layers
  - Activation functions
  - Regularization (Dropout/BatchNorm)

### Part B: Transfer Learning with InceptionV3
- **Pre-trained InceptionV3** backbone (ImageNet weights)
- **Custom classifier head** for target task
- **Fine-tuning options** for partial/full network

## File Structure

### Part A Files
| File | Purpose |
|------|---------|
| `model_builder.py` | - Implements custom Conv2d/Linear layers <br> - Handles dimension calculations <br> - Provides architecture configuration |
| `main.py` | - Data loading <br> - Training loop <br> - Hyperparameter sweep |

### Part B Files
| File | Purpose |
|------|---------|
| `model_utils.py` | - Data augmentation <br> - Stratified splits <br> - Model modification |
| `main.py` | - Fine-tuning logic <br> - Performance evaluation |

## Links
[Wandb Link](https://api.wandb.ai/links/surendarmohan283-indian-institute-of-technology-madras/8g72qh0l)
