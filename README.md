# UrbanSound8K Audio Classification

## Overview
This project implements a Deep Learning pipeline to classify urban sounds using the UrbanSound8K dataset. The study compares a baseline Multi-Layer Perceptron (MLP) against a Convolutional Neural Network (CNN), investigating the impact of feature resolution, regularisation, and data augmentation on model performance.

## Dataset
The UrbanSound8K dataset contains 8,732 labeled sound excerpts (≤ 4s) from the following 10 classes:

air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music.

The dataset is pre-divided into 10 folds, enabling a rotating 10-fold cross-validation setup:
- 8 folds → Training
- 1 fold → Validation
- 1 fold → Test

## Methodology

### 1. Preprocessing
To ensure consistent inputs for all neural networks, the following preprocessing steps were applied:

- Resampling: Converted all audio clips to 22,050 Hz
- Duration Standardisation: Fixed length of 4.0 seconds  
- Normalisation: Amplitude scaled to [-1, 1]

### 2. Feature Extraction
Raw audio is transformed into Log-Mel Spectrograms.

Feature parameters:
- Window Size: 2048
- Hop Length: 512
- Mel Bands: 64 → refined to 96

## Models & Experiments

### Model A — MLP Baseline
- Input: Flattened Mel spectrograms (~11k features)
- Architecture: 512 → 256 units, BatchNorm, Dropout
- Performance: ~48% test accuracy

### Model B — CNN (2D Convolution)
#### Input  
2D Mel spectrograms.

#### Architecture  
Conv2D → BatchNorm → MaxPool → Dropout (×3 blocks), Dense classifier.

#### Variants
- Base CNN (64 Mel Bands): ~54% accuracy  
- Refined CNN (96 Mel Bands): ~61% accuracy  
- Final CNN (Reg + Aug): Dropout, L2, SpecAugment → ~62% accuracy

## Key Results

| Model | Val Accuracy | Test Accuracy | Key Observation |
|-------|--------------|---------------|------------------|
| MLP (Baseline) | 53.0% | 47.8% | High overfitting |
| CNN (64 Bands) | 66.9% | 53.9% | Limited spectral detail |
| CNN (96 Bands) | 68.7% | 60.7% | Better low-frequency resolution |
| CNN (Reg + Aug) | 67.9% | 62.0% | Best robustness and stability |

## Robustness Analysis (DeepFool)
- Mean L2 Perturbation: ~1.38e6  
- Attack Success Rate: 87%  

## Dependencies
- Python 3.x  
- TensorFlow / Keras  
- Librosa  
- NumPy, Pandas  
- Matplotlib, Seaborn  

## Usage
1. Place the UrbanSound8K dataset in the project directory.  
2. Run the notebook/scripts to:
   - Preprocess data  
   - Train MLP  
   - Train CNN variants  
   - Visualize results  

