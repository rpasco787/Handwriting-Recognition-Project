# Handwriting Recognition Project

## Overview

A **handwriting recognition system** built from scratch using vanilla Python and NumPy. Combines a custom feedforward neural network with character segmentation algorithms to recognize handwritten text. Trained on Extended MNIST (EMNIST) to recognize 47 characters: digits (0-9), uppercase (A-Z), and select lowercase letters.

**System Pipeline**: Image upload → Segmentation (line/word/character) → 28×28 normalization → Neural network classification → Text output

## Neural Network Architecture

### Structure
```
Input Layer:    784 neurons  (28×28 flattened grayscale images)
Hidden Layer:   30-100 neurons (configurable)
Output Layer:   47 neurons   (character classes)
```

- **Activation Function**: Sigmoid σ(z) = 1/(1 + e^(-z))
- **Weight Initialization**: Gaussian distribution (μ=0, σ=1/√n_in) to prevent vanishing/exploding gradients
- **Output**: Argmax of output layer activations

### Optimization & Training

**Algorithm**: Mini-batch Stochastic Gradient Descent (SGD)

**Loss Function**: Cross-Entropy
```
C = -1/n Σ[y·ln(a) + (1-y)·ln(1-a)]
```

**Regularization**: L2 Weight Decay
```
Update: w → w(1 - ηλ/n) - (η/m)∇C
```

**Backpropagation Equations**:
```
Error:          δ^L = ∇C ⊙ σ'(z^L)
Recursion:      δ^l = ((w^(l+1))^T · δ^(l+1)) ⊙ σ'(z^l)
Weight Grad:    ∂C/∂w^l = δ^l · (a^(l-1))^T
Bias Grad:      ∂C/∂b^l = δ^l
```

**Training Features**:
- 90/10 train/validation split
- Early stopping (20 epoch window)
- Per-epoch data shuffling
- Mini-batch gradient averaging

**Dataset**: EMNIST - 697K training samples, 116K test samples

## Installation & Usage

1. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv/Scripts/Activate  # Windows
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
flask --app flaskUI run
```

4. **Access**: Navigate to `http://localhost:5000`, upload handwritten text image, view results

## Technical Stack

- **Core**: Pure Python/NumPy (no ML frameworks)
- **Computer Vision**: OpenCV for segmentation
- **Web Interface**: Flask
- **Model**: Serialized to JSON for persistence
