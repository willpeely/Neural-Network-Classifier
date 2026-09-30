# CIFAR-10 Neural Network Classifier

A convolutional neural network built with **PyTorch** to classify images from the **CIFAR-10** dataset.

The project covers the complete machine learning workflow, including data preprocessing and augmentation, CNN architecture design, model training, validation, learning-rate scheduling and final evaluation.

View the code here - https://willpeely.github.io/Neural-Network-Classifier/

## Results

The trained model achieved approximately **88% test accuracy** on the CIFAR-10 test dataset.

## Model Architecture

The classifier uses a custom convolutional neural network consisting of five convolutional blocks followed by fully connected layers.

Each convolutional block uses a combination of:

- `Conv2d` layers
- Batch normalisation
- ReLU activation
- Max pooling
- Progressive dropout regularisation

The number of convolutional channels increases throughout the network:

```text
3 → 32 → 64 → 128 → 256 → 512
```

The convolutional output is then flattened and passed through fully connected layers:

```text
Flatten
  ↓
256 neurons
  ↓
128 neurons
  ↓
10 output classes
```

The final layer produces predictions for the ten CIFAR-10 classes.

## Training Pipeline

The model is trained for **70 epochs** using mini-batch gradient descent with a batch size of **128**.

### Optimisation

Training uses:

- **Adam optimiser**
- Learning rate of `0.001`
- Weight decay of `0.0001`
- **Cross-entropy loss**

A `ReduceLROnPlateau` learning-rate scheduler monitors validation accuracy and reduces the learning rate when improvement begins to plateau.

## Data Augmentation

To improve generalisation and reduce overfitting, several transformations are applied to the training images:

- Colour jitter
- Random rotation
- Random horizontal flipping
- Random cropping with padding
- Normalisation

Validation and test images are not augmented, allowing performance to be measured on unmodified data.

## Regularisation

Several techniques are used throughout the network to reduce overfitting:

- Batch normalisation
- Dropout
- Weight decay
- Data augmentation
- Validation-based learning-rate scheduling

Dropout is increased in deeper convolutional layers, reaching `0.5` in the final convolutional block.

## Training and Validation

The CIFAR-10 training dataset is divided into:

```text
90% Training
10% Validation
```

After each training epoch, the model is evaluated against the validation dataset.

The training process reports:

```text
Loss
Training Accuracy
Validation Accuracy
Epoch Training Time
```

This makes it possible to monitor learning progress and identify when the model begins to plateau.

## CIFAR-10

CIFAR-10 contains **60,000 32×32 RGB images** across ten categories:

```text
Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
```

The dataset consists of 50,000 training images and 10,000 test images.

## Project Structure

```text
.
├── CNN.py
├── train.py
├── test.py
├── model.pth
└── data/
```

### `CNN.py`

Defines the neural network architecture and reusable functions for constructing convolutional and fully connected layers.

### `train.py`

Handles:

- CIFAR-10 loading
- Data augmentation
- Training/validation splitting
- Model training
- Validation
- Optimisation
- Learning-rate scheduling
- Saving the trained model

### `test.py`

Loads the saved model and evaluates its classification accuracy against the CIFAR-10 test dataset.

## Running the Project

### 1. Install dependencies

```bash
pip install torch torchvision
```

### 2. Train the model

```bash
python train.py
```

The CIFAR-10 dataset will be downloaded automatically if it is not already available.

Once training finishes, the model weights are saved as:

```text
model.pth
```

### 3. Evaluate the model

```bash
python test.py
```

The trained model is loaded and evaluated against the CIFAR-10 test dataset.

Example output:

```text
Test Accuracy - 88.00%
```

## Technologies

- Python
- PyTorch
- Torchvision
- Convolutional Neural Networks
- Data augmentation
- Batch normalisation
- Dropout
- Adam optimisation
- Learning-rate scheduling

## Key Learning

This project provided practical experience designing and training convolutional neural networks while exploring techniques for improving model generalisation.

The main areas explored included:

- Designing increasingly deep CNN architectures
- Controlling overfitting through regularisation
- Creating data augmentation pipelines
- Monitoring training and validation performance
- Dynamically adjusting the learning rate
- Evaluating trained models on unseen test data
