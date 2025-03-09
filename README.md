# Neural Network From Scratch

A Rust implementation of neural networks built from scratch as demonstrated in my [YouTube video](https://www.youtube.com/watch?v=qXRJnMe2NNc).

## Table of Contents

- [Neural Network From Scratch](#neural-network-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Usage](#usage)
    - [Command-Line Interface](#command-line-interface)
      - [Training a Model](#training-a-model)
      - [Evaluating a Model](#evaluating-a-model)
    - [Using as a Library](#using-as-a-library)
  - [Project Structure](#project-structure)
  - [YouTube Series](#youtube-series)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

This repository contains the code for a neural network implementation built entirely from scratch using Rust. The project demonstrates the fundamental concepts of neural networks without relying on external machine learning libraries.

## Features

- Matrix operations implementation
- Feed-forward neural network architecture
- Backpropagation algorithm
- Various activation functions
- Training and evaluation capabilities
- Model saving and loading functionality
- Command-line interface for training and evaluation

## Requirements

- Rust (latest stable version recommended)
- Cargo (comes with Rust)

## Installation

Clone this repository:

```bash
git clone https://github.com/share424/neural-network-from-scratch.git
cd neural-network-from-scratch
```

Build the project:

```bash
cargo build --release
```

The executable will be created as `main` (or `main.exe` on Windows) in the `target/release` directory. You can copy it to your project root for easier access:

```bash
cp target/release/main .
```

## Dataset

This project uses the MNIST dataset for training and testing the neural network. You can download the preprocessed binary files from the following URL:

- [X_train.bin](https://github.com/share424/neural-network-from-scratch/releases/download/Dataset/X_train.bin)
- [y_train.bin](https://github.com/share424/neural-network-from-scratch/releases/download/Dataset/y_train.bin)
- [X_test.bin](https://github.com/share424/neural-network-from-scratch/releases/download/Dataset/X_test.bin)
- [y_test.bin](https://github.com/share424/neural-network-from-scratch/releases/download/Dataset/y_test.bin)
  
The dataset consists of four files:
- `X_train.bin` - Training images (60,000 images, 784 features each)
- `y_train.bin` - Training labels (60,000 labels)
- `X_test.bin` - Test images (10,000 images, 784 features each)
- `y_test.bin` - Test labels (10,000 labels)

After downloading, place these files in a `data` directory in your project root:

```bash
mkdir -p data
# Copy or move the downloaded files to the data directory
mv X_train.bin y_train.bin X_test.bin y_test.bin data/
```

## Usage

### Command-Line Interface

The application provides a command-line interface with two main functionalities:

#### Training a Model

```bash
./main train --train-data <x_train.bin> --label <y_train.bin> [--output <model.bin>] [--lr <learning_rate>] [--epoch <num_epochs>] [--batch-size <batch_size>]
```

Parameters:
- `--train-data`: Path to the training data file (required)
- `--label`: Path to the training labels file (required)
- `--output`: Path where the trained model will be saved (optional, defaults to "model.bin")
- `--lr`: Learning rate for training (optional, defaults to 0.003)
- `--epoch`: Number of training epochs (optional, defaults to 1)
- `--batch-size`: Batch size for training (optional, defaults to 128)

Examples:
```bash
# Train with default parameters
./main train --train-data data/X_train.bin --label data/y_train.bin

# Train with custom learning rate and epochs
./main train --train-data data/X_train.bin --label data/y_train.bin --lr 0.001 --epoch 5

# Train with custom batch size and output file
./main train --train-data data/X_train.bin --label data/y_train.bin --batch-size 64 --output my_model.bin
```

#### Evaluating a Model

```bash
./main evaluate --test-data <x_test.bin> --label <y_test.bin> [--model <model.bin>]
```

Parameters:
- `--test-data`: Path to the test data file (required)
- `--label`: Path to the test labels file (required)
- `--model`: Path to the saved model file to evaluate (optional, defaults to "model.bin")

### Using as a Library

You can also use the neural network implementation as a library in your Rust code:

```rust
use belajar_neural_networks::{
    activation::Activation,
    model::Model,
    nn::Linear,
};

// Create layers
let layers = vec![
    Linear::new(784, 128, Activation::RELU),
    Linear::new(128, 64, Activation::RELU),
    Linear::new(64, 10, Activation::SOFTMAX),
];

// Create model
let mut model = Model::new(layers);

// Train model
model.train(&training_data, &training_labels, 0.01, 10, 32);

// Evaluate model
let accuracy = model.evaluate(&test_data, &test_labels);
println!("Accuracy: {}", accuracy);

// Save model
model.save("model.bin");

// Load model
model.load("model.bin");
```

## Project Structure

- `src/matrix.rs` - Matrix operations implementation
- `src/nn.rs` - Neural network layer implementation
- `src/model.rs` - Model training and evaluation
- `src/activation.rs` - Activation functions
- `src/main.rs` - Command-line interface implementation

## YouTube Series

This code was developed as part of my YouTube series on building neural networks from scratch. Watch the series here: [YouTube Channel Link]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

