# New-Name-Generator

This project implements a neural network to generate unique names using inspiration from the WaveNet paper published by Google's DeepMind. The model is trained on a dataset of names and can generate new, unique names based on learned patterns.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Web Application](#web-application)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

The Unique Name Generator uses a neural network to create unique names. The model is trained on a dataset of names and generates new names by predicting the next character based on a sequence of previous characters. This approach is inspired by the WaveNet architecture, which is known for its powerful generative capabilities.

## Project Structure

```
New-Name-Generator/
├── app.py               # Flask application for generating names via web interface
├── model.py             # Model definition and training script
├── names.txt            # Dataset of names used for training
├── templates/
│   └── index.html       # HTML template for the web interface      
└── README.md            # This README file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Anaconda](https://www.anaconda.com/products/distribution) (optional but recommended for managing dependencies)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/ankannn10/New-Name-Generator.git
   cd New-Name-Generator
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

## Usage

### Training the Model

The model can be trained using the `model.py` script. Ensure you have the `names.txt` file in the same directory, which contains the dataset of names.

To train the model, simply run:
```bash
python model.py
```

This will train the model and save the trained parameters to `model.pth`.

### Running the Web Application

To start the Flask web application, run:
```bash
python app.py
```

Open your web browser and go to `http://127.0.0.1:5000/`. You will see a simple web interface where you can specify the number of unique names to generate.


---


## Model Architecture

The model architecture is a hierarchical network designed to predict the next character in a sequence, given the previous characters as context. The architecture is composed of several key components:

1. **Embedding Layer**: 
   - The first layer of the network is an embedding layer, which converts input characters into dense vectors of a fixed size (`n_embd`).

2. **Consecutive Flattening Layers**:
   - The network includes multiple `FlattenConsecutive` layers, which reshape the input tensor to concatenate consecutive vectors. This step helps in capturing the relationships between consecutive characters.

3. **Linear Layers**:
   - Following the embedding layer and flattening layers, there are multiple linear (fully connected) layers. These layers perform affine transformations, mapping the input to a higher-dimensional space (`n_hidden`).

4. **Batch Normalization Layers**:
   - Batch normalization is applied after each linear transformation to stabilize and accelerate training. It normalizes the output of the previous layer by adjusting and scaling the activations.

5. **Non-linearity with Tanh**:
   - Each batch-normalized output is passed through a `Tanh` activation function, introducing non-linearity into the model and enabling it to learn more complex patterns.

6. **Hierarchical Structure**:
   - The model uses a hierarchical structure where each set of layers processes a different level of granularity in the input data, similar to the dilation structure in WaveNet. However, instead of dilated convolutions, this model uses flattening and linear transformations to capture context.
  
**Model Intuition:**<br>
<br>![wavenet](https://github.com/user-attachments/assets/0928a058-2c4a-46ff-acc1-208a13d9e7c9)

The model's architecture is implemented using a custom `Sequential` class, which stacks the aforementioned layers in the following order:

- `Embedding`
- `FlattenConsecutive(2)`, `Linear(n_embd * 2, n_hidden, bias=False)`, `BatchNorm1d(n_hidden)`, `Tanh()`
- `FlattenConsecutive(2)`, `Linear(n_hidden * 2, n_hidden, bias=False)`, `BatchNorm1d(n_hidden)`, `Tanh()`
- `FlattenConsecutive(2)`, `Linear(n_hidden * 2, n_hidden, bias=False)`, `BatchNorm1d(n_hidden)`, `Tanh()`
- `Linear(n_hidden, vocab_size)`


## Training

The model is trained on a dataset of names with a context length of 8 characters (block_size). During training, the model learns to predict the next character in a sequence, given the previous characters. The training process involves:
<br>
<br><img width="817" alt="Screenshot 2024-07-27 at 9 39 52 PM" src="https://github.com/user-attachments/assets/adefce0d-4ec4-46f1-b326-a1765535c835">


- **Data Preparation**: The dataset is split into training, validation, and test sets.
- **Forward Pass**: The model processes input data through its layers to compute the logits.
- **Loss Calculation**: The cross-entropy loss between the predicted logits and the true labels is calculated.
- **Backward Pass**: Gradients are computed using backpropagation.
- **Parameter Update**: Model parameters are updated using Stochastic Gradient Descent (SGD) with learning rate scheduling.

The training script (`model.py`) uses a dataset of names to train the neural network. The dataset is split into training, validation, and test sets. The training process includes:
- **Minibatch Gradient Descent:** For efficient training.
- **Cross-Entropy Loss:** As the loss function.
- **Simple SGD with Learning Rate Decay:** For optimization. <br>
<br>

**Training and Validation Loss**
<br>
<br>
<img width="182" alt="Screenshot 2024-07-27 at 9 27 36 PM" src="https://github.com/user-attachments/assets/0ffef300-e20d-4af8-9c80-7c8392a330da">
  <br><br>
**Plot**
<br><br>
<img width="581" alt="Screenshot 2024-07-27 at 10 04 22 PM" src="https://github.com/user-attachments/assets/70e59d34-108c-4637-ba4c-7f82fed9182c"><br>

 **Examples of Generated Names**
 <br><br>
<img width="146" alt="Screenshot 2024-07-27 at 9 27 49 PM" src="https://github.com/user-attachments/assets/5b448b8e-348b-4cc4-8eba-38440bc1f547">


## Web Application

The web application (`app.py`) uses Flask to provide a web interface for generating names. The application allows users to input the number of names to generate and displays the results.

## Acknowledgements

- This project is inspired by the WaveNet paper published by Google's DeepMind.
- The dataset of names is sourced from public domain sources.

## License

MIT<br>
Feel free to contribute to this project by opening issues or submitting pull requests. Your feedback and suggestions are highly appreciated!
