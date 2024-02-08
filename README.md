##Task 1: Machine Learning Environment Setup with Pixi and W&B

Description
This repository contains code for setting up a Machine Learning (ML) development environment using Pixi for package management and integrating Weights & Biases (W&B) for experiment tracking and visualization. The code includes improvements over the original implementation, Pixi configuration, and W&B integration.

##Contents

Code Files:
test.py: Main Python script containing the ML development environment setup
pixi.yml: Pixi configuration file specifying dependencies and settings.

Environment Setup:
Install Pixi by following the instructions in the Pixi Documentation.
Run the Pixi environment setup script: pixi run start.

W&B Integration:
Sign up for a W&B account and obtain an API key.
<WANDB_API_KEY> in the code with your actual API key.
Run the Pixi environment setup script to log experiments on W&B.

Experiment Tracking:
View experiment results on the W&B dashboard.
Metrics and visualizations logged include:
Image Data Shape
Label Data Shape
Sanity Check Plot
Visualize Test Plots
Mean Accuracy, Balanced Accuracy, Precision, and Recall
Confusion Matrix and Confusion Matrix Plot
Training Trials:

Conducted 5 training trials with the code and logged the results.
Getting Started
Follow these steps to set up the ML environment and run experiments:

Contact
For any questions or issues, please contact Harikrishnan at harikrishnan46624@gmail.com.



##Task 2: SimCLR Implementation with PyTorch

Description
This repository contains the implementation of a deep learning framework, SimCLR, for contrastive learning using PyTorch. The code follows the methodology and loss described in the SimCLR paper and is designed for self-supervised learning. The implementation uses small backbones like ResNet18 or ResNet50 and is trained on a vision dataset of your choice.

Contents
Code Files:
SimCLR.ipynb: Main Python script containing the SimCLR model, loss, training, and evaluation functions.

Training and Evaluation:
The code trains the SimCLR model using the specified loss and evaluates its performance on a vision dataset.
Metrics such as loss, accuracy, etc., are reported during training.

Hyperparameters:
Hyperparameters such as batch size, projection dimension, learning rate, and epochs are defined in the main function.
Visualization of Augmentations:

The code includes a function (visualize_augmentations) to visualize augmentations applied to the dataset.
Getting Started:

Update the hyperparameters in the main function according to your preferences.
Run the SimCLR implementation script: SimCLR.ipynb.
Dependencies
PyTorch
torchvision
matplotlib
Dataset
The code is designed to work with a vision dataset, and for this implementation, it uses the CIFAR-10 dataset. You can easily modify the dataset by changing the transform in the main function.

Experimentation
Conduct experiments to compare SimCLR performance against training a vanilla CNN for classification. Report relevant metrics and analyze the advantages of using SimCLR.

Contact
For any questions or issues, please contact Harikrishnan at harikrishnan46624@gmail.com.
