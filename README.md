##Task 1: Machine Learning Environment Setup with Pixi and W&B

Description
This repository contains code for setting up a Machine Learning (ML) development environment using Pixi for package management and integrating Weights & Biases (W&B) for experiment tracking and visualization. The code includes improvements over the original implementation, Pixi configuration, and W&B integration.

##Contents

Code Files:
test.py: Main Python script containing the ML development environment setup
pixi.yml: Pixi configuration file specifying dependencies and settings.


W&B Integration:
Sign up for a W&B account and obtain an API key.
<e57a4d626b612a171a76930cd20adf3ec3e7f09e> in the code with your actual API key.
create a project
Run the Pixi environment setup script to log experiments on W&B.

To Run
Install Pixi : iwr -useb https://pixi.sh/install.ps1 | iex
intialize pixi : pixi init texnano
add dependencies : pixi add python
to run : pixi build start


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



