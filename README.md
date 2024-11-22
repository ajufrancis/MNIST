# MNIST CI/CD Pipeline

This project demonstrates a basic CI/CD pipeline for training, testing, and deploying a Convolutional Neural Network (CNN) model on the MNIST dataset using PyTorch. The repository includes automated training, testing, and deployment scripts, integrated with GitHub Actions for continuous integration and deployment.

Directory Structure

```
project-directory/
|-- train.py
|-- test.py
|-- deploy.py
|-- .gitignore
|-- .github/
    |-- workflows/
        |-- ci.yml
```

- train.py: Script for training a CNN model on the MNIST dataset.
- test.py: Script for testing the trained model.
- deploy.py: Script for simulating the deployment of the model.
- .github/workflows/ci.yml: GitHub Actions workflow configuration for CI/CD.
- .gitignore: Lists files to be ignored by Git.

# Setup Instructions

## Prerequisites

Python 3.8+
Git

**Step 1:** Clone the Repository

Clone the repository to your local machine:
```
git clone <repository-url>
cd project-directory
```
**Step 2:** Set Up Virtual Environment

Create and activate a Python virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
**Step 3:** Install Dependencies

Install the required Python dependencies:
```
pip install --upgrade pip
pip install torch torchvision
```
**Step 4:** Train the Model

Run the training script to train a CNN model on the MNIST dataset:
```
python train.py
```
The model will be saved with a timestamp suffix after training is complete.

**Step 5:** Test the Model

Run the test script to verify the model's correctness:
```
python test.py
```
**Step 6:** Deploy the Model

Run the deployment script to simulate deploying the model:
```
python deploy.py
```
The deployment script will print the name of the latest model file to simulate deployment.

# CI/CD Pipeline with GitHub Actions

A GitHub Actions workflow is configured to automate the CI/CD pipeline. The workflow is triggered on every push or pull request to the main branch. It performs the following steps:

Checkout the Repository
Set up Python Environment
Install Dependencies
Train the Model
Test the Model
Deploy the Model

The workflow configuration can be found in .github/workflows/ci.yml.

How to Push to GitHub

To push your local changes to GitHub, run the following commands:

```
git add .
git commit -m "Add MNIST CI/CD pipeline"
git push origin main
```

# Project Files Overview

**train.py**

The train.py script defines a simple CNN model using PyTorch, trains it on the MNIST dataset for one epoch, and saves the model if it achieves an accuracy of at least 95%.

**test.py**

The test.py script tests the trained model to ensure it has the correct output dimensions and can process a standard MNIST input image (28x28 pixels).

**deploy.py**

The deploy.py script simulates deployment by printing out the name of the latest model saved during training.

# GitHub Actions Workflow

The GitHub Actions workflow (.github/workflows/ci.yml) automates training, testing, and deployment whenever code is pushed to the main branch. This helps ensure that the model and pipeline are always in a working state.

License

This project is licensed under the MIT License.

## Acknowledgements

PyTorch
GitHub Actions
MNIST Dataset

Contact

For any questions or issues, please open an issue in the repository or contact the repository owner.
