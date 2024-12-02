# Face Emotion Recognition

This project focuses on facial emotion recognition using the `fer2013` dataset and a convolutional neural network (CNN) model. The following sections provide details on the setup, running the program, and the required software packages.

## Project Structure
- **Model.py**: Contains the neural network architecture.
- **load_data.py**: Loads and preprocesses data from the `fer2013.csv` file.
- **train.py**: Script to train the model.
- **residual_net.pth**: Pretrained model file.
- **fer2013.csv**: Dataset with labeled emotion images.
- **README.md**: Documentation file.
- **.gitattributes**: Configuration file for Git (if applicable).

## Requirements
Ensure the following software and libraries are installed:
- Python >= 3.8
- `numpy` for numerical operations
- `pandas` for data manipulation
- `torch` and `torchvision` for building and training the neural network
- `matplotlib` for visualizing results

Install the dependencies with:
```bash
pip install numpy pandas torch torchvision matplotlib
