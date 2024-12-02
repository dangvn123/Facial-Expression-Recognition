# Facial Expression Recognition


This project focuses on facial emotion recognition using the `fer2013` dataset and a convolutional neural network (CNN) model. The following sections provide details on the setup, running the program, and the required software packages.

## Project Structure
- **Model.py**: Contains the neural network architecture.
- **load_data.py**: Loads and preprocesses data from the `fer2013.csv` file.
- **train.py**: Script to train the model.
- **residual_net.pth**: Pretrained model file.
- **fer2013.csv**: Dataset with labeled emotion images.


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
```

## How to Use
1. Prepare the Dataset
Place the fer2013.csv file in the project directory.

2. Train the Model
Run the following command to train the model:
```bash
python train.py
```
3. Test the Model
To test the pretrained model (residual_net.pth), you can:
- Modify the train.py script to add evaluation logic, or
- Create a separate script to load the model and test it on sample data.

