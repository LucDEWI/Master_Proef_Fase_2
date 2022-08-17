# Master_Proef_Fase_2

## Masters project 2022: Design and validation of a bin picking robot vison algoritm.

This project contains an algorithm for a vision system for a robot to detect grapsing positions on  an object in a box. This project was commisioned by Mitsubishi Chemical Advanced Materials. 

This project requires a dpeth camrera to function. It is set up to use an Intel Realsense D415 camera. 

# Installation
* Install Python 3.9
* Download this repository as a zip
* Download the Cornell grapsing dataset via: https://www.kaggle.com/datasets/oneoneliu/cornell-grasp 
* Open this repository in an IDE
* Install the required packages via Requirements.txt
* Instal the CUDA toolkit via: https://developer.nvidia.com/cuda-toolkit (if a Nvidia graphics card is available)
* Run the scripts in the order prescibed

# Sctructure of project

The following section descrbes the scripts and how to use these scripts. Extra information such as the use of CUDA are available in the comments of the scripts.

## Data
This file contains the neccesary files for the script to run. These are several .npy files.

## Models
These files contain the two models and neccesary functions for these models. The models are GGCNN and GGCNN2 from https://github.com/dougsm/ggcnn.

## Output
This directory contains all the trained models from the project. Newly trained models will also be put in this directory.

## Source
This directory contains the functions used in the prediction scripts and the calibration of the camera.

## Tensorboard
This directory contains the logging information of the trainingcycles. How to use this will be explained in the use section.

## Toepassingen
This directory contains the scripts to calibrate the camera, the position of the box and the prediction scripts.

## Utils
Contains the functions for the trainingsproces and evaluation of the network. Also contains functions related to the dataset.

## Eval_ggcnn.py
Script to perfom an evaluation of the network. Futher explained in use.

## Train_ggcnn.py
Script to train the network. Further explanation in use

# Use

To use this project execute the scripts in folowing order.

1. Torch_test

    Gives an output if CUDA is available to use. Use to test if CUDA is installed correctly. 
2. Cam_test
    
    Gives an color and dpeth image of the scene through the camera.
3. Intr_cal

    Performs the intrinsic calibration of the camera. Use the chessboard provided in the data directory.
4. EXTR_cal

    Performs the extrinsic calibration of the camera. Use the chessboard provided in the data directory.
5. Generate Cornell depth images

    Do this via following command.
    ```py
    python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>
    ```
6. Add_noise_dataset

    If desired extra noise can be added on the Cornell dataset.
    The path of to the dataset must be given in the script. Note that after noise has been to the dataset it can not be removed. It is recommended to take a copy of the dataset and add noise to this.
7. Training the network

    The network is trained is trained via the following command.
    ```py
    python train_ggcnn.py --description training_example --network ggcnn --dataset cornell --dataset-path <Path To Dataset>
    ```
    Via --help a full list will be given of all the commands.

    Models are saved in the Output Directory.
8. Evaluation of the model

    An evaluation can be performed via the following command.
    ```py
    python train_ggcnn.py --description training_example2 --network ggcnn --dataset cornell --dataset-path <Path To Dataset>
    ```
    Again via --help a full list of all the commands wil be given.
9. Box_finder

    This script is used to calibrate the position of the box. The user must enter the distance from the top of the box to the camera in millimeters. If the box is shown on the screen press and hold q to save the posistion of the box.

10. Prediction_without_box
    This script is used to make a prediction with a camera on a scene that does not use a box. A calibration of a box position is not required to use this script.

11. Prediction_with_box

    This script is used to make a prediction with a camera on scene with an open box. This must be calibrated wth the earlier scripts before this script is used.