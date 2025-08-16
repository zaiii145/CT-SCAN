# DL-Based-CT-Scan-Classification
## Overview
This project involves classifying CT scan images using deep learning with TensorFlow. The dataset includes images categorized into aneurysm, cancer, and tumor, stored in DICOM (.dcm) and JPEG (.jpg) formats. The main code is in the Jupyter notebook ct_scan.ipynb , which handles data loading, model building, training, and evaluation for image classification tasks.

## Project Structure
- files/ : Contains subfolders for different categories:
  - aneurysm/ : CT scans related to aneurysms.
  - cancer/ : CT scans related to cancer.
  - tumor/ : CT scans related to tumors.
- ct_scan.ipynb : Jupyter notebook with the code for loading data, building and training a CNN model using TensorFlow/Keras.
- requirements.txt : List of dependencies.
- ct_env/ and tf_env/ : Virtual environments for the project.
- archive/ : Archived files.


## Setup
1. Clone the repository or navigate to the project directory:
   cd /Users/zaina/ct classify
   
2. Create and activate a virtual environment (if not using the provided ones):
   python -m venv ct_env
   source ct_env/bin/activate
   
3. Install dependencies:
   pip install -r requirements.txt
   Note: Ensure TensorFlow and other required libraries like numpy, pandas, matplotlib, opencv, scikit-learn are installed.

   Dataset: https://www.kaggle.com/datasets/trainingdatapro/computed-tomography-ct-of-the-brain

## Usage
1.  Open the Jupyter notebook:
   jupyter notebook dl_simplified_ct.ipynb
   
2. Run the cells to:
   - Import libraries and load the dataset from the files/ directory.
   - Preprocess images (resizing, normalization).
   - Build a Convolutional Neural Network (CNN) model with layers like Conv2D, MaxPooling2D, Dense, etc.
   - Train the model using train-test split and evaluate its performance.
## Model Details
The notebook uses TensorFlow/Keras to create a sequential model for multi-class classification. It includes data augmentation (RandomFlip, RandomBrightness), early stopping, and visualization of results.

## Dependencies
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV
- Scikit-learn
- Jupyter
For a full list, see requirements.txt .

## Contributing
Feel free to fork the repository and submit pull requests for improvements.

## License
This project is open-source and available under the MIT License (assuming standard open-source practices; update as needed).
"# CT-Scan-" 
"# CT-Scan-" 

#


