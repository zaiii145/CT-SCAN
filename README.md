# DL-Based CT Scan Classification

## Overview

This project focuses on **multi-class classification of CT scan images** using deep learning. The model classifies images into three categories:

* Aneurysm
* Cancer
* Tumor

The implementation is done using **TensorFlow/Keras** in a Jupyter Notebook.

---

## Project Structure

```
CT-SCAN/
│
├── ct_scan.ipynb        # Main notebook (model + training + evaluation)
├── requirements.txt     # Dependencies
├── files/               # Dataset (not recommended to upload fully)
│   ├── aneurysm/
│   ├── cancer/
│   └── tumor/
├── archive/             # Optional archived data
└── README.md
```

---

## Features

* Image preprocessing (resizing, normalization)
* Data augmentation (flip, brightness)
* CNN-based image classification
* Model training and evaluation
* Visualization of results

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Scikit-learn

---

## Setup Instructions

1. Clone the repository:

```
git clone https://github.com/zaiii145/CT-SCAN.git
cd CT-SCAN
```

2. Create virtual environment:

```
python -m venv venv
```

3. Activate environment:

* Windows:

```
venv\Scripts\activate
```

4. Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the notebook:

```
jupyter notebook ct_scan.ipynb
```

The notebook performs:

* Data loading
* Preprocessing
* Model training
* Evaluation

---

## Model Details

* Convolutional Neural Network (CNN)
* Layers: Conv2D, MaxPooling, Dense
* Techniques:

  * Data augmentation
  * Early stopping
  * Train-test split

---

## Notes

* Large datasets are not fully uploaded due to size constraints.
* You can replace the dataset with your own CT scan images.

---

## Future Improvements

* Use pretrained models (ResNet, EfficientNet)
* Deploy model using Streamlit
* Improve dataset size and balance

---

## License

MIT License
