# Patch-Based Neural Network Regularization for Hyperspectral Image Fusion

## Overview
This repository contains code for the implementation of a neural network-based approach for fusing hyperspectral images, utilizing patch-based regularization techniques. The project aims to enhance feature extraction and improve classification performance by effectively combining information from multiple hyperspectral datasets.

## Datasets
The code is tailored for the following hyperspectral datasets:
- **Indian Pines**: This scene captured by AVIRIS sensor over the Indian Pines test site in North-western Indiana and consists of a mix of agricultural and forested areas.
- **Pavia University**: Captured by the ROSIS sensor during a flight campaign over Pavia, northern Italy, this dataset primarily covers urban areas.

## Getting Started
### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Additional Python libraries as required (see `requirements.txt` for a complete list)

### Installation
1. Clone the repository:
git clone https://github.com/AfonsoSCCarvalho/Patch_NN.git
2. Navigate to the cloned repository:
cd Patch_NN
3. Install the necessary Python packages:
pip install -r requirements.txt

### Running the Scripts
To run the scripts for image fusion:
1. Ensure that the data for both Indian Pines and Pavia University is downloaded and located in the appropriate directory.
2. Execute the main script:
python PatchNN_clean_fuse.py

## Contact
If you have any questions or would like to contribute to the project, please feel free to contact me.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

