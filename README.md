Lunar Crater Detection and Diameter Estimation
Overview
This project implements an automated system for detecting lunar craters and estimating their diameters using high-resolution images from the Orbiter High-Resolution Camera (OHRC) onboard the Chandrayaan-2 spacecraft. The approach leverages the YOLOv9 deep learning model for crater detection, followed by post-processing techniques to estimate crater diameters. The system significantly improves the speed and reliability of lunar geological analysis compared to traditional manual methods, offering a scalable solution for large-scale crater mapping.
Features

Preprocessing: Noise reduction and image enhancement for OHRC images.
Crater Detection: YOLOv9 model trained on annotated lunar images for accurate crater identification.
Diameter Estimation: Post-processing using bounding box dimensions and scaling to calculate crater diameters.
Evaluation Metrics: Precision, recall, and Intersection over Union (IoU) for model performance assessment.
Applications: Supports lunar surface evolution studies, impact history analysis, and mission planning.

Prerequisites

Hardware: GPU recommended (e.g., NVIDIA CUDA-compatible GPU for faster training).
Software:
Python 3.8+
PyTorch (for YOLOv9 implementation)
OpenCV (for image preprocessing)
NumPy, Pandas (for data handling)
Matplotlib (for visualization)


Dataset: Annotated lunar images (e.g., OHRC dataset or similar). Note: The dataset used in this project is not publicly available in this repository due to proprietary restrictions. Users must provide their own dataset or contact the relevant authority (e.g., ISRO for OHRC data).

Installation

Clone the Repository:
git clone https://github.com/your-username/lunar-crater-detection.git
cd lunar-crater-detection


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download YOLOv9 Weights:

Download pretrained YOLOv9 weights from the official repository or train the model from scratch (see Training section).
Place the weights in the weights/ directory.



Dataset Preparation

Format: Images and annotations should follow the YOLO format (e.g., .txt files with bounding box coordinates).
Directory Structure:dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── dataset.yaml  # Configuration file for dataset paths


Update dataset.yaml with paths to your training, validation, and test sets.

Usage

Preprocessing:Run the preprocessing script to enhance OHRC images:
python preprocess.py --input_dir dataset/images --output_dir dataset/preprocessed


Inference:Use the trained model to detect craters and estimate diameters:
python detect.py --weights weights/yolov9.pt --source dataset/preprocessed --output results/


Diameter Estimation:Post-process detection results to calculate crater diameters:
python estimate_diameter.py --results results/ --scale_factor 0.1  # Adjust scale factor based on image resolution


Evaluation:Evaluate model performance using precision, recall, and IoU:
python evaluate.py --results results/ --ground_truth dataset/labels



Training
To train the YOLOv9 model on your dataset:

Ensure the dataset is prepared (see Dataset Preparation).
Run the training script:python train.py --data dataset.yaml --weights yolov9.pt --epochs 50 --batch-size 16


Trained weights will be saved in the runs/train/ directory.

Results

Detection Accuracy: High precision and recall for crater detection, surpassing traditional methods.
Diameter Estimation: Accurate measurements with minimal error, validated against ground truth.
Speed: Processes large datasets significantly faster than manual analysis.
Example outputs are saved in the results/ directory, including visualizations of detected craters and estimated diameters.

Directory Structure
lunar-crater-detection/
├── dataset/              # Dataset directory (images, labels, dataset.yaml)
├── weights/              # Pretrained and trained model weights
├── results/              # Output directory for detection and diameter estimation
├── preprocess.py         # Script for image preprocessing
├── detect.py             # Script for crater detection
├── estimate_diameter.py  # Script for diameter estimation
├── evaluate.py           # Script for model evaluation
├── train.py              # Script for model training
├── requirements.txt      # Python dependencies
└── README.md             # This file

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Chandrayaan-2 OHRC: For providing high-resolution lunar imagery.
YOLOv9 Developers: For the robust object detection framework.
Planetary Science Community: For inspiration and validation datasets.

Contact
For questions or collaborations, please contact [your-email@example.com] or open an issue on GitHub.

This project is a proof-of-concept for automated lunar crater analysis and is intended for research purposes.
