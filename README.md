# Rock Paper Scissors Image Classification

## Description:
This project demonstrates an image classification model for rock-paper-scissors hand gestures. I received a scholarship from the DBS Foundation Coding Camp 2024 X Dicoding, where I learned about the topic 'Learning Machine Learning for Beginners' with a final project on image classification. See [my certificate](https://www.dicoding.com/certificates/MEPJY86L4P3V).

## Features:
- Downloading and extracting dataset
- Preprocessing images (resizing, normalization)
- Data augmentation
- Building and training a CNN model
- Evaluating model performance
- Saving evaluation results

## Screenshot
![Untitled](https://github.com/aprayogaaa/rock-papper-scissor-image-classification/assets/70948216/b047ae78-b2ab-4a77-b758-0302c89743e2)


## Usage:
1. Clone the repository:
```bash
git clone https://github.com/yourusername/rock-paper-scissors-classification.git
```
```bash
cd rock-paper-scissors-classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
python ```main.py```

## Requirements:
- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- numpy
- OpenCV

## Additional Information:
- Check the evaluation results in the Result/MetricsEvaluation folder.
- Check the augmantation image in the Result/AugmantationData folder.

## License:
This dataset is provided under the Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0).

AUTHOR:
Julien de la Bruère-Terreault
Email: drgfreeman@tuta.io

DESCRIPTION:
This dataset contains images of hand gestures from the Rock-Paper-Scissors game. The images were captured as part of a hobby project where I developed a Rock-Paper-Scissors game using computer vision and machine learning on the Raspberry Pi. You can find more information about the project on GitHub: https://github.com/DrGFreeman/rps-cv

CONTENTS:
The dataset contains a total of 2188 images corresponding to the 'Rock' (726 images), 'Paper' (710 images), and 'Scissors' (752 images) hand gestures of the Rock-Paper-Scissors game. All images are taken on a green background with relatively consistent lighting and white balance.

FORMAT:
All images are RGB images of 300 pixels wide by 200 pixels high in .png format. The images are separated into three sub-folders named 'rock', 'paper', and 'scissors' according to their respective class.

