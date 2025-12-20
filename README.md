# Vehicles & Animals Image Classification with CIFAR-10

This project implements a complete Convolutional Neural Network (CNN) image classification pipeline using PyTorch.  
The model is trained on the CIFAR-10 dataset and evaluated on real-world smartphone images to analyze generalization performance.

---

## Dataset
- **Standard Dataset:** CIFAR-10 (10 classes)
- **Classes Used:**  
  airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

- **Custom Dataset:**  
  Real-world smartphone images captured by the author and stored in `data/custom`.

---

## Project Structure


<img width="536" height="316" alt="image" src="https://github.com/user-attachments/assets/bff7c293-0c9d-4fba-b39d-17bf3eea60c1" />

---

## Model Architecture
- Convolutional Neural Network implemented using `torch.nn.Module`
- Key layers:
  - Convolution + ReLU
  - MaxPooling
  - Fully Connected layers
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

---

## Training
- Dataset automatically downloaded using `torchvision.datasets`
- Images preprocessed using `torchvision.transforms`
- Model trained on CIFAR-10 training set
- Training loss and accuracy visualized across epochs

---

## Evaluation & Results
- Confusion Matrix generated on CIFAR-10 test set
- Visual Error Analysis performed using misclassified samples
- Real-world testing conducted using smartphone images

For each custom image, the model outputs:
- Predicted class label
- Softmax confidence score

---

## Real-World Testing Observation
The model demonstrates strong performance on structured objects such as automobiles and trucks.  
Some confusion occurs between visually similar animal classes (e.g., bird and cat), highlighting domain shift and resolution limitations of CIFAR-10.

---

## How to Run (Google Colab)
1. Clone the repository:
   ```bash
   git clone https://github.com/Foysal348/Vehicles-Animals-Image-Classification-with-CIFAR-10.git
2. Open CNN_Image_Classification.ipynb in Google Colab

3. Select Runtime → Run all

No manual file uploads are required.
***Author***
**Foysal Emon Shanto**
