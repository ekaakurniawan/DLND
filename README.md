# DLND
Deep Learning Nanodegree from Udacity

## Contents
### Assignments
#### Part 3: Convolutional Neural Networks
 - Lecture 3: Transfer Learning
    - [Transfer Learning](./assignments/P3-CNN/L3-transfer-learning/Transfer_Learning_Exercise.ipynb)
 - Lecture 4: Weight Initialization
    - [Weight Initialization](./assignments/P3-CNN/L4-weight-initialization/weight_initialization_exercise.ipynb)
 - Lecture 5: Autoencoder
    - [Simple Autoencoder](./assignments/P3-CNN/L5-autoencoder/Simple_Autoencoder_Exercise.ipynb)
    - [Convolutional Autoencoder](./assignments/P3-CNN/L5-autoencoder/Convolutional_Autoencoder_Exercise.ipynb)
    - [Denoising Autoencoder](./assignments/P3-CNN/L5-autoencoder/Denoising_Autoencoder_Exercise.ipynb)
#### Part 4: Recurrent Neural Networks
 - Lecture 5: Embeddings & Word2Vec
    - [Skip Grams](./assignments/P4-RNN/L5-embeddings-word2vec/Skip_Grams_Exercise.ipynb)
 - Lecture 8: Attention
    - [Attention Basics](./assignments/P4-RNN/L8-attention/Attention%20Basics.ipynb)
#### Part 5: Generative Adversarial Networks
 - Lecture 1: Basic GAN
    - [MNIST GAN](./assignments/P5-GAN/L1-generative-adversarial-networks/MNIST_GAN_Exercise.ipynb)
 - Lecture 2: Deep Convolutional GANs
    - [Batch Normalization](./assignments/P5-GAN/L2-deep-convolutional-gans/batch-norm/Batch_Normalization.ipynb)
    - [Generating Street View Home Number (SVHN)](./assignments/P5-GAN/L2-deep-convolutional-gans/dcgan-svhn/DCGAN_Exercise.ipynb): Only for CPU and CUDA. XPU has memory leak issue.
 - Lecture 4: CycleGAN
    - [Summer to Winter Image Transformation](./assignments/P5-GAN/L4-implementing-a-cyclegan/CycleGAN_Exercise.ipynb): Only for CPU and CUDA. XPU has memory leak issue.
#### Part 6: Deploying Model
Inference on CPU only (not using XPU or CUDA).
 - Lecture 2: Building a Model using SageMaker
    - [Predicting Boston Housing - XGBoost - Batch Transform - High Level](./assignments/P6-deploying-model/L2-building-a-model-using-sagemaker/Boston%20Housing%20-%20XGBoost%20(Batch%20Transform)%20-%20High%20Level.ipynb)
    - [Sentiment Analysis - XGBoost - Batch Transform](./assignments/P6-deploying-model/L2-building-a-model-using-sagemaker/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Batch%20Transform).ipynb)
 - Lecture 3: Deploying and Using a Model
    - [Predicting Boston Housing - XGBoost - Deploy - High Level](./assignments/P6-deploying-model/L2-building-a-model-using-sagemaker/Boston%20Housing%20-%20XGBoost%20(Deploy)%20-%20High%20Level.ipynb)
    - [Sentiment Analysis - XGBoost - Web App](./assignments/P6-deploying-model/L2-building-a-model-using-sagemaker/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20-%20Web%20App.ipynb)
 
### Projects
 - [Project 1: Predicting Bike-Sharing Patterns](./P1-Predicting-Bike-Sharing-Patterns/Your_first_neural_network.ipynb)
 - [Project 2: Classifying Dog's Breed](./P2-Dog-Classification/dog_app.ipynb)
 - [Project 4: Generating TV Script](./P4-Generating-TV-Script/dlnd_tv_script_generation.ipynb)
 - [Project 5: Generating Face](./P5-Generating-Face/dlnd_face_generation.ipynb): Only for CPU and CUDA. XPU has memory leak issue.
 - [Project 7: Deploying Sentiment Analysis Model](./P7-Deploying-Sentiment-Analysis-Model/SageMaker%20Project.ipynb): Only for CPU and CUDA. SageMaker does not support XPU.

## Setup

### Intel GPU

Although we are going to use PyTorch 2.7 nightly with 
memory leak issue fixed, please follow 
[PyTorch 2.6 Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-6.html)
article to install Intel GPU driver and deep learning essentials.

Tested on the following hardware specification and software version.

__Hardware Specification__
 - CPU: Intel® Core™ Ultra 9 Processor 285K
 - CPU Cores: 24 (8 Performance-cores and 16 Efficient-cores)
 - CPU Threads: 24
 - Memory: 32 GiB
 - GPU: Intel® Arc™ A770 Graphics
 - GPU Memory: 16 GiB
 
__Software Version__
 - Ubuntu 24.04.1 LTS
 - Intel Deep Learning Essentials 2025.0.1-25
 - Python 3.12.3
 - PyTorch 2.7.0.dev20250214+xpu
 - TorchVision 0.22.0.dev20250214+xpu
 - opencv-python 4.11.0.86
 - NumPy 2.1.2
 - Matplotlib 3.10.0
 - Pandas 2.2.3
 - scikit-learn 1.6.1

### Install Requirements

Create virtual environment.
```
$ python3 -m venv pytorch_arc_env
$ source pytorch_arc_env/bin/activate
$ python -m pip install --upgrade pip
```

Install PyTorch and other required packages.
```
$ pip install --pre torch==2.7.0.dev20250214+xpu torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
$ cd DLND
$ pip install --upgrade -r requirements.txt
```

### Test Run

Activate virtual environment and setup variables.
```
$ source pytorch_arc_env/bin/activate

$ source /opt/intel/oneapi/compiler/2025.0/env/vars.sh
$ source /opt/intel/oneapi/umf/0.9/env/vars.sh
$ source /opt/intel/oneapi/pti/0.10/env/vars.sh
```

Detect GPU.
```
$ python -c "import torch; print(torch.xpu.is_available())"
```
```
True
```

Run notebooks.
```
$ cd DLND
$ jupyter lab
```
