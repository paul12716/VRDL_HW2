# VRDL_HW2

code for Selected Topics in Visual Recognition using Deep Learning Homework 2

## Hardware

- Ubuntu 16.04 LTS
- NVIDIA 1080ti

## Installation

1. Clone this repo
2. Install the required packages:
```bach
apt-get install tk-dev python-tk
```
3. Install the python packages:
```bach
pip install pandas
pip install pycocotools
pip install opencv-python
pip install requests
```
4. Train model by running Train.py:
5. Output predicted result by running Test.py and upload to Google drive submission

## Training
The network can be trained using the train.py script. We use HW2 as our dataset.
```bash
python train.py --dataset HW2 --HW2_path ../train --depth 50
```

In Train.py, we preprocess the data and train our model.

## Pre-trained model
A pre-trained model is available at:
- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS 
The state dict model can be loaded using:
```bash
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```

### Data preprocess

We first read training_labels.csv to make a dictionary that correspond a car's class name to a simple int.  
For example : Ford F-150 Regular Cab 2007 correspond to 108.  

Then we do image preprocessing in the following order.
- Normalize
- Augmentation
- Resize

After preprocessing, we use dataloader with batch_size=10 to feed training_data into our model.

### Train our model

```bach
python train.py --dataset HW2 --HW2_path ../train --depth 50
```

We use RetinaNet as backbone network, and modify the class numbers to 10.  
10 is the total class numbers of our training_data.  

We use focalLoss as loss function and Adam optimizer, and save model parameters every single epochs in total 100 epochs.  
Set learning rate = 1e-4  
The model parameters would be saved in folder saved_models_2/

## Test accuracy by running test.py

```bach
python test.py
```

For testing data, we first use simliar dataloader to load testing images.  
Then we import the saved model.
```python
retinanet = torch.load('saved_models_2/HW2_retinanet_xx.pt')
retinanet = retinanet.cuda()
retinanet.eval()
```

For each testing image, we normalize and resize it.  
In the end, write image id and predicted result to a .josn file.  
Upload google drive submission.  
Best accuracy : .
