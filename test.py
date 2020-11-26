import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, HW2Dataset, CSVDataset, collater, Resizer_test, AspectRatioBasedSampler, Augmenter, \
    Normalizer_test, HW2Dataset_test
from torch.utils.data import DataLoader
from retinanet import coco_eval
from retinanet import csv_eval
import os
import skimage.io
import skimage.transform
import skimage.color
import skimage

#### import model
retinanet = torch.load('saved_models_3/HW2_retinanet_20.pt')
retinanet = retinanet.cuda()
retinanet.eval()

dataset_test = HW2Dataset_test("../test/", transform=transforms.Compose([Normalizer_test(), Resizer_test()]))

coco_eval.evaluate_HW2(dataset_test, retinanet)
