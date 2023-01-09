# Introduction
Recently, research on SNN that are expected to have higher energy efficiency than conventional DNN has increased.
SNN is expected to be efficient in processing event data, which is sparse spatio-temporal data.
To utilize event data on SNN, pre-processing is required and it has a profound effect on SNN learning. So it is necessary to investigate the effect of the pre-processing method of event data on the learning performance of SNN.
In this paper, we analyze the change of learning performance according to the event accumulation method and time interval(Ti), which is major variable of voxel grid-based preprocessing.

# SNN
To design SNN accelerator, We had to study SNN firts.
And here is the result of our short study 
[An Analysis of Preprocessing Effect on SNN Training with Event Data]


# Voxel grid

# code
```py
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import os
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import tonic.transforms as transforms
import tonic
import numpy as np
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn
import os
from torch.utils.data import DataLoader, random_split
import torch
import logging
```
