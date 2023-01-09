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
A three-dimensional structure in which temporal information of event data is recorded.

![화면 캡처 2023-01-09 152823](https://user-images.githubusercontent.com/122242141/211251530-73c864bf-b71a-4d79-af8b-e55848ed63ba.png)

![화면 캡처 2023-01-09 153416](https://user-images.githubusercontent.com/122242141/211252154-6e520211-a01e-4d8d-b12d-af4fc4bfa80c.png)


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
