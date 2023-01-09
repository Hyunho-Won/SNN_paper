# SNN Hardware Accelerator Project
This project involves designing a hardware accelerator that can efficiently perform SNN operations.

# Part 1
# Studying SNN
To design SNN accelerator, We had to study SNN firts.
And here is the result of our short study 
[An Analysis of Preprocessing Effect on SNN Training with Event Data]


# Introduction

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
