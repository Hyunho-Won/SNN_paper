# An Analysis of Preprocessing Effect on SNN Training with Event Data

GitHub repository:https://github.com/whh0503/thesis

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
