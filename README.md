# Introduction
Recently, research on SNN that are expected to have higher energy efficiency than conventional DNN has increased.
SNN is expected to be efficient in processing event data, which is sparse spatio-temporal data.
To utilize event data on SNN, pre-processing is required and it has a profound effect on SNN learning. So it is necessary to investigate the effect of the pre-processing method of event data on the learning performance of SNN.
In this paper, we analyze the change of learning performance according to the event accumulation method and time interval(T𝑖), which is major variable of voxel grid-based preprocessing.

# Spiking Neural Network(SNN)
A neural network that mimics the mechanism by which neurons in the biological brain determine output.

<img src = "https://user-images.githubusercontent.com/122242141/211256745-3a4c85e5-8c4b-492f-b2e7-5c755ff6b43c.png" width="70%" height="70%">
(Figure: "Enabling Spike-Based Backpropagation for Training Deep Neural Network Architectures", Chankyu Lee, Syed Shakib Sarwar, Priyadarshini Panda, Gopalakrishnan Srinivasan, Kaushik Roy)

# Event data
Event camera: Unlike conventional cameras, images are not captured in fixed frame units, but are asynchronously measured and recorded in brightness between pixels.

# Pre-processing
Voxel grid: A three-dimensional structure in which temporal information of event data is recorded.




### Pre-processing process based on Voxel grid



1. As shown in Equation (1), the time step length of raw event data is divided into T𝑖 units to forms event data into groups of output time steps(Tt').

<img src = "https://user-images.githubusercontent.com/122242141/211255110-55c0ea00-9878-4023-810b-f3b93555219b.png" width="20%" height="20%">



2. Event data of each group is accumulated and output based on the time (t) axis.

<img src = "https://user-images.githubusercontent.com/122242141/211251530-73c864bf-b71a-4d79-af8b-e55848ed63ba.png" width="80%" height="80%">


### Methods of accumulating event data based on the t-axis.
(a) Accumulate event data per pixel



(b) Event data accumulated per pixel and binarized



(c) Event data accumulated per pixel and normalized

<img src = "https://user-images.githubusercontent.com/122242141/211252154-6e520211-a01e-4d8d-b12d-af4fc4bfa80c.png" width="70%" height="70%">


# Enviroments
download requirements.txt
than execute
```py
$ pip install -r requirements.txt
```
