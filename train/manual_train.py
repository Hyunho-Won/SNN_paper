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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.ToFrame(sensor_size=sensor_size, time_window=1)


frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=10000)
                                     ])

trainset = tonic.datasets.NMNIST(save_to='/home/hubo1024/PycharmProjects/snntorch/data/NMNIST', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./home/hubo1024/PycharmProjects/snntorch/data/NMNIST', transform=frame_transform, train=False)

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

#batch_size = 100

batch_size = 32
dataset_size = len(trainset)
train_size = int(dataset_size * 0.9)
validation_size = int(dataset_size * 0.1)


trainset, valset = random_split(trainset, [train_size, validation_size])
print(len(valset))
print(len(trainset))
trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())


spike_grad = surrogate.fast_sigmoid(slope=75)
beta = 0.5

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(2, 12, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True))

        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(12, 32, 5),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True))

        # L4 FC 4x4x128 inputs -> 625 outputs

        self.layer4 = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True))
        # L5 Final FC 625 inputs -> 10 outputs

    def forward(self, data):
        spk_rec = []
        layer1_rec = []
        layer2_rec = []
        utils.reset(self.layer1)  # resets hidden states for all LIF neurons in net
        utils.reset(self.layer2)
        utils.reset(self.layer4)

        for step in range(data.size(1)):  # data.size(0) = number of time steps
            input_torch = data[:, step, :, :, :]
            input_torch = input_torch.cuda()
            #print(input_torch)
            out = self.layer1(input_torch)
            out1 = out

            out = self.layer2(out)
            out2 = out
            out, mem = self.layer4(out)

            spk_rec.append(out)

            layer1_rec.append(out1)
            layer2_rec.append(out2)

        return torch.stack(spk_rec), torch.stack(layer1_rec), torch.stack(layer2_rec)
# CNN 모델 정의

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
model = CNN().to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.005,betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
#model = nn.DataParallel(model)

total_batch = len(trainloader)
print('총 배치의 수 : {}'.format(total_batch))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
num_epochs = 15
loss_hist = []
acc_hist = []
v_acc_hist = []
t_spk_rec_sum = []
start = time.time()
val_cnt = 0
v_acc_sum= 0
avg_loss = 0
index = 0
#################################################



for epoch in range(num_epochs):
    torch.save(model.state_dict(), '/home/hubo1024/PycharmProjects/snntorch/model_pt/Nadam_05loss-10000.pt')
    for i, (data, targets) in enumerate(iter(trainloader)):
        data = data.cuda()
        targets = targets.cuda()

        model.train()
        spk_rec, h1, h2 = model( data)
        #print(spk_rec.shape)
        loss_val = loss_fn(spk_rec, targets)
        avg_loss += loss_val.item()
        # Gradient calculation + weight update
        optimizer.zero_grad()

        loss_val.backward()
        optimizer.step()
        #print(spk_rec.shape)
        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        val_cnt = val_cnt+1

        if val_cnt == len(trainloader)/2-1:
            val_cnt=0
            torch.save(model.state_dict(), '/home/hubo1024/PycharmProjects/snntorch/model_pt/Nadam_05loss-10000.pt')
            for ii, (v_data, v_targets) in enumerate(iter(valloader)):
                v_data = v_data.to(device)
                v_targets = v_targets.to(device)

                v_spk_rec, h1, h2 = model(v_data)
                # print(t_spk_rec.shape)
                v_acc = SF.accuracy_rate(v_spk_rec, v_targets)
                if ii == 0:
                    v_acc_sum = v_acc
                    cnt = 1

                else:
                    v_acc_sum += v_acc
                    cnt += 1
            plt.plot(acc_hist)
            plt.plot(v_acc_hist)
            plt.legend(['train accuracy', 'validation accuracy'])
            plt.title("Train, Validation Accuracy-Nadam_05loss-10000")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            # plt.show()
            plt.savefig('Nadam_05loss-10000.png')
            plt.clf()
            v_acc_sum = v_acc_sum/cnt

            avg_loss = avg_loss / (len(trainloader) / 2)
            print('average loss while half epoch:', avg_loss)
            if avg_loss <= 0.5:
                index = 1
                break
            else:
                avg_loss = 0
                index = 0

        print('Nadam_05loss-10000')
        print("time :", time.time() - start,"sec")
        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        v_acc_hist.append(v_acc_sum)
        print(f"Train Accuracy: {acc * 100:.2f}%")
        print(f"Validation Accuracy: {v_acc_sum * 100:.2f}%\n")
        if index == 1:
            torch.save(model.state_dict(), '/home/hubo1024/PycharmProjects/snntorch/model_pt/Nadam_05loss-10000.pt')
            break
    if index == 1:
        break
