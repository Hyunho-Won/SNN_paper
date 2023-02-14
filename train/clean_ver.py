import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split  # To make validation set
import torch
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn
import matplotlib.pyplot as plt
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sensor_size = tonic.datasets.NMNIST.sensor_size

frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=10000000)
                                      ])

trainset = tonic.datasets.NMNIST(save_to='/home/tmxk503/PycharmProjects/whh/data/NMNIST', transform=frame_transform,
                                 train=True)
testset = tonic.datasets.NMNIST(save_to='/home/tmxk503/PycharmProjects/whh/data/NMNIST', transform=frame_transform,
                                train=False)

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

batch_size = 32
dataset_size = len(trainset)
train_size = int(dataset_size * 0.9)
validation_size = int(dataset_size * 0.1)

trainset, validationset = random_split(trainset, [train_size, validation_size])

trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
valloader = DataLoader(validationset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

# test
event_tensor, target = next(iter(trainloader))
print(event_tensor.shape)
print(len(trainloader))
# neuron and simulation parameters
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

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            input_torch = data[step]
            # print(input_torch)
            out = self.layer1(input_torch)
            out1 = out

            out = self.layer2(out)
            out2 = out

            out, mem = self.layer4(out)

            layer1_rec.append(out1)
            layer2_rec.append(out2)
            spk_rec.append(out)
        return torch.stack(spk_rec), torch.stack(layer1_rec), torch.stack(layer2_rec)

model = CNN().to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=0.005, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 2

loss_hist = []
acc_hist = []
v_acc_hist = []
val_cnt = 0
v_acc_sum = 0
start = time.time()
avg_loss = 0
index = 0
# training loop
for epoch in range(num_epochs):
    # torch.save(model.state_dict(), '/home/tmxk503/PycharmProjects/whh/result/CSNN.pt')
    for i, (data, targets) in enumerate(iter(trainloader)):  # iter수만큼 반복
        data = data.to(device)
        targets = targets.to(device)

        model.train()
        spk_rec, h1, h2 = model(data)  # 현호
        loss_val = loss_fn(spk_rec, targets)
        avg_loss += loss_val.item()  # 현호
        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # 여기부터 현호
        val_cnt = val_cnt + 1

        if val_cnt == len(trainloader)/2-1:
            print("validation")
            val_cnt = 0
            v_acc_sum = 0
            cnt = 0
            for ii, (v_data, v_targets) in enumerate(iter(valloader)):
                v_data = v_data.to(device)
                v_targets = v_targets.to(device)
                v_spk_rec, h1, h2 = model(v_data)
                v_acc = SF.accuracy_rate(v_spk_rec, v_targets)
                v_acc_sum += v_acc
                cnt += 1
            plt.plot(acc_hist)
            plt.plot(v_acc_hist)
            plt.legend(['train accuracy', 'validation accuracy'])
            plt.title("Train, Validation Accuracy-CSNN")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.show()
            plt.savefig('CSNN.png')
            plt.clf()
            v_acc_sum = v_acc_sum / cnt
            v_acc_hist.append(v_acc_sum)
            avg_loss = avg_loss / (len(trainloader) / 2)
            print('average loss while half epoch:', avg_loss)
            if avg_loss <= 0.5:
                index = 1
                break
            else:
                avg_loss = 0
                index = 0
        # 여기까지 현호

        print("time :", time.time() - start, "sec")
        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        # accuracy
        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%")

        # validation accuracy
        print(f"Validation Accuracy: {v_acc_sum * 100:.2f}%\n")

        # s1 = torch.sum(h1)
        # print(s1)

        if index == 1:
            torch.save(model.state_dict(), '/home/tmxk503/PycharmProjects/whh/result/CSNN.pt')
            break
    if index == 1:
        break


