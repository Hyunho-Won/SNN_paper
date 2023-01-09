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
We used SNN-torch to construct and train SNN   
github repository: https://github.com/jeshraghian/snntorch

# Code 
Use tonic.transforms package to make preprocess filter.
Using a filter, preprocess the datasets. 
```py
import tonic.transforms as transforms

sensor_size = tonic.datasets.NMNIST.sensor_size

frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=10000)
                                     ])

trainset = tonic.datasets.NMNIST(save_to='User_Path', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='.User_Path', transform=frame_transform, train=False)
```
      
Define CSNN model
```py
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
 
```

Train model with surrogate gradient descent. 
```py
for epoch in range(num_epochs):
    torch.save(model.state_dict(), 'User_Path.pt')
    for i, (data, targets) in enumerate(iter(trainloader)):
        data = data.cuda()
        targets = targets.cuda()

        model.train()
        spk_rec, h1, h2 = model( data)

        loss_val = loss_fn(spk_rec, targets)
        avg_loss += loss_val.item()

        optimizer.zero_grad()

        loss_val.backward()
        optimizer.step()


        loss_hist.append(loss_val.item())
        val_cnt = val_cnt+1

        if val_cnt == len(trainloader)/2-1:
            val_cnt=0
            torch.save(model.state_dict(), 'User_Path.pt')
            for ii, (v_data, v_targets) in enumerate(iter(valloader)):
                v_data = v_data.to(device)
                v_targets = v_targets.to(device)

                v_spk_rec, h1, h2 = model(v_data)

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
            plt.title("Title")
            plt.xlabel("Iteration")
            plt.ylabel("Accuracy")
            plt.savefig('User_Path.png')
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
            torch.save(model.state_dict(), 'User_Path.pt')
            break
    if index == 1:
        break
```



