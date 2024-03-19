import torch
import torchvision
import torch.utils.data
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')
batch_size = 512
transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(torch.flatten)])

train_data = torchvision.datasets.MNIST('./datasets/',download=True,transform=transform)
val_data = torchvision.datasets.MNIST('./datasets/',train=False,download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_data,shuffle=False,batch_size=batch_size)

torchvision_mlp = torchvision.ops.MLP(28*28,[512,512,10],activation_layer=torch.nn.ReLU,bias=True,dropout=0.3).to(device)
print(torchvision_mlp)
lr = 0.001

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(torchvision_mlp.parameters(),lr,momentum=0.9)

def train(model,epochs):
    for epoch in tqdm(range(epochs)):
        last_loss = 0.
        last_vloss = 0.

        model.train(True)

        for ix, data in enumerate(train_loader):
            inputs, labels = data
            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()


        if epoch % 10 ==0:
            model.eval()

            with torch.no_grad():
                for i,vdata in enumerate(val_loader):
                    vinputs,vlabels = vdata
                    vinputs,vlabels = vinputs.to(device),vlabels.to(device)
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs,vlabels)
                    last_vloss = vloss

            print(f'Last Loss: {last_loss}')
            print(f'Last Val Loss: {last_vloss}')
train(torchvision_mlp,51)

def plot_results(n,model):
    fig = plt.figure()
    rows =1 
    columns = n
    for i in range(n):
        fig.add_subplot(rows,columns,i+1)
        model.eval()
        model.to('cpu')
        with torch.no_grad():
            img = val_data[i][0]
            plt.title(str(torch.argmax(model(img)).item()))
            plt.imshow(img.view((28,28,1)))
            plt.axis('off')
    plt.show()
plot_results(5,torchvision_mlp)
