import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torch import nn
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
batch_size = 512
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(torch.flatten)]
)

train_data = torchvision.datasets.MNIST(
    "./datasets/", download=True, transform=transform
)
val_data = torchvision.datasets.MNIST(
    "./datasets/", train=False, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=batch_size
)
val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)


class DeepMLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inlin = nn.Linear(28 * 28, 512)
        self.lin1 = nn.Linear(512, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.inlin(x))
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.lin4(x)


deep_mlp = DeepMLP()
torchvision_mlp = torchvision.ops.MLP(
    28 * 28, [512, 512, 10], activation_layer=torch.nn.ReLU, bias=True, dropout=0.3
)
shallow_mlp = nn.Sequential(nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 10))
super_deep_mlp = torchvision.ops.MLP(
    28 * 28,
    [512, 512, 512, 512, 512, 10],
    activation_layer=torch.nn.ReLU,
    bias=True,
    dropout=0.0,
)

loss_fn = nn.CrossEntropyLoss()


def train(model, epochs, lr):
    model.to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    last_vloss = 0.0
    for epoch in tqdm(range(epochs)):
        last_loss = 0.0

        model.train(True)

        for ix, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            last_loss += loss.item()
        print(f"Last Epoch avg Loss: {last_loss/len(train_data)}")

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            last_vloss += vloss
    print(f"Final Epoch avg Val Loss: {last_vloss/len(val_data)}")


def accuracy(model):
    model.eval()
    model.to("cpu")
    hits = 0
    for i, vdata in enumerate(val_loader):
        vinputs, vlabels = vdata
        voutputs = F.softmax(model(vinputs), dim=0)
        hits += (torch.argmax(voutputs, dim=1) == vlabels).sum()
    print(f"Accuracy: {hits/len(val_data)}")


def plot_results(n, model):
    model.eval()
    model.to("cpu")
    fig = plt.figure()
    rows = 1
    columns = n
    for i in range(n):
        fig.add_subplot(rows, columns, i + 1)
        with torch.no_grad():
            img = val_data[i][0]
            plt.title(str(torch.argmax(model(img)).item()))
            plt.imshow(img.view((28, 28, 1)))
            plt.axis("off")
    plt.show()


model = super_deep_mlp
train(model=model, epochs=100, lr=0.005)
accuracy(model)
plot_results(5, model)

