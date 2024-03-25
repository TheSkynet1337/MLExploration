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
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_data = torchvision.datasets.FashionMNIST(
    root="./datasets/", train=True, transform=transform, download=True
)
val_data = torchvision.datasets.FashionMNIST(
    root="./datasets/", train=False, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_data, shuffle=True, batch_size=batch_size
)
val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=batch_size)

seq_cnn = nn.Sequential(
    # in: N x 1 x 28 x28
    nn.Conv2d(
        in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1, bias=True
    ),
    nn.ReLU(),
    # in: N x 4 x 14 x 14
    nn.Conv2d(
        in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True
    ),
    nn.ReLU(),
    # in: N x 16 x 7 x 7
    nn.Flatten(),
    nn.Linear(784, 10),
)

loss_fn = nn.CrossEntropyLoss()


def train(model, epochs, lr):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001, lr=lr)
    last_vloss = 0.0
    for epoch in tqdm(range(epochs)):
        model.train(True)
        last_loss = 0.0
        for ix, data in enumerate(train_loader):
            ins, labels = data
            ins, labels = ins.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(ins)
            loss = loss_fn(outs, labels)
            loss.backward()
            optimizer.step()
            last_loss += loss.item()
        print(f"\n Last Epoch avg Loss:{last_loss/len(train_data)}")
    model.eval()
    with torch.no_grad():
        for _, vdata in enumerate(val_loader):
            vins, vlabels = vdata
            vins, vlabels = vins.to(device), vlabels.to(device)
            vouts = model(vins)
            vloss = loss_fn(vouts, vlabels)
            last_vloss += vloss
    print(f"\n Final Epoch avg Val Loss: {last_vloss/len(val_data)}")


def accuracy(model):
    model.eval()
    model.to("cpu")
    hits = 0
    for _, vdata in enumerate(val_loader):
        vins, vlabels = vdata
        vouts = nn.functional.softmax(model(vins), dim=0)
        hits += (torch.argmax(vouts, dim=1) == vlabels).sum()
    print(f"Accuracy: {hits/len(val_data)}")


human_readable_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def plot_results(n, model):
    model.eval()
    model.to("cpu")
    fig = plt.figure()
    rows = 1
    columns = n
    for i in range(n):
        fig.add_subplot(rows, columns, i + 1)
        with torch.no_grad():
            img = val_data[i][0].unsqueeze(0)
            plt.title(human_readable_labels[int(torch.argmax(model(img)).item())])
            plt.imshow(img.view((28, 28, 1)))
            plt.axis("off")
    plt.show()


model = seq_cnn
train(model, 100, 0.001)
accuracy(model)
plot_results(5, model)

