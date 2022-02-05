import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

def get_mnist(binary_labels):
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
    )
    mnist = MNIST(root='./_data', transform=mnist_transform, download=True)
    bin_x = torch.stack([
        x for x, y in mnist if y in binary_labels[0]+binary_labels[1]
    ])
    bin_x = torch.cat([
        bin_x.reshape(bin_x.shape[0], -1), torch.ones((bin_x.shape[0], 1))
    ], dim=1)
    bin_y = torch.stack([
        torch.tensor(y in binary_labels[0]).long() for x, y in mnist if y in binary_labels[0]+binary_labels[1]
    ])
    mnist = data.TensorDataset(bin_x, bin_y)
    return mnist

def split_mnist(mnist, data_seed, train_size, val_size=12000):
    generator = torch.Generator().manual_seed(data_seed)
    val_size = 12000
    train_set, _, val_set = data.random_split(
        mnist, [train_size, len(mnist)-(train_size+val_size), val_size], generator=generator
    )
    return train_set, val_set

def get_mnist_dataloader(mnist, data_seed, train_size, val_size=12000, batch_size=128):
    print(train_size)
    generator = torch.Generator().manual_seed(data_seed)
    train_set, val_set = split_mnist(mnist, data_seed, train_size, val_size=val_size)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator, num_workers=2)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader
