import os
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def get_transform(dataset_name, mode):
    if mode == 'rgb':
        if dataset_name == 'CIFAR100':
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        elif dataset_name == 'CIFAR10':
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif dataset_name in ['MNIST', 'FashionMNIST']:
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif dataset_name == 'SVHN':
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        elif dataset_name == 'TinyImageNet':
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
            ])
    elif mode == 'grey':
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])


class CIFAR100CDataset(Dataset):
    def __init__(self, corruption_type, data_path, transform=None):
        self.data_path = os.path.join(data_path, 'CIFAR-100-C')
        self.corruption_type = corruption_type
        self.transform = transform
        self.images = np.load(os.path.join(self.data_path, f"{corruption_type}.npy"))
        self.labels = np.load(os.path.join(self.data_path, "labels.npy"))
        assert self.images.shape[0] == self.labels.shape[0], "Mismatch between images and labels."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



class CIFAR10CDataset(Dataset):
    def __init__(self, corruption_type, data_path, transform=None):
        self.data_path = os.path.join(data_path, 'CIFAR-10-C')
        self.corruption_type = corruption_type
        self.transform = transform
        self.images = np.load(os.path.join(self.data_path, f"{corruption_type}.npy"))
        self.labels = np.load(os.path.join(self.data_path, "labels.npy"))
        assert self.images.shape[0] == self.labels.shape[0], "Mismatch between images and labels."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_dataset(dataset_name, dataset_type, mode='rgb', batch_size_train=32, batch_size_test=32, drop_last=False, data_path='/path/to/data'):
    transform = get_transform(dataset_name, mode)

    if dataset_name == 'CIFAR100':
        if dataset_type == 'train':
            train_dataset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
            val_size = int(0.1 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=drop_last, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return train_loader, val_loader
        elif dataset_type == 'test':
            test_dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return test_loader
        elif dataset_type == 'ood':
            ood_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN', 'TinyImageNet']
            ood_loaders = {}
            for dset in ood_datasets:
                ood_transform = get_transform(dset, 'rgb')
                if dset == 'SVHN':
                    dataset = datasets.SVHN(root=data_path, split='test', download=True, transform=ood_transform)
                elif dset == 'TinyImageNet':
                    tiny_imagenet_path = os.path.join(data_path, 'tiny-imagenet/tiny-imagenet-200')
                    dataset = datasets.ImageFolder(root=os.path.join(tiny_imagenet_path, 'val'), transform=ood_transform)
                else:
                    dataset = datasets.__dict__[dset](root=data_path, train=False, download=True, transform=ood_transform)
                ood_loaders[dset] = DataLoader(dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return ood_loaders
        elif dataset_type == 'corrupted':
            corruption_types = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
            loaders = {}
            for corruption in corruption_types:
                cifar100_c_dataset = CIFAR100CDataset(corruption, data_path, transform=transform)
                loaders[corruption] = DataLoader(cifar100_c_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return loaders

    elif dataset_name == 'CIFAR10':
        if dataset_type == 'train':
            train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
            val_size = int(0.1 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=drop_last, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return train_loader, val_loader
        elif dataset_type == 'test':
            test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return test_loader
        elif dataset_type == 'ood':
            ood_datasets = ['MNIST', 'FashionMNIST', 'CIFAR100', 'SVHN', 'TinyImageNet']
            ood_loaders = {}
            for dset in ood_datasets:
                ood_transform = get_transform(dset, 'rgb')
                if dset == 'SVHN':
                    dataset = datasets.SVHN(root=data_path, split='test', download=True, transform=ood_transform)
                elif dset == 'TinyImageNet':
                    tiny_imagenet_path = os.path.join(data_path, 'tiny-imagenet/tiny-imagenet-200')
                    dataset = datasets.ImageFolder(root=os.path.join(tiny_imagenet_path, 'val'), transform=ood_transform)
                else:
                    dataset = datasets.__dict__[dset](root=data_path, train=False, download=True, transform=ood_transform)
                ood_loaders[dset] = DataLoader(dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return ood_loaders
        elif dataset_type == 'corrupted':
            corruption_types = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
            loaders = {}
            for corruption in corruption_types:
                cifar10_c_dataset = CIFAR10CDataset(corruption, data_path, transform=transform)
                loaders[corruption] = DataLoader(cifar10_c_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return loaders

    elif dataset_name == 'MNIST':
        if dataset_type == 'train':
            train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
            val_size = int(0.1 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=drop_last, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return train_loader, val_loader
        elif dataset_type == 'test':
            test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
            test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return test_loader
        elif dataset_type == 'ood':
            ood_datasets = ['FashionMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
            ood_loaders = {}
            for dset in ood_datasets:
                ood_transform = get_transform(dset, 'grey')
                if dset == 'SVHN':
                    dataset = datasets.SVHN(root=data_path, split='test', download=True, transform=ood_transform)
                elif dset == 'TinyImageNet':
                    tiny_imagenet_path = os.path.join(data_path, 'tiny-imagenet/tiny-imagenet-200')
                    dataset = datasets.ImageFolder(root=os.path.join(tiny_imagenet_path, 'val'), transform=ood_transform)
                else:
                    dataset = datasets.__dict__[dset](root=data_path, train=False, download=True, transform=ood_transform)
                ood_loaders[dset] = DataLoader(dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return ood_loaders

    elif dataset_name == 'FashionMNIST':
        if dataset_type == 'train':
            train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
            val_size = int(0.1 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, drop_last=drop_last, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return train_loader, val_loader
        elif dataset_type == 'test':
            test_dataset = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
            test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return test_loader
        elif dataset_type == 'ood':
            ood_datasets = ['MNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'TinyImageNet']
            ood_loaders = {}
            for dset in ood_datasets:
                ood_transform = get_transform(dset, 'grey')
                if dset == 'SVHN':
                    dataset = datasets.SVHN(root=data_path, split='test', download=True, transform=ood_transform)
                elif dset == 'TinyImageNet':
                    tiny_imagenet_path = os.path.join(data_path, 'tiny-imagenet/tiny-imagenet-200')
                    dataset = datasets.ImageFolder(root=os.path.join(tiny_imagenet_path, 'val'), transform=ood_transform)
                else:
                    dataset = datasets.__dict__[dset](root=data_path, train=False, download=True, transform=ood_transform)
                ood_loaders[dset] = DataLoader(dataset, batch_size=batch_size_test, shuffle=True, drop_last=drop_last, num_workers=2)
            return ood_loaders
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    
