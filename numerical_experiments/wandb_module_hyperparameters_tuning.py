import numpy as np
import pprint

import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets

import models
from multi_output_module.multi_output_module import Multi_output_module


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved_model_state_dict ='./models/saved_models/fashion_mnist/fashion_mnist_lenet5.pth'

base_model = models.lenet.LeNet5().to(device)
#base_model = models.wide_resenet.Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10)

# Load the saved state dictionary
state_dict = torch.load(saved_model_state_dict)
base_model.load_state_dict(state_dict)
base_model = base_model.to(device)
base_model.eval()

my_secret = 'your_wandb_api_key_here'  # Replace with your actual WandB API key
wandb.login(key=my_secret)

'''
class CIFAR10Dataset(Dataset):
    def __init__(self, subset='train', data_path='/home/natcgx/Single-pass UQ research project/data', validation_split=0.2):
        super().__init__()
        self.subset = subset

        # Define the transformations for the CIFAR-10 dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Load the CIFAR-10 dataset
        full_train_dataset = datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform)
        train_size = int((1 - validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])

        if self.subset == 'train':
            self.dataset = self.train_dataset
        elif self.subset == 'val':
            self.dataset = self.val_dataset
        elif self.subset == 'test':
            self.dataset = datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform)
        else:
            raise Exception("subset must be 'train', 'val', or 'test'")

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = './data'
num_classes=10

# Load the CIFAR-10 training and validation datasets
#train_dataset = CIFAR10Dataset(subset='train', data_path=data_path)
#val_dataset = CIFAR10Dataset(subset='val', data_path=data_path)

class FashionMNISTDataset(Dataset):
    def __init__(self, subset='train', data_path='./data', validation_split=0.2):
        super().__init__()
        self.subset = subset

        # Define the transformations for the FashionMNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization based on FashionMNIST
        ])

        full_train_dataset = datasets.FashionMNIST(root=data_path, train=True, download=False, transform=transform)
        train_size = int((1 - validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])

        if self.subset == 'train':
            self.dataset = self.train_dataset
        elif self.subset == 'val':
            self.dataset = self.val_dataset
        elif self.subset == 'test':
            self.dataset = datasets.FashionMNIST(root=data_path, train=False, download=False, transform=transform)
        else:
            raise Exception("subset must be 'train', 'val', or 'test'")

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


train_dataset = FashionMNISTDataset(subset='train', data_path=data_path)
val_dataset = FashionMNISTDataset(subset='val', data_path=data_path)

def train_one_epoch(train_loader, module, loss, optimizer, num_heads):
    loss_epoch = 0
    
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Ensure labels are correctly shaped
        labels = labels.view(-1, num_heads)

        optimizer.zero_grad()
        predictions = module(imgs, 'training')

        total_loss = 0
        for i in range(num_heads):
            total_loss += loss(predictions[:, i, :], labels[:, i])
        total_loss /= num_heads

        total_loss.backward()

        optimizer.step()

        # Track loss
        loss_epoch += total_loss.detach().item()
        
    return loss_epoch/len(train_loader)


# Define the Brier Score function for multi-class classification
def brier_score_multi_class(y_true, y_prob, num_classes):
    """
    Calculate the Brier Score for a multi-class classification problem.

    Parameters:
    y_true (array-like): Array of true class labels (shape: [n_samples]).
    y_prob (array-like): Array of predicted probabilities for each class (shape: [n_samples, num_classes]).
    num_classes (int): The number of classes.

    Returns:
    float: The Brier Score.
    """

    brier_score = 0.0

    y_true_one_hot = np.zeros((len(y_true), num_classes))
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1

    brier_score = np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1)) / num_classes

    return brier_score

class BrierScoreLoss(nn.Module):
    def __init__(self, num_classes):
        super(BrierScoreLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, y_pred, y_true):
        y_prob = torch.nn.functional.softmax(y_pred, dim=1)
        
        y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=self.num_classes).float()
        
        brier_score = torch.mean(torch.sum((y_prob - y_true_one_hot) ** 2, dim=1)) / self.num_classes
        
        return brier_score

def evaluate_one_epoch(val_loader, module, loss, num_classes):
    """
    Evaluate the model on the validation set for one epoch and calculate the Brier Score.

    Parameters:
    val_loader (DataLoader): DataLoader for the validation set.
    model (torch.nn.Module): The model to evaluate.
    loss (torch.nn.Module): Loss function.
    num_classes (int): Number of classes.

    Returns:
    tuple: (Average loss over the validation set, Brier Score)
    """

    # Track loss
    val_loss_epoch = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for val_imgs, val_labels in val_loader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            val_preds = module(val_imgs, 'inference').mean(dim=1)
            val_L = loss(val_preds, val_labels)

            val_loss_epoch += val_L.item()

            all_labels.extend(val_labels.cpu().numpy())
            all_preds.extend(torch.softmax(val_preds, dim=1).cpu().numpy())

    avg_val_loss = val_loss_epoch / len(val_loader)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    brier_score = brier_score_multi_class(all_labels, all_preds, num_classes)

    return avg_val_loss, brier_score


def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, 
                              momentum=0.9,
                              weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate,
                               weight_decay=weight_decay)
    return optimizer

def build_loss(loss):
    if loss == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss == "brier":
        return BrierScoreLoss(num_classes)
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
    


sweep_config = {
    'method': 'random',  # random, grid or bayes
    'name': 'sweep-random-Multi output module',
    'metric': {'goal': 'minimize', 'name': 'val_brier_score'},
    'parameters': 
    {
        'batch_size': {'values': [8, 16, 32]},
        'n_epochs': {'values': [100]},
        'learning_rate': {'values': [5e-4, 1e-4, 1e-5, 5e-5, 5e-6]},
        'optimizer': {'values': ['adam', 'sgd']}, #, 'sgd'
        'num_heads': {'values': [3, 5, 7, 10]},
        'weight_decay': {'values': [0, 5e-4]},
        'loss': {'values': ['CrossEntropyLoss']}, #, 'brier'
    }
}

def activation(string):
    if string == 'nn.ReLU()':
        return nn.ReLU()
    elif string == 'nn.LeakyReLU()':
        return nn.LeakyReLU()
    else:
        return nn.Tanh()

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="Multi output module")

def train(verbose=True):
    # Initialize a new wandb run
    with wandb.init(config=sweep_config, project="Multi output module"):
    
        train_loader = DataLoader(dataset=train_dataset, batch_size=wandb.config.batch_size * wandb.config.num_heads, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

        module = Multi_output_module(wandb.config.num_heads, base_model, device).to(device)
        
        loss = build_loss(wandb.config.loss)
        optimizer = build_optimizer(module, wandb.config.optimizer, wandb.config.learning_rate, wandb.config.weight_decay)

        for epoch in range(wandb.config.n_epochs):
            
            train_loss = train_one_epoch(train_loader, module, loss, optimizer, wandb.config.num_heads)
                
            avg_val_loss, val_brier_score = evaluate_one_epoch(val_loader, module, loss, num_classes)

            wandb.log({
                'epoch': epoch, 
                'train_loss': train_loss,
                'val_brier_score': val_brier_score,
                'val_loss': avg_val_loss
            })

            if verbose:
                print(f'Epoch {epoch+1}/{wandb.config.n_epochs}, loss {train_loss:.5f}, val_loss {avg_val_loss:.5f}')


wandb.agent(sweep_id, function=train, count=64)