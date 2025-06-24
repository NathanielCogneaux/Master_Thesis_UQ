import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from dataloaders_and_metrics.dataloaders import load_dataset
from dataloaders_and_metrics.metrics import compute_all_metrics

import models
from multi_output_module.multi_output_module import Multi_output_module

#torch.cuda.set_device(0)

data_path = './data'
result_dir = './Final_results/Saved_results_early_stopping'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
num_classes_dict = {'MNIST': 10, 'FashionMNIST': 10, 'CIFAR10': 10, 'CIFAR100': 100}
weight_decay = {'MNIST': 0, 'FashionMNIST': 0, 'CIFAR10': 0.0005, 'CIFAR100': 0.0005}

num_exp = 5
num_heads_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
batch_size_train = 32
batch_size_test = 32
num_epochs = 200
patience = 10

def load_model(data_set_name, device):
    model_paths = {
        'CIFAR100': './models/saved_models/cifar/cifar100/wide-resnet-28x10.pth',
        'CIFAR10': './models/saved_models/cifar/cifar10/wide-resnet-28x10.pth',
        'MNIST': './models/saved_models/mnist/mnist_lenet5.pth',
        'FashionMNIST': './models/saved_models/fashion_mnist/fashion_mnist_lenet5.pth'
    }

    try:
        if data_set_name == 'CIFAR100':
            model = models.wide_resnet.Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100).to(device)
        elif data_set_name == 'CIFAR10':
            model = models.wide_resnet.Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=10).to(device)
        elif data_set_name == 'MNIST' or data_set_name == 'FashionMNIST':
            model = models.lenet.LeNet5().to(device)
        else:
            raise ValueError(f"Unsupported dataset name: {data_set_name}")

        model_path = model_paths[data_set_name]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        return model

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")


def write_calibration_metrics_to_txt(file_path, dataset_name, model_names, calibration_metrics):
    content_lines = []
    content_lines.append(f"Calibration Metrics for {dataset_name}\n\n")

    header = "Model\t" + "\t".join(calibration_metrics[0].keys()) + "\n"
    content_lines.append(header)

    for model_name, metrics in zip(model_names, calibration_metrics):
        row = model_name + "\t"
        row += "\t".join(
            f"{metrics[metric]:.4f}" if isinstance(metrics[metric], float)
            else f"{metrics[metric]['mean']:.4f} ± {metrics[metric]['std']:.4f}" 
            if isinstance(metrics[metric]['mean'], (int, float)) and isinstance(metrics[metric]['std'], (int, float))
            else f"{metrics[metric]['mean']} ± {metrics[metric]['std']}" 
            if isinstance(metrics[metric], dict) else f"{metrics[metric]}"
            for metric in metrics.keys()
        ) + "\n"
        content_lines.append(row)

    content_lines.append("\n")

    with open(file_path, 'a') as f:
        f.write("".join(content_lines))

def write_ood_metrics_to_txt(file_path, dataset_name, model_metric_names, ood_datasets, ood_metrics):
    content_lines = []
    content_lines.append(f"\nOOD Metrics for {dataset_name}\n")
    
    header = "Model/Metric\t" + "\t".join([f"{ood}_AUROC (mean ± std)\t{ood}_AUPR (mean ± std)\t{ood}_AUR (mean ± std)" for ood in ood_datasets]) + "\n"
    content_lines.append(header)
    
    for model_idx, model_name in enumerate(model_metric_names):
        row = model_name + "\t"
        for ood_name in ood_datasets:
            metrics = ood_metrics[model_idx][ood_name]

            if isinstance(metrics['AUROC'], dict):
                auroc_mean = metrics['AUROC']['mean']
                auroc_std = metrics['AUROC']['std']
                aupr_mean = metrics['AUPR']['mean']
                aupr_std = metrics['AUPR']['std']
                aur_mean = metrics['AUR']['mean']
                aur_std = metrics['AUR']['std']

                row += f"{auroc_mean:.4f} ± {auroc_std:.4f}\t{aupr_mean:.4f} ± {aupr_std:.4f}\t{aur_mean:.4f} ± {aur_std:.4f}\t"
            else:
                row += f"{metrics['AUROC']:.4f}\t{metrics['AUPR']:.4f}\t{metrics['AUR']:.4f}\t"

        content_lines.append(row.strip() + "\n")

    with open(file_path, 'a') as f:
        f.write("".join(content_lines))

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def train_module(module, train_loader, val_loader, optimizer, criterion, device, num_epochs, num_heads, patience=5):
    best_loss = float('inf')
    epochs_no_improve = 0 
    best_model_state = None

    start_time = time.time()
    
    for epoch in range(num_epochs):
        module.train()
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            labels = labels.view(-1, num_heads)

            optimizer.zero_grad()
            predictions = module(images, 'training')

            total_loss = 0
            for i in range(num_heads):
                total_loss += criterion(predictions[:, i, :], labels[:, i])
            total_loss /= num_heads

            total_loss.backward()
            optimizer.step()

            #if batch_idx % 100 == 0:
            #    print(f"Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)} "
            #          f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss.item():.6f}")

        module.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                predictions = module(images, 'inference')
                
                module_output_avg = predictions.mean(dim=1)
                val_loss += criterion(module_output_avg, labels).item()
                
                _, predicted = torch.max(module_output_avg, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / total

        print(f'Epoch {epoch}, Val Loss: {val_loss:.6f}, Accuracy: {correct}/{len(val_loader.dataset)} ({accuracy:.2f}%)')

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state = module.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if best_model_state is not None:
        module.load_state_dict(best_model_state)
    
    total_train_time = time.time() - start_time

    return module, epoch, total_train_time


def main():
    for dataset in train_datasets:
        print(f'Preparing {dataset}')
        results_file = os.path.join(result_dir, f'{dataset}_metrics_results.txt')

        base_model = load_model(dataset, device)
        base_model.eval()

        color_mode = 'rgb' if dataset in ['CIFAR10', 'CIFAR100'] else 'grey'
        num_classes = num_classes_dict[dataset]

        test_loader = load_dataset(dataset, 'test', mode=color_mode, batch_size_train=batch_size_train, batch_size_test=batch_size_test, drop_last=False, data_path=data_path)

        if dataset in ['CIFAR10', 'CIFAR100']:
            corrupted_loaders = load_dataset(dataset, 'corrupted', mode=color_mode, batch_size_train=batch_size_train, batch_size_test=batch_size_test, drop_last=False, data_path=data_path)
        else:
            corrupted_loaders = None  # No corrupted data for MNIST and FashionMNIST

        ood_loaders = load_dataset(dataset, 'ood', mode=color_mode, batch_size_train=batch_size_train, batch_size_test=batch_size_test, drop_last=False, data_path=data_path)

        all_calibration_metrics = []
        all_ood_metrics = []

        print('Computing base_model metrics')
        base_metrics = compute_all_metrics(
            test_loader=test_loader,
            corrupted_loaders=corrupted_loaders,
            ood_loaders=ood_loaders,
            model=base_model,
            device=device,
            num_classes=num_classes
        )

        all_calibration_metrics.append(
            ("Base Model", base_metrics['Test Calibration'])
        )

        model_metric_names = ["Base Model Entropy", "Base Model MSP", "Base Model Energy"]
        ood_metrics = []

        for metric_type in ['entropy', 'msp', 'energy']:
            model_metrics = {}
            for ood_name in ood_loaders.keys():
                if ood_name in base_metrics['OOD Detection'] and metric_type in base_metrics['OOD Detection'][ood_name]:
                    model_metrics[ood_name] = base_metrics['OOD Detection'][ood_name][metric_type]
                else:
                    model_metrics[ood_name] = {'AUROC': 'N/A', 'AUPR': 'N/A', 'AUR': 'N/A'}
            ood_metrics.append(model_metrics)

        all_ood_metrics.append((model_metric_names, ood_metrics))

        for num_heads in num_heads_list:
            train_loader, val_loader = load_dataset(dataset, 'train', mode=color_mode, batch_size_train=batch_size_train * num_heads, batch_size_test=batch_size_test, drop_last=True, data_path=data_path)

            print(f'Initializing module with {num_heads} heads')

            epoch_counts = []
            train_times = []

            multi_output_cal_metrics = []
            multi_output_ood_metrics = {ood_name: {'entropy': [], 'msp': [], 'energy': []} for ood_name in ood_loaders.keys()}

            for exp in range(num_exp):
                print(f'Experiment number {exp} for {num_heads} heads')
                module = Multi_output_module(num_heads, base_model, device).to(device)

                optimizer = optim.Adam(module.parameters(), lr=0.0005, weight_decay=weight_decay[dataset])
                criterion = nn.CrossEntropyLoss()

                module, total_epochs, total_train_time = train_module(module, train_loader, val_loader, optimizer, criterion, device, num_epochs, num_heads, patience)

                epoch_counts.append(total_epochs)
                train_times.append(total_train_time)

                print(f'Computing Experiment number {exp} for {num_heads} heads metrics')
                all_metrics = compute_all_metrics(
                    test_loader=test_loader,
                    corrupted_loaders=corrupted_loaders,
                    ood_loaders=ood_loaders,
                    model=module,
                    device=device,
                    num_classes=num_classes,
                    is_module=True
                )

                multi_output_cal_metrics.append(all_metrics['Test Calibration'])

                for ood_name, ood_metric in all_metrics['OOD Detection'].items():
                    for metric_type in ['entropy', 'msp', 'energy']:
                        if metric_type in ood_metric:
                            multi_output_ood_metrics[ood_name][metric_type].append(ood_metric[metric_type])


            mean_epochs = np.mean(epoch_counts)
            std_epochs = np.std(epoch_counts)
            mean_time = format_time(np.mean(train_times))
            std_time = format_time(np.std(train_times))


            mean_std_cal_metrics = {
                k: {'mean': np.mean([m[k] for m in multi_output_cal_metrics]),
                    'std': np.std([m[k] for m in multi_output_cal_metrics])}
                for k in multi_output_cal_metrics[0].keys()
            }
            mean_std_cal_metrics['Epochs'] = {'mean': mean_epochs, 'std': std_epochs}
            mean_std_cal_metrics['Training Time (s)'] = {'mean': mean_time, 'std': std_time}   

            all_calibration_metrics.append((f"Multi-Output {num_heads} heads", mean_std_cal_metrics))

            avg_ood_metrics = []
            for metric_type in ['entropy', 'msp', 'energy']:
                ood_metric_dict = {}
                for ood_name in ood_loaders.keys():
                    ood_metric_values = np.array(multi_output_ood_metrics[ood_name][metric_type])
                    ood_metric_dict[ood_name] = {
                        'AUROC': {'mean': np.mean([x['AUROC'] for x in ood_metric_values]),
                                  'std': np.std([x['AUROC'] for x in ood_metric_values])},
                        'AUPR': {'mean': np.mean([x['AUPR'] for x in ood_metric_values]),
                                 'std': np.std([x['AUPR'] for x in ood_metric_values])},
                        'AUR': {'mean': np.mean([x['AUR'] for x in ood_metric_values]),
                                'std': np.std([x['AUR'] for x in ood_metric_values])}
                    }
                avg_ood_metrics.append(ood_metric_dict)

            all_ood_metrics.append(([f"Multi-Output {num_heads} heads {metric_type.capitalize()}" for metric_type in ['entropy', 'msp', 'energy']], avg_ood_metrics))

        write_calibration_metrics_to_txt(
            file_path=results_file,
            dataset_name=dataset,
            model_names=[x[0] for x in all_calibration_metrics],
            calibration_metrics=[x[1] for x in all_calibration_metrics]
        )

        for model_metric_names, ood_metrics in all_ood_metrics:
            write_ood_metrics_to_txt(
                file_path=results_file,
                dataset_name=dataset,
                model_metric_names=model_metric_names,
                ood_datasets=ood_loaders.keys(),
                ood_metrics=ood_metrics
            )



main()