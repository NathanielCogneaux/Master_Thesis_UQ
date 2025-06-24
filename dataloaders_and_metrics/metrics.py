import numpy as np

from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, precision_recall_curve, auc
from scipy.stats import entropy

import torch
import torch.nn.functional as F
from torchmetrics.classification import CalibrationError


def compute_mce(predictions, confidences, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [(conf >= bin_lower) and (conf < bin_upper) for conf in confidences]
        if np.any(in_bin):
            accuracy_in_bin = np.mean([pred == label for pred, label, inb in zip(predictions, labels, in_bin) if inb])
            avg_confidence_in_bin = np.mean([conf for conf, inb in zip(confidences, in_bin) if inb])
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    return mce


def compute_calibration_metrics(loader, model, device, num_classes=100, n_bins=15, is_module=False):
    model.eval()
    ece_metric = CalibrationError(n_bins=n_bins, task="multiclass", num_classes=num_classes).to(device)

    correct = 0
    total = 0
    test_loss = 0.0

    all_labels = []
    all_confidences = []
    all_predictions = []
    all_softmax_outputs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_module:
                outputs = model(images, 'inference')
                softmax_outputs = F.softmax(outputs.mean(dim=1), dim=-1)
            else:
                outputs = model(images)
                softmax_outputs = F.softmax(outputs, dim=1)

            test_loss += F.cross_entropy(softmax_outputs, labels, reduction='sum').item()
            preds = softmax_outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            ece_metric.update(softmax_outputs, labels)
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(softmax_outputs.max(dim=1)[0].cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_softmax_outputs.append(softmax_outputs.cpu().numpy())

    accuracy = 100. * correct / total
    test_loss /= total
    ece = ece_metric.compute().item()

    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_predictions = np.array(all_predictions)
    all_softmax_outputs = np.concatenate(all_softmax_outputs, axis=0)

    nll = log_loss(all_labels, all_softmax_outputs)
    brier = mean_squared_error(
        F.one_hot(torch.tensor(all_labels), num_classes=num_classes).cpu().numpy(), all_softmax_outputs
    )
    mce = compute_mce(all_predictions, all_confidences, all_labels)

    return {
        "Accuracy": accuracy,
        "NLL": nll,
        "Brier": brier,
        "ECE": ece,
        "MCE": mce
    }


def compute_corruption_metrics(c_loader, model, device, num_classes=100, n_bins=15, is_module=False):
    model.to(device)
    model.eval()

    cA, cECE, cBrier, cMCE, cNLL = [], [], [], [], []

    for corruption, loader in c_loader.items():
        correct = 0
        total = 0
        test_loss = 0.0

        ece_metric = CalibrationError(n_bins=n_bins, task="multiclass", num_classes=num_classes).to(device)

        all_labels = []
        all_confidences = []
        all_predictions = []

        with torch.no_grad():
            all_softmax_outputs = []
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)

                if is_module:
                    outputs = model(images, 'inference')
                    softmax_outputs = F.softmax(outputs.mean(dim=1), dim=-1)
                else:
                    outputs = model(images)
                    softmax_outputs = F.softmax(outputs, dim=1)

                test_loss += F.cross_entropy(softmax_outputs, labels, reduction='sum').item()
                preds = softmax_outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

                ece_metric.update(softmax_outputs, labels)
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(softmax_outputs.max(dim=1)[0].cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

                all_softmax_outputs.append(softmax_outputs.cpu().numpy())

            accuracy = 100. * correct / total
            test_loss /= total
            ece = ece_metric.compute().item()

            softmax_outputs = np.concatenate(all_softmax_outputs)
            
            all_labels = torch.tensor(all_labels, dtype=torch.long)
            one_hot_labels = F.one_hot(all_labels, num_classes=num_classes).cpu().numpy()

            nll = log_loss(all_labels.cpu().numpy(), softmax_outputs)
            brier = mean_squared_error(one_hot_labels, softmax_outputs)
            mce = compute_mce(all_predictions, all_confidences, all_labels.cpu().numpy())

            cA.append(accuracy)
            cNLL.append(nll)
            cECE.append(ece)
            cBrier.append(brier)
            cMCE.append(mce)

    return {
        "cA": np.mean(cA),
        "cNLL": np.mean(cNLL),
        "cECE": np.mean(cECE),
        "cBrier": np.mean(cBrier),
        "cMCE": np.mean(cMCE),
    }


def compute_energy_score(logits):
    return -torch.logsumexp(logits, dim=1).cpu().numpy()

def compute_metrics_for_scores(labels, scores):
    auroc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    aur = np.trapz(recall, precision)
    return auroc, aupr, aur

def compute_ood_detection_metrics(train_loader, ood_loaders, model, num_classes, device, is_module=False):
    model.eval()

    ind_metrics = {'entropy': [], 'msp': [], 'energy': []}
    ood_metrics = {dset: {'entropy': [], 'msp': [], 'energy': []} for dset in ood_loaders.keys()}

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)

            if is_module:
                outputs = model(images, 'inference')
                outputs_avg = outputs.mean(dim=1)
            else:
                outputs = model(images)

            outputs_softmax = F.softmax(outputs_avg if is_module else outputs, dim=1).cpu().numpy()

            entropy_val = entropy(outputs_softmax, axis=1)
            msp = np.max(outputs_softmax, axis=1)
            energy = compute_energy_score(outputs_avg if is_module else outputs)

            ind_metrics['entropy'].extend(entropy_val)
            ind_metrics['msp'].extend(-msp)
            ind_metrics['energy'].extend(energy)

        for dset, ood_loader in ood_loaders.items():
            for images, _ in ood_loader:
                images = images.to(device)

                if is_module:
                    outputs = model(images, 'inference')
                    outputs_avg = outputs.mean(dim=1)
                else:
                    outputs = model(images)

                outputs_softmax = F.softmax(outputs_avg if is_module else outputs, dim=1).cpu().numpy()

                entropy_val = entropy(outputs_softmax, axis=1)
                msp = np.max(outputs_softmax, axis=1)
                energy = compute_energy_score(outputs_avg if is_module else outputs)

                ood_metrics[dset]['entropy'].extend(entropy_val)
                ood_metrics[dset]['msp'].extend(-msp)
                ood_metrics[dset]['energy'].extend(energy)

    results = {}
    labels_id = np.zeros(len(ind_metrics['entropy']))

    for dset, ood_metric_values in ood_metrics.items():
        labels_ood = np.ones(len(ood_metric_values['entropy']))
        results[dset] = {}

        for metric in ['entropy', 'msp', 'energy']:
            combined_labels = np.concatenate([labels_id, labels_ood])
            combined_scores = np.concatenate([ind_metrics[metric], ood_metric_values[metric]])

            if len(np.unique(combined_labels)) == 2:
                auroc, aupr, aur = compute_metrics_for_scores(combined_labels, combined_scores)
                results[dset][metric] = {
                    'AUROC': auroc,
                    'AUPR': aupr,
                    'AUR': aur
                }

    return results

'''
def compute_all_metrics(test_loader, corrupted_loaders, ood_loaders, model, device, num_classes=100, is_module=False):
    all_metrics = {}

    all_metrics['Test Calibration'] = compute_calibration_metrics(test_loader, model, device, num_classes, is_module=is_module)

    if corrupted_loaders is not None:
        all_metrics['Corrupted Calibration'] = compute_corruption_metrics(corrupted_loaders, model, device, num_classes, is_module=is_module)

    if ood_loaders is not None:
        all_metrics['OOD Detection'] = compute_ood_detection_metrics(test_loader, ood_loaders, model, num_classes, device, is_module=is_module)

    return all_metrics
'''

def compute_all_metrics(test_loader, corrupted_loaders, ood_loaders, model, device, num_classes=100, is_module=False):
    all_metrics = {}

    all_metrics['Test Calibration'] = compute_calibration_metrics(test_loader, model, device, num_classes, is_module=is_module)

    if corrupted_loaders is not None:
        corrupted_metrics = compute_corruption_metrics(corrupted_loaders, model, device, num_classes, is_module=is_module)
        for key, value in corrupted_metrics.items():
            all_metrics['Test Calibration'][key] = value

    if ood_loaders is not None:
        all_metrics['OOD Detection'] = compute_ood_detection_metrics(test_loader, ood_loaders, model, num_classes, device, is_module=is_module)

    return all_metrics
