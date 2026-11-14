import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms.functional import gaussian_blur


ROOT = Path(__file__).resolve().parent
BLUR_KERNEL = 9
BLUR_SIGMA = 1.8


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 24), nn.ReLU(),
        )
        self.classifier = nn.Linear(24, 10)


def load_checkpoints():
    baseline_path = ROOT / 'checkpoints/baseline.pt'
    patch = json.loads((ROOT / 'results/llm_patch.json').read_text(encoding='utf-8'))
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != patch['base_sha256']:
        raise ValueError('The patch requires the bundled baseline checkpoint.')
    baseline = torch.load(baseline_path, map_location='cpu', weights_only=True)
    adam = torch.load(ROOT / 'checkpoints/adam_head.pt', map_location='cpu', weights_only=True)
    model = FashionCNN().eval()
    model.load_state_dict(baseline)
    for name, value in baseline.items():
        if name.startswith('encoder.') and not torch.equal(value, adam[name]):
            raise ValueError('The Adam checkpoint has a different encoder.')
    return model, adam, patch['edits']


@torch.inference_mode()
def encode_test_split(model, data_dir):
    FashionMNIST.mirrors = [
        'https://fashion-mnist.s3.eu-central-1.amazonaws.com/',
        'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/',
        *FashionMNIST.mirrors,
    ]
    test = FashionMNIST(data_dir, train=False, download=True)
    clean_features, blurred_features = [], []
    for batch in DataLoader(test.data, batch_size=256):
        clean = batch.unsqueeze(1).float().div_(255)
        blurred = gaussian_blur(clean, [BLUR_KERNEL] * 2, [BLUR_SIGMA] * 2)
        clean_features.append(model.encoder(clean))
        blurred_features.append(model.encoder(blurred))
    return torch.cat(clean_features), torch.cat(blurred_features), test.targets


def edit_bias(bias, edits):
    edited = bias.clone()
    for class_id, delta in edits:
        edited[class_id] += delta
    return edited


@torch.inference_mode()
def evaluate(model, adam, patch, clean, blurred, labels):
    weight, bias = model.classifier.weight, model.classifier.bias

    def accuracy(features, head_weight, head_bias):
        predicted = F.linear(features, head_weight, head_bias).argmax(1)
        return (predicted == labels).double().mean().item() * 100

    def row(method, head_weight, head_bias, seed=''):
        return dict(method=method, seed=seed,
                    clean_accuracy_pct=accuracy(clean, head_weight, head_bias),
                    blurred_accuracy_pct=accuracy(blurred, head_weight, head_bias),
                    changed_head_parameters=int((head_weight != weight).sum() + (head_bias != bias).sum()))

    llm_edits = [(edit['class_id'], edit['delta']) for edit in patch]
    magnitudes = [abs(delta) for _, delta in llm_edits]
    rows = [
        row('Original CNN', weight, bias),
        row('Adam head fine-tuning', adam['classifier.weight'], adam['classifier.bias']),
        row('ChatGPT direct bias edit', weight, edit_bias(bias, llm_edits)),
    ]
    for seed in range(100, 130):
        rng = random.Random(seed)
        indices = rng.sample(range(len(bias)), len(magnitudes))
        edits = [(index, magnitude * rng.choice((-1, 1))) for index, magnitude in zip(indices, magnitudes)]
        rows.append(row('Matched random edit', weight, edit_bias(bias, edits), seed))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Replay the checkpoint comparison on Fashion-MNIST.')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model, adam, patch = load_checkpoints()
    rows = evaluate(model, adam, patch, *encode_test_split(model, args.data_dir))
    for row in rows[:3]:
        print(f"{row['method']:24} | clean {row['clean_accuracy_pct']:6.2f}% | blurred {row['blurred_accuracy_pct']:6.2f}%")
    for condition in ('clean', 'blurred'):
        scores = [row[f'{condition}_accuracy_pct'] for row in rows[3:]]
        print(f'Random edits, {condition:7} | {mean(scores):.2f}% (SD {stdev(scores):.2f} pp, n={len(scores)})')
    if args.output:
        with args.output.open('w', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
