# CNN Weight Repair in PyTorch

How much accuracy can a CNN recover on blurred images when a language model edits just six classifier biases? This experiment starts from a CNN trained on clean Fashion-MNIST and compares direct bias edits with Adam fine-tuning and matched random edits.

[Open the training notebook in Colab](https://colab.research.google.com/github/Ahmad3752/cnn-weight-repair-pytorch/blob/main/CNN_Weight_Repair_Fashion_MNIST.ipynb) or [browse it on GitHub](CNN_Weight_Repair_Fashion_MNIST.ipynb).

## Held-out test results

| Method | Clean accuracy | Blurred accuracy | Head parameters changed |
|---|---:|---:|---:|
| Original CNN | 86.11% | 65.12% | 0 |
| Adam fine-tuning | 76.60% | 78.31% | 180 |
| ChatGPT direct bias edit | 83.68% | 71.12% | 6 |
| Matched random edits, 30-run mean | 85.11% | 62.55% | 6 |

The six-bias edit gained **6.00 percentage points** on blurred images and lost **2.43 points** on clean images. Full-head Adam achieved the highest blurred accuracy, with a **9.51-point** drop on clean images. The random control's blurred accuracy had a sample standard deviation of 3.09 points.

## Replay the comparison

Use Python 3.10 or newer. The script runs on CPU, downloads Fashion-MNIST on first use, and evaluates the exact checkpoints shipped here. It verifies the baseline checkpoint hash before applying the patch. No API key is required.

```bash
python -m pip install -r requirements.txt
python replay.py --output replay_results.csv
```

The recorded scores came from the Colab training and evaluation run. PyTorch versions can differ in floating-point rounding. The CSV files in `results/` record the original training, validation, and test results.

## Experiment design

1. Split the Fashion-MNIST training portion into 54,000 training, 2,000 adaptation, and 4,000 validation images using seed 42.
2. Train a two-convolution CNN on clean training images for four epochs with Adam. Select the best checkpoint by clean validation accuracy.
3. Apply a fixed Gaussian blur with a 9 × 9 kernel and sigma of 1.8. Inspect misclassifications on blurred adaptation images.
4. Fit the Adam comparison on blurred adaptation features with the encoder frozen. Select the best Adam epoch using blurred validation accuracy. Ask ChatGPT to propose one six-bias patch from the adaptation errors. Generate 30 random controls with the same six absolute edit magnitudes, random target biases, and random signs.
5. Freeze the methods before evaluating them on the same 10,000 official test images in clean and blurred form.

The LLM patch is a single manual proposal. Its six values are recorded in `results/llm_patch.json`. The experiment covers one trained CNN and one fixed blur setting. Comparing against an Adam run that can edit only the same six biases would help isolate the effect of the update method from the number and location of parameters it can edit.
