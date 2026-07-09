"""Download CIFAR-10 from HuggingFace and save in torchvision-compatible format.

Avoids the cs.toronto.edu server which throttles to ~10KB/s.
Saves to data/cifar10/cifar-10-batches-py/ — the exact path torchvision expects.

Usage:
    python download_cifar10.py [--root data/cifar10]
"""

import argparse
import os
import pickle
import sys

import numpy as np

LABEL_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]


def split_batches(data, labels, n_batches=5):
    n = len(data)
    size = n // n_batches
    batches = []
    for i in range(n_batches):
        start, end = i * size, (i + 1) * size if i < n_batches - 1 else n
        batches.append((data[start:end], labels[start:end]))
    return batches


def hf_img_to_chw_flat(img):
    """PIL image or HWC numpy array → flat (3072,) array in CHW order."""
    arr = np.array(img, dtype=np.uint8)           # (32, 32, 3) HWC
    arr = arr.transpose(2, 0, 1)                   # (3, 32, 32) CHW
    return arr.flatten()                           # (3072,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/cifar10",
                        help="root directory (will contain cifar-10-batches-py/)")
    args = parser.parse_args()

    out_dir = os.path.join(args.root, "cifar-10-batches-py")
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) >= 7:
        print(f"Already exists: {out_dir}  (delete it to re-download)")
        sys.exit(0)

    print("Loading CIFAR-10 from HuggingFace (uoft-cs/cifar10)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.  Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset("uoft-cs/cifar10")

    print("Converting train split...")
    train = ds["train"]
    train_imgs   = np.array([hf_img_to_chw_flat(ex["img"]) for ex in train],
                             dtype=np.uint8)          # (50000, 3072)
    train_labels = [ex["label"] for ex in train]

    print("Converting test split...")
    test = ds["test"]
    test_imgs   = np.array([hf_img_to_chw_flat(ex["img"]) for ex in test],
                            dtype=np.uint8)           # (10000, 3072)
    test_labels = [ex["label"] for ex in test]

    os.makedirs(out_dir, exist_ok=True)

    # Write data_batch_1 … data_batch_5
    batches = split_batches(train_imgs, train_labels, n_batches=5)
    for i, (data, labels) in enumerate(batches, start=1):
        path = os.path.join(out_dir, f"data_batch_{i}")
        with open(path, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f, protocol=2)
        print(f"  wrote {path}  ({len(labels)} samples)")

    # Write test_batch
    path = os.path.join(out_dir, "test_batch")
    with open(path, "wb") as f:
        pickle.dump({"data": test_imgs, "labels": test_labels}, f, protocol=2)
    print(f"  wrote {path}  ({len(test_labels)} samples)")

    # Write batches.meta
    path = os.path.join(out_dir, "batches.meta")
    with open(path, "wb") as f:
        pickle.dump({
            "label_names":       [n.encode() for n in LABEL_NAMES],
            "num_cases_per_batch": 10000,
            "num_vis":             3072,
        }, f, protocol=2)
    print(f"  wrote {path}")

    print(f"\nDone. CIFAR-10 saved to: {os.path.abspath(out_dir)}")
    print("torchvision.datasets.CIFAR10 will use this directory automatically.")


if __name__ == "__main__":
    main()
