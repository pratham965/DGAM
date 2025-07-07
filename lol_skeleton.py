#!/usr/bin/env python
# coding: utf-8
"""
LOL ST-GCN training script with hardcoded configuration
-----------------------------------------------------
Train a Spatio-Temporal Graph Convolutional Network (ST-GCN) on a
folder-structured LOL action-recognition dataset:

LOL/
 ├── drink/
 │   └── clips/*.mp4
 ├── jump/
 │   └── clips/*.mp4
 └── …

Requirements:
    pip install torch-geometric torch-geometric-temporal mediapipe==0.10.9

Usage:
    python lol_stgcn_training.py
"""

# ---------------------------------------------------------------------- #
# 1. Imports
# ---------------------------------------------------------------------- #
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Data
from torch_geometric_temporal.nn.attention import STConv
from tqdm import tqdm

# ---------------------------------------------------------------------- #
# 2. Configuration Variables (Hardcoded)
# ---------------------------------------------------------------------- #
# Dataset Configuration
DATASET_ROOT = "/path/to/LOL"  # Change this to your LOL dataset path
SKELETON_CACHE = "lol_skeletons.json"

# Training Configuration
SEQUENCE_LENGTH = 32
BATCH_SIZE = 8
NUM_WORKERS = 2
EPOCHS = 100
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2

# Model Configuration
NUM_JOINTS = 33
IN_CHANNELS = 3
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 64
KERNEL_SIZE = 3
K = 3

# Other Configuration
PRECOMPUTE_SKELETONS = False  # Set to True to precompute skeletons first
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------- #
# 3. Skeleton extraction using MediaPipe
# ---------------------------------------------------------------------- #
class SkeletonExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        keypoints_seq = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            if results.pose_landmarks:
                keypoints = []
                for lm in results.pose_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                keypoints_seq.append(keypoints)
            else:
                keypoints_seq.append([0.0] * 99)  # 33×3

        cap.release()
        return np.array(keypoints_seq, dtype=np.float32)

# ---------------------------------------------------------------------- #
# 4. Pre-processing utilities
# ---------------------------------------------------------------------- #
class STGCNDataPreprocessor:
    def __init__(self, sequence_length: int = 32, num_joints: int = 33):
        self.sequence_length = sequence_length
        self.num_joints = num_joints

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        coords = keypoints.reshape(-1, self.num_joints, 3)
        if coords.shape[0] > 0:
            hip_center = (coords[:, 23, :] + coords[:, 24, :]) / 2
            coords -= hip_center[:, None, :]
        return coords

    def create_temporal_windows(self, skel: np.ndarray) -> np.ndarray:
        T = len(skel)
        if T < self.sequence_length:
            pad = np.repeat(skel[-1:], self.sequence_length - T, axis=0) \
                  if T else np.zeros((self.sequence_length, self.num_joints, 3))
            skel = np.concatenate([skel, pad], axis=0)
        elif T > self.sequence_length:
            idx = np.linspace(0, T - 1, self.sequence_length, dtype=int)
            skel = skel[idx]
        return skel.astype(np.float32)

# ---------------------------------------------------------------------- #
# 5. Graph construction
# ---------------------------------------------------------------------- #
class GraphConstructor:
    def __init__(self):
        self.edges = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),      # arms
            (11, 23), (12, 24), (23, 24),                          # torso
            (23, 25), (25, 27), (27, 29), (27, 31),                # left leg
            (24, 26), (26, 28), (28, 30), (28, 32)                 # right leg
        ]

    def adjacency(self, N: int = 33) -> np.ndarray:
        A = np.eye(N, dtype=np.float32)
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1.0
        return A

    def skeleton_to_graph(self, skel: np.ndarray):
        T, N, C = skel.shape
        edge_index = torch.tensor(np.array(np.where(self.adjacency(N))), dtype=torch.long)
        x = torch.tensor(skel, dtype=torch.float32)  # (T, N, C)
        return x, edge_index

# ---------------------------------------------------------------------- #
# 6. Dataset
# ---------------------------------------------------------------------- #
class LOLDataset(Dataset):
    """
    Folder-structured dataset:

        root/
          └── <class_name>/clips/*.mp4
    """
    def __init__(self,
                 root_dir: str,
                 sequence_length: int = 32,
                 precomputed_skeletons: str | None = None):
        self.root = Path(root_dir)
        self.seq_len = sequence_length
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[str, int]] = self._index_clips()

        self.skel_ext = SkeletonExtractor()
        self.prep = STGCNDataPreprocessor(sequence_length)
        self.graph = GraphConstructor()

        self.skel_cache = {}
        if precomputed_skeletons and Path(precomputed_skeletons).exists():
            self.skel_cache = json.load(open(precomputed_skeletons))

        print(f"Indexed {len(self.samples)} clips from {len(self.classes)} classes")

    def _index_clips(self) -> List[Tuple[str, int]]:
        items = []
        for cls in self.classes:
            for p in (self.root / cls / "clips").glob("*.*"):
                if p.is_file():
                    items.append((str(p), self.class2idx[cls]))
        return items

    # ------------------------------------------------------------------ #
    # torch Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def _load_skeleton(self, video_path: str) -> np.ndarray:
        key = Path(video_path).name
        if key in self.skel_cache:
            return np.array(self.skel_cache[key], dtype=np.float32)
        return self.skel_ext.extract_keypoints(video_path)

    def __getitem__(self, idx):
        vpath, label = self.samples[idx]
        try:
            skel = self._load_skeleton(vpath)
            skel = self.prep.normalize_keypoints(skel)
            skel = self.prep.create_temporal_windows(skel)
        except Exception as e:
            print("Skeleton error:", e, vpath)
            skel = np.zeros((self.seq_len, 33, 3), dtype=np.float32)

        x, edge_index = self.graph.skeleton_to_graph(skel)
        return x, edge_index, torch.tensor(label, dtype=torch.long)

# ---------------------------------------------------------------------- #
# 7. Dataloader helpers
# ---------------------------------------------------------------------- #
def collate_fn(batch):
    xs, edge_indices, labels = zip(*batch)
    x = torch.stack(xs, 0)             # (B, T, N, C)
    edge_index = edge_indices[0]       # identical for every sample
    labels = torch.stack(labels, 0)
    return x, edge_index, labels


def make_lol_loaders(root_dir: str,
                     seq_len: int = 32,
                     batch_size: int = 8,
                     num_workers: int = 2,
                     val_ratio: float = 0.2,
                     precomp_json: str | None = None):
    full_ds = LOLDataset(root_dir, seq_len, precomp_json)
    n_val = int(len(full_ds) * val_ratio)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_ld = DataLoader(train_ds, batch_size, True,
                          num_workers=num_workers, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, batch_size, False,
                        num_workers=num_workers, collate_fn=collate_fn)
    return train_ld, val_ld, len(full_ds.classes), full_ds.classes

# ---------------------------------------------------------------------- #
# 8. Model
# ---------------------------------------------------------------------- #
class STGCNClassifier(nn.Module):
    def __init__(self,
                 num_nodes: int = 33,
                 in_channels: int = 3,
                 hidden_channels: int = 64,
                 out_channels: int = 64,
                 num_classes: int = 10,   # will be overwritten
                 kernel_size: int = 3,
                 K: int = 3):
        super().__init__()
        self.stconv1 = STConv(num_nodes, in_channels,
                              hidden_channels, out_channels,
                              kernel_size, K)
        self.stconv2 = STConv(num_nodes, out_channels,
                              hidden_channels, out_channels,
                              kernel_size, K)

        self.classifier = nn.Sequential(
            nn.Linear(out_channels * num_nodes, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # x: (B, T, N, C)
        x = F.relu(self.stconv1(x, edge_index, edge_weight))
        x = F.relu(self.stconv2(x, edge_index, edge_weight))
        x = x.mean(dim=1)                     # temporal average → (B, N, C)
        x = x.view(x.size(0), -1)             # flatten
        return self.classifier(x)

# ---------------------------------------------------------------------- #
# 9. Training & evaluation
# ---------------------------------------------------------------------- #
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, edge_index, y in loader:
            x, y = x.to(device), y.to(device)
            edge_index = edge_index.to(device)
            logits = model(x, edge_index)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100 * correct / total


def train(model, train_ld, val_ld, epochs: int, lr: float, device):
    device = torch.device(device)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for x, edge_index, y in tqdm(train_ld, desc=f"Epoch {epoch}/{epochs}"):
            x, y = x.to(device), y.to(device)
            edge_index = edge_index.to(device)

            opt.zero_grad()
            logits = model(x, edge_index)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        sched.step()
        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_ld, device)

        print(f"[{epoch:03d}/{epochs}] "
              f"loss={running_loss/len(train_ld):.4f} "
              f"train_acc={train_acc:.2f}% "
              f"val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_lol_stgcn.pth")

    print(f"Best validation accuracy: {best_acc:.2f}%")
    return best_acc

# ---------------------------------------------------------------------- #
# 10. Optional skeleton pre-computation
# ---------------------------------------------------------------------- #
def precompute_skeletons(dataset_root: str, output_file: str, seq_len: int = 32):
    ds = LOLDataset(dataset_root, seq_len)
    skel_cache = {}
    for vpath, _ in tqdm(ds.samples, desc="Extracting skeletons"):
        key = Path(vpath).name
        skel_cache[key] = ds.skel_ext.extract_keypoints(vpath).tolist()

    with open(output_file, "w") as f:
        json.dump(skel_cache, f)
    print(f"Saved {len(skel_cache)} skeleton sequences to {output_file}")

# ---------------------------------------------------------------------- #
# 11. Main execution
# ---------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("LOL ST-GCN Action Recognition Training")
    print("=" * 60)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Check if skeleton precomputation is needed
    if PRECOMPUTE_SKELETONS:
        print("\nPrecomputing skeletons...")
        precompute_skeletons(DATASET_ROOT, SKELETON_CACHE, SEQUENCE_LENGTH)
        print("Skeleton precomputation completed!")
        return

    # Create data loaders
    print("\nCreating data loaders...")
    train_ld, val_ld, num_classes, class_names = make_lol_loaders(
        root_dir=DATASET_ROOT,
        seq_len=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=VAL_RATIO,
        precomp_json=SKELETON_CACHE if Path(SKELETON_CACHE).exists() else None
    )

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Training samples: {len(train_ld.dataset)}")
    print(f"Validation samples: {len(val_ld.dataset)}")

    # Create model
    print(f"\nCreating ST-GCN model...")
    model = STGCNClassifier(
        num_nodes=NUM_JOINTS,
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_classes=num_classes,
        kernel_size=KERNEL_SIZE,
        K=K
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Test data loader
    print(f"\nTesting data loader...")
    for batch_idx, (x, edge_index, labels) in enumerate(train_ld):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input shape: {x.shape}")
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sample labels: {[class_names[l.item()] for l in labels[:3]]}")
        if batch_idx >= 2:  # Show first 3 batches
            break

    # Start training
    print(f"\nStarting training...")
    best_acc = train(model, train_ld, val_ld, EPOCHS, LEARNING_RATE, DEVICE)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved as: best_lol_stgcn.pth")
    if Path(SKELETON_CACHE).exists():
        print(f"Skeleton cache: {SKELETON_CACHE}")


if __name__ == "__main__":
    main()
