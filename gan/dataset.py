import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PianoRollDataset(Dataset):
    def __init__(self, data_dir="processed_rolls"):
        self.data = []
        self.labels = []
        self.composer2id = {}
        composer_id_counter = 0

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith(".npy"):
                continue
            path = os.path.join(data_dir, fname)
            composer = fname.split("_")[0]

            # 自动创建 composer → id 映射
            if composer not in self.composer2id:
                self.composer2id[composer] = composer_id_counter
                composer_id_counter += 1

            label = self.composer2id[composer]
            self.data.append(path)
            self.labels.append(label)

        print(f"[INFO] Found {len(self.composer2id)} composers:", self.composer2id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        roll = np.load(self.data[idx])  # shape: [T, 88]
        roll = torch.tensor(roll, dtype=torch.float32)  # [T, 88]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return roll, label





