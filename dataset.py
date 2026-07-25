import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=64, seed=42):
        np.random.seed(seed)
        self.num_samples = num_samples
        self.img_size = img_size

        self.data = []
        self.labels = []

        self.class_specs = {
            0: ('circle', [1.0, 0.0, 0.0], 'grid'),
            1: ('square', [0.0, 1.0, 0.0], 'striped'),
            2: ('triangle', [0.0, 0.0, 1.0], 'dotted'),
            3: ('circle', [1.0, 1.0, 0.0], 'striped'),
            4: ('square', [1.0, 0.0, 1.0], 'dotted'),
            5: ('triangle', [0.0, 1.0, 1.0], 'grid'),
            6: ('square', [1.0, 0.0, 0.0], 'solid'),
            7: ('triangle', [0.0, 1.0, 0.0], 'solid'),
            8: ('circle', [0.0, 0.0, 1.0], 'solid'),
            9: ('square', [1.0, 1.0, 0.0], 'grid'),
        }

        for _ in range(num_samples):
            label = np.random.randint(0, 10)
            img = self._generate_image(label)
            self.data.append(img)
            self.labels.append(label)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def _generate_image(self, label):
        shape, color, texture = self.class_specs[label]
        img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)

        img += np.random.normal(0, 0.02, img.shape)

        mask = np.zeros((self.img_size, self.img_size), dtype=bool)
        cy, cx = self.img_size // 2, self.img_size // 2
        r = self.img_size // 3

        y, x = np.ogrid[:self.img_size, :self.img_size]

        if shape == 'circle':
            mask = (x - cx)**2 + (y - cy)**2 <= r**2
        elif shape == 'square':
            mask = (np.abs(x - cx) <= r) & (np.abs(y - cy) <= r)
        elif shape == 'triangle':
            mask = (y - cy <= r) & (y - cy >= -r) & (np.abs(x - cx) <= (y - cy + r) * 0.7)

        for c in range(3):
            img[c, mask] = color[c]

        if texture == 'striped':
            stripes = (y % 4 < 2)
            for c in range(3):
                img[c, mask & stripes] *= 0.5
        elif texture == 'dotted':
            dots = (x % 4 == 0) & (y % 4 == 0)
            for c in range(3):
                img[c, mask & dots] *= 0.3
        elif texture == 'grid':
            grid = (x % 6 == 0) | (y % 6 == 0)
            for c in range(3):
                img[c, mask & grid] *= 0.4

        img = np.clip(img, 0.0, 1.0)
        return img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloaders(batch_size=128, train_samples=5000, val_samples=1000, seed=42):
    train_dataset = SyntheticDataset(num_samples=train_samples, seed=seed)
    val_dataset = SyntheticDataset(num_samples=val_samples, seed=seed+1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
