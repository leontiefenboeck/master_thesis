import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class DataManager:
    def __init__(self, name, samples=1000, features=20, test_size=0.2, random_state=42):
        if name == 'sk_class':
            x, y = make_classification(
                n_samples=samples,
                n_features=features,
                weights=[0.5, 0.5],
                random_state=random_state,
            )
        elif name == 'moons':
            x, y = make_moons(n_samples=samples, noise=0.1, random_state=random_state)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        x = x.astype(np.float32)
        y = y.astype(np.int64)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    

    def get_dataloaders(self, batch_size=256):
        train_dataset = TensorDataset(torch.from_numpy(self.x_train), torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.x_test), torch.from_numpy(self.y_test))

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        return train_loader, test_loader
    
    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def get_test_missing(self, missing_rate=0.1):
        x_test_missing = self.x_test.copy()
        
        n_features = self.x_test.shape[1]
        n_missing = int(missing_rate * n_features)
        missing_feature_indices = np.random.choice(n_features, n_missing, replace=False)
        
        for idx in missing_feature_indices:
            x_test_missing[:, idx] = np.nan
        
        return x_test_missing, self.y_test.copy()


 