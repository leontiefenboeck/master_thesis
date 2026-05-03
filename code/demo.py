from gmm import GMM
from data import DataManager
from utils import *
from baselines import MCBaseline, MeanImputationBaseline
from pcpg import PCPG

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch

def evaluate(name, method, x_test, missing_mask, y_test, w, b=0.0):
    t0 = time.perf_counter()
    probs = method.pred(x_test, missing_mask, w, b)
    t = time.perf_counter() - t0

    acc      = ((probs >= 0.5) == y_test).float().mean().item()
    brier    = ((probs - y_test) ** 2).mean().item()
    print(f"{name:<30} acc={acc:.4f}  brier={brier:.4f}  t={t:.2f}s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_samples = 1000 
num_features = 5  
missing_rate = 0.4

dm = DataManager(
    'sk_class',
    samples=num_samples,
    features=num_features,
    test_size=0.2,
    random_state=42,
)

x_train, x_test, y_train, y_test = dm.get_data()
train_dataloader, test_dataloader = dm.get_dataloaders(batch_size=64)

x_train_tensor, _ = train_dataloader.dataset.tensors
means_init = x_train_tensor[:8].to(device) 

px = GMM(device, K=8, means_init=means_init, n_features=num_features).to(device)
px.fit(train_dataloader)

clf = LogisticRegression()
clf.fit(x_train, y_train)

x_test_missing, y_test_missing = dm.get_test_missing(missing_rate=missing_rate)

w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)
b = float(clf.intercept_[0])

missing_mask = torch.tensor(np.isnan(x_test_missing), dtype=torch.bool, device=device)
x_t = torch.tensor(np.nan_to_num(x_test_missing, nan=0.0), device=device)
y_t = torch.tensor(np.array(y_test_missing), dtype=torch.long, device=device)

mc   = MCBaseline(px, n_samples=1000)
mean = MeanImputationBaseline(x_train.mean(axis=0))
pcpg = PCPG(px, n_pg=20, n_gh=32)


evaluate("MC sampling",      mc,   x_t, missing_mask, y_t, w, b)
evaluate("Mean imputation",  mean, x_t, missing_mask, y_t, w, b)
evaluate("PCPG",             pcpg, x_t, missing_mask, y_t, w, b)



