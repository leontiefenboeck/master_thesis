from gmm import GMM
from data import DataManager
from baselines import MCBaseline, MeanImputation
from pcpg import PCPG, PCPGHermite
from utils import *

import torch
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# for data
num_samples  = 10000
num_features = 15
obs_idx = 5

# for models
K         = 4
n_mc      = 10000
n_pg      = 100
n_gauss   = 100
n_hermite = 30

dm = DataManager('sk_class', samples=num_samples, features=num_features,
                 test_size=0.2, random_state=42)

x_train, x_test, y_train, y_test = dm.get_data()
train_dataloader, _ = dm.get_dataloaders(batch_size=64)

# --- fit GMM ---
x_train_tensor, _ = train_dataloader.dataset.tensors
means_init = x_train_tensor[:K].to(device)
px = GMM(device, K=K, means_init=means_init, n_features=num_features).to(device)
px.fit(train_dataloader)

# --- fit logistic regression ---
clf = LogisticRegression()
clf.fit(x_train, y_train)
w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)

# --- one partially observed test point ---
x_partial_np = dm.get_test_missing(missing_rate=0.5, idx=obs_idx)
x_partial    = torch.tensor(x_partial_np, dtype=torch.float32, device=device)

obs_mask = ~torch.isnan(x_partial)
print(f"Missing  features : {(~obs_mask).sum().item()} / {num_features}\n")

# --- computing the expectation ---
mean_imp = MeanImputation(x_train_tensor.mean(dim=0).to(device))
mc       = MCBaseline(px, n_samples=n_mc)
pcpg     = PCPG(px, n_pg=n_pg, n_gauss=n_gauss)
pcpg_gh  = PCPGHermite(px, n_pg=n_pg, n_hermite=n_hermite)

evaluate("Mean",    mean_imp, w, x_partial, 1)
evaluate("MC",      mc,       w, x_partial)
evaluate("PCPG",    pcpg,     w, x_partial)
evaluate("PCPG-GH", pcpg_gh,  w, x_partial)

# TODO
# - how can I compare in high dim settings