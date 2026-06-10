import random
import numpy as np
import torch

from data import DataManager
from gmm import GMM

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K = 4
n_features = 10

dm = DataManager('sk_class', samples=10_000, features=n_features,
                 test_size=0.2, random_state=seed)
train_dataloader, _ = dm.get_dataloaders(batch_size=64)
x_train_tensor, _ = train_dataloader.dataset.tensors

px = GMM(device, K=K, means_init=x_train_tensor[:K].to(device), n_features=n_features).to(device)
px.fit(train_dataloader)

x_partial = dm.get_test_missing(missing_rate=0.5, idx=5, device=device)
d_mis = torch.isnan(x_partial).sum().item()

# --- test 1: phi(0) == 1 ---
t_zero = torch.zeros(d_mis, device=device)
phi0 = px.characteristic_function(t_zero, x_partial)
print(f"phi(0) = {phi0.item():.8f}  (expected 1.0)")
assert abs(phi0.real.item() - 1.0) < 1e-5 and abs(phi0.imag.item()) < 1e-5, "phi(0) != 1"

# --- test 2: |phi(t)| <= 1 for random t ---
n_test = 10_000
t_rand = torch.randn(n_test, d_mis, device=device)
phi = px.characteristic_function(t_rand, x_partial)
mods = phi.abs()

print(f"max |phi(t)| over {n_test} random t: {mods.max().item():.8f}  (expected <= 1.0)")
print(f"mean |phi(t)|:                        {mods.mean().item():.8f}")
violations = (mods > 1.0 + 1e-5).sum().item()
print(f"violations (|phi| > 1 + 1e-5):        {violations}")
assert violations == 0, f"CF modulus exceeded 1 in {violations} cases!"

print("\nAll checks passed.")
