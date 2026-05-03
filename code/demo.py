from gmm import GMM
from data import DataManager
from utils import *
from baselines import *
from pcpg import PCPG

import time
from sklearn.linear_model import LogisticRegression
import torch

def evaluate(name, method, w, b=0.0):
    t0 = time.perf_counter()
    expectation = method(w, b)
    t = time.perf_counter() - t0
    print(f"{name:<10} expectation={expectation:.4f} | t={t:.2f}s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_samples = 2000
num_features = 10  

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
w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)
b = float(clf.intercept_[0])

mc   = MCBaseline(px, n_samples=1000)
pcpg = PCPG(px, 20, 20)
gh = GaussHermiteBaseline(px, n_points=20)

evaluate("MC",               mc, w, b)
evaluate("PCPG",             pcpg, w, b)
evaluate("GH-20",  gh, w, b)


