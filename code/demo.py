from gmm import GMM
from data import DataManager
from utils import *
from baselines import *
from pcpg import PCPG

import time
from sklearn.linear_model import LogisticRegression
import torch
import random
import numpy as np

def evaluate(name, method, w):
    t0 = time.perf_counter()
    expectation = method(w)
    t = time.perf_counter() - t0
    print(f"{name:<10} expectation={expectation:.4f} | t={t:.2f}s")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_samples = 1000
num_features = 2  
K = 4

dm = DataManager(
    'moons',
    samples=num_samples,
    features=num_features,
    test_size=0.2,
    random_state=42,
)

x_train, x_test, y_train, y_test = dm.get_data()
train_dataloader, test_dataloader = dm.get_dataloaders(batch_size=64)

x_train_tensor, _ = train_dataloader.dataset.tensors
means_init = x_train_tensor[:K].to(device) 

px = GMM(device, K=K, means_init=means_init, n_features=num_features).to(device)
px.fit(train_dataloader)
# plot_gmm_results(px, x_train, title="GMM Results")

clf = LogisticRegression()
clf.fit(x_train, y_train)
# plot_classifier_results(clf, x_test, y_test, title="Logistic Regression Results")

w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)

pcpg = PCPG(px, 20, 20)
gh   = GaussHermiteBaseline(px, n_points=20)
mc   = MCBaseline(px, n_samples=1000)

evaluate("MC",     mc, w)
evaluate("PCPG",   pcpg, w)
evaluate("GH-20",  gh, w)


