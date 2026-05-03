from utils import *
import random
from cirkit.pipeline import PipelineContext, compile
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.distributions as D
import math
from cirkit.symbolic.circuit import Circuit, Scope
from cirkit.symbolic.layers import GaussianLayer, SumLayer, HadamardLayer
from cirkit.templates import utils

K = 2

radius = 2  # Distance of the centers from the origin
mus = torch.tensor([
    [math.cos(2*math.pi*n / K) for n in range(K)],
    [math.sin(2*math.pi*n / K) for n in range(K)]
]).T * radius
sigma = .2  # Standard deviation

mix = D.Categorical(torch.ones(K,))
comp = D.Independent(D.Normal(mus, sigma), 1)
gmm = D.MixtureSameFamily(mix, comp)

def sample_points(n_points):
    return gmm.sample((n_points,))

def true_density(xy):
    return gmm.log_prob(xy).exp()

def build_symbolic_circuit() -> Circuit:
    # This parametrizes the mixture weights such that they add up to one.
    weight_factory = utils.parameterization_to_factory(utils.Parameterization(
        activation='softmax',   # Parameterize the sum weights by using a softmax activation
        initialization='uniform' # Initialize the sum weights by sampling from a standard normal distribution
    ))

    # We introduce one more mixture than in the original model
    # Again, SGD/Adam is not the best way to fit a (shallow) Gaussian mixture model
    units = K+1 
    
    g0 = GaussianLayer(Scope((0,)), units)
    g1 = GaussianLayer(Scope((1,)), units)
    prod = HadamardLayer(num_input_units=units, arity=2)
    sl = SumLayer(units, 1, 1, weight_factory=weight_factory)

    return Circuit(
        layers=[g0, g1, prod, sl],  # Layers that appear in the circuit (i.e. nodes in the graph)
        in_layers={  # Connections between layers (i.e. edges in the graph as an adjacency list)
            g0: [],
            g1: [],
            prod: [g0, g1],
            sl: [prod],
        },
        outputs=[sl]  # Nodes that are returned by the circuit
    )


plot_2D(true_density, title='Original density')
symbolic_circuit = build_symbolic_circuit()

# Print which structural properties the circuit satisfies
print(f'Structural properties:')
print(f'  - Smoothness: {symbolic_circuit.is_smooth}')
print(f'  - Decomposability: {symbolic_circuit.is_decomposable}')
print(f'  - Structured-decomposability: {symbolic_circuit.is_structured_decomposable}')

dataset_size = 10000
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device('cuda')

data_train = TensorDataset(sample_points(dataset_size))
data_test = TensorDataset(sample_points(dataset_size//10))

train_dataloader = DataLoader(data_train, shuffle=True, batch_size=256)
test_dataloader = DataLoader(data_test, shuffle=False, batch_size=256)


ctx = PipelineContext(
    backend='torch',  # Choose PyTorch as compilation backend
    semiring='lse-sum',
    fold=True,     # Fold the circuit to better exploit GPU parallelism
    optimize=True  # Optimize the layers of the circuit
)

with ctx:  # Compile the circuits computing log |c(X)| and log |Z|
    circuit = compile(symbolic_circuit)

def model_density(xy):
    return circuit(xy).exp()

plot_2D(model_density, title='Model density (before training)', xmin=-3, xmax=3)

optimizer = optim.Adam(circuit.parameters(), lr=0.01)
num_epochs = 30
step_idx = 0
running_loss = 0.0
running_samples = 0

circuit = circuit.to(device)

for epoch_idx in range(num_epochs):
    for i, (batch,) in enumerate(train_dataloader):
        batch = batch.to(device)

        log_likelihoods = circuit(batch)                 
        loss = -torch.mean(log_likelihoods)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.detach() * len(batch)
        running_samples += len(batch)
        step_idx += 1
        if step_idx % 50 == 0:
            average_nll = running_loss / running_samples
            print(f"Step {step_idx}: Average NLL: {average_nll:.3f}")
            running_loss = 0.0
            running_samples = 0

plot_2D(model_density, title='Model density (after training)', xmin=-3, xmax=3)