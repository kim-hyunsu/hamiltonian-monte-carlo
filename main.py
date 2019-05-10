import hmc
import autograd.numpy as np

# Generate random precision and mean parameters for a Gaussian
n_dim = 50
rng = np.random.RandomState(seed=1234)
rnd_eigvec, _ = np.linalg.qr(rng.normal(size=(n_dim, n_dim)))
rnd_eigval = np.exp(rng.normal(size=n_dim) * 2)
prec = (rnd_eigvec / rnd_eigval) @ rnd_eigvec.T
mean = rng.normal(size=n_dim)

# Eigenvalue decomposition
w, v = np.linalg.eig(prec)
prec = np.diagflat(prec)

# Deine potential energy (negative log density) for the Gaussian target
# distribution (gradient will be automatically calculated using autograd)
def pot_energy(pos):
    pos_minus_mean = pos - mean
    return 0.5 * pos_minus_mean @ prec @ pos_minus_mean

# Specify Hamiltonian system with isotropic Gaussian kinetic energy
# system = hmc.systems.EuclideanMetricSystem(pot_energy)

# H = 0.5 * mom @ inv(prec) @ mom + 0.5 * pos_minus_mean @ prec @ pos_minus_mean
metric = hmc.metrics.DenseEuclideanMetric(prec)
system = hmc.systems.EuclideanMetricSystem(pot_energy, metric)

# Hamiltonian is separable therefore use explicit leapfrog integrator
integrator = hmc.integrators.LeapfrogIntegrator(system, step_size=0.15)

# Use dynamic integration-time HMC implementation with multinomial 
# sampling from trajectories
# sampler = hmc.samplers.DynamicMultinomialHMC(system, integrator, rng)

# Standard HMC
sampler = hmc.samplers.StaticMetropolisHMC(system, integrator, rng, n_step=10)

# Sample an initial position from zero-mean isotropic Gaussian
init_pos = rng.normal(size=n_dim)

# Sample a Markov chain with 1000 transitions
# chains, chain_stats = sampler.sample_chain(1000, init_pos)

# for test
chains, chain_stats = sampler.sample_chain(100, init_pos)
print(chains['pos'])
print(chain_stats['hamiltonian'])
print(chain_stats['n_step'])
print(chain_stats['accept_prob'])
print(chain_stats['non_reversible_step'])
print(chain_stats['convergence_error'])
print(chain_stats['tree_depth'])
print(chain_stats['diverging'])

# Print RMSE in mean estimate
mean_rmse = np.mean((chains['pos'].mean(0) - mean)**2)**0.5
print(f'Mean estimate RMSE: {mean_rmse}')

# Print average acceptance probability
mean_accept_prob = chain_stats['accept_prob'].mean()
print(f'Mean accept prob: {mean_accept_prob:0.2f}')