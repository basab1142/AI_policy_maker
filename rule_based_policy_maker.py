
# View the notebook for results and plots of the dynamics of the system under this hardcoded tax policy

from environment import  CommonsSim
import numpy as np
sim = CommonsSim(n_agents=10, seed=42)
field_health_history = []
avg_harvest_history = []
avg_reward_history = []

tax_rate_history = []

tax_rate = np.random.rand()  # Random tax rate between 0 and 1
tax_rate_history.append(tax_rate)
for round in range(1000):
    obs = sim.step(tax_rate)
    field_health_history.append(obs['field_health'])
    avg_harvest_history.append(obs['avg_harvest'])
    avg_reward_history.append(obs['avg_reward'])
    # hardcoded tax policy, if field health is below 0.65 increase tax rate, otherwise decrease it [number is experimentally chosen to create some dynamics in the system]
    if obs['field_health'] < 0.65:
        tax_rate = min(tax_rate + 0.01, 1.0)  # Increase tax rate but cap at 1.0
    else:
        tax_rate = max(tax_rate - 0.01, 0.0)  # Decrease tax rate but floor at 0.0
    tax_rate_history.append(tax_rate)