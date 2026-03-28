# A shared apple field. 10 agents. Each round, every agent decides how aggressively to harvest — a number
# between 0 (gentle) and 1 (maximum harvest). The field has a health score that goes up when agents harvest
# gently and goes down when they overharvest. If the field health drops below 30%, all rewards are halved until
# it recovers.
# Trying to make something like that
# sim = CommonsSim(n_agents=10, seed=42)
# obs = sim.reset() # {'field_health': 1.0, 'avg_harvest': 0.0, 'avg_reward': 0.0}
# for round in range(150):
# tax_rate = policymaker.act(obs) # number between 0 and 1
# obs, field_health, info = sim.step(tax_rate)

import numpy as np
class CommonsSim:
    def __init__(self, n_agents = 10, seed = 42):
        self.num_agents = n_agents
        self.field_health = 1.0  # Start with a healthy field
        np.random.seed(seed)
        # assigning greedy levels to agents
        self.agent_greed = np.random.rand(n_agents)  # Random greed levels between 0 and 1
        self.avg_harvest = 0.0
        self.avg_reward = 0.0
    def reset(self): # reset the environment to the initial state
        self.field_health = 1.0
        self.round = 0
        self.avg_harvest = 0.0
        self.avg_reward = 0.0
        return {'field_health': self.field_health, 'avg_harvest': self.avg_harvest, 'avg_reward': self.avg_reward}
    def step(self, tax_rate):
        # Agents decide how much to harvest based on their greed and the tax rate
        harvests = self.agent_greed * (1 - tax_rate)  # Harvest is reduced by the tax rate
        avg_harvest = np.mean(harvests)
        # Update field health based on overharvesting
        if avg_harvest > 0.5:  # If average harvest is too high, field health decreases
            self.field_health -= 0.1  
        else:  # If average harvest is low, field health recovers
            self.field_health += 0.05 
        self.field_health = np.clip(self.field_health, 0, 1)  # Keep health between 0 and 1
        # Calculate rewards for agents
        rewards = harvests * self.field_health  # Rewards are based on harvest and field health
        avg_reward = np.mean(rewards)
        # If field health drops below 30%, halve the rewards until it recovers
        if self.field_health < 0.3:
            avg_reward *= 0.5
        self.avg_harvest = avg_harvest
        self.avg_reward = avg_reward
        self.round += 1
        return {'field_health': self.field_health, 'avg_harvest': self.avg_harvest, 'avg_reward': self.avg_reward}, rewards
