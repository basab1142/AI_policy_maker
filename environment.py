# A shared apple field. 10 agents. Each round, every agent decides how aggressively to harvest — a number
# between 0 (gentle) and 1 (maximum harvest). The field has a health score that goes up when agents harvest
# gently and goes down when they overharvest. If the field health drops below 30%, all rewards are halved until
# it recovers.
# Trying to make something like that
# sim = CommonsSim(n_agents=10, seed=42)
# obs = sim.reset() # {'field_health': 1.0, 'avg_harvest': 0.0, 'avg_reward': 0.0}
# for round in range(150):
# tax_rate = policymaker.act(obs) # number between 0 and 1
# obs = sim.step(tax_rate)

import numpy as np
from collections import deque

class CommonsSim:
    def __init__(self, n_agents:int = 10, seed: int = 42):
        # if agent is less than 1, raise an error
        if n_agents < 1:
            raise ValueError("Number of agents must be at least 1")
        self.num_agents = n_agents
        self.field_health = 1.0  # Start with a healthy field
        np.random.seed(seed)
        # assigning greedy levels to agents
        self.agent_greed = np.random.rand(n_agents)  # Random greed levels between 0 and 1
        self.avg_harvest = 0.0
        self.avg_reward = 0.0
        # average tax distribution over last 10 rounds
        self.tax_distribution_history = deque(maxlen=10)
        self.tax_distribution_history.append(0.0)  # Start with 0 tax distribution
        # every agent has 2 action options: 0 (harvest gently) and 1 (harvest aggressively)
        #self.action_space = [np.exp(0)/ (np.exp(0) + np.exp(1)), np.exp(1)/ (np.exp(0) + np.exp(1))]# normalized 0 and 1 using softmax
        self.action_space = [0.1, 0.9]
        # temperature parameter for softmax action selection, higher means more exploration
        self.temperature = 0.5
    def reset(self): # reset the environment to the initial state
        self.field_health = 1.0
        self.round = 0
        self.avg_harvest = 0.0
        self.avg_reward = 0.0
        return {'field_health': self.field_health, 'avg_harvest': self.avg_harvest, 'avg_reward': self.avg_reward}
    def step(self, tax_rate: float):
        # Agents choice how much to harvest based on the expected reward that they will get given the action taken
        agents_choice = []
        for agent, greed in enumerate(self.agent_greed):
            # Reward production*(1 - tax_rate) + redistribution 
            production_gently = self.production(agent, 0) * (1 - tax_rate) + np.mean(self.tax_distribution_history) 
            production_aggressive = self.production(agent, 1) * (1 - tax_rate) + np.mean(self.tax_distribution_history)
            # choose action from softmax distribution based on expected reward with greed level as a weight with temperature parameter to control exploration
            action_prob = np.array([np.exp(production_gently / self.temperature), np.exp((production_aggressive+greed) / self.temperature)])  # Add greed to aggressive production
            action_prob = action_prob / np.sum(action_prob)  # Normalize to get probabilities
            action = np.random.choice([0, 1], p=action_prob)
            agents_choice.append(action)
        # agents take action and harvest the field
        harvest_of_agents = [self.production(agent, action) for agent, action in enumerate(agents_choice)]
        total_harvest = sum(harvest_of_agents)
        # tax collected from agents
        tax_collected = total_harvest * tax_rate
        # update tax distribution history
        episode_tax_distribution = tax_collected / self.num_agents  # Distribute tax equally among agents
        self.tax_distribution_history.append(episode_tax_distribution)  # Distribute tax equally among agents
        # update field health based on total mean harvest
        self.avg_harvest = total_harvest / self.num_agents 
        # update field health based on total mean harvest
        if self.avg_harvest > 0.5:  # If average harvest is too high, field health decreases
            self.field_health -= 0.5*(  self.avg_harvest - 0.5)  # Decrease health more if harvest is much higher than 0.5
        else:  # If average harvest is low, field health recovers
            self.field_health += 0.5*(  0.5 - self.avg_harvest)  # Increase health more if harvest is much lower than 0.5
        self.field_health = np.clip(self.field_health, 0, 1)  # Keep health between 0 and 1

        # calculate rewards for agents based on their harvest and tax distribution
        rewards = [harvest * (1 - tax_rate) + episode_tax_distribution for harvest in harvest_of_agents]  
        # if field health drops below 0.3, rewards are halved
        if self.field_health < 0.3:
            rewards = [reward * 0.5 for reward in rewards]
        self.avg_reward = np.mean(rewards)      

        return {'field_health': self.field_health, 'avg_harvest': self.avg_harvest, 'avg_reward': self.avg_reward}

    def production(self, agent:int, action:int):
        # production is based on field health and action they choose(to harvest gently or aggressively)
        production = self.field_health*self.action_space[action]
        return production