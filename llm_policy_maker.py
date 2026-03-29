from environment import CommonsSim

from openai import OpenAI
import os
import time
import re
class LLMPolicyMaker:
    def __init__(self, model="meta/llama3-8b-instruct"):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key= os.getenv("NVIDIA_API")
        )
        self.model = model
        self.history = []

        # token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # building prompt with context, history, instructions, and query
    def build_prompt(self):
        history_lines = []

        for h in self.history[-5:]:  # last 5 steps only
            history_lines.append(
                f"Tax={h['tax']:.2f} -> Field={h['field']:.2f}, Reward={h['reward']:.2f}, Harvest={h['harvest']:.2f}"
            )

        history_block = "\n".join(history_lines) if history_lines else "None"

        prompt = f"""
        ### CONTEXT

        You are an economic policymaker controlling a shared apple field.

        There are multiple agents harvesting apples each round.

        * Each agent chooses how aggressively to harvest (low or high).
        * High harvesting increases short-term rewards but damages the field.
        * Low harvesting preserves the field but reduces rewards.

        The field has a health value between 0 and 1:

        * If field health drops too low, rewards decrease significantly.
        * If field health stays high, long-term rewards are better.

        You control a tax rate (between 0 and 1):

        * Higher tax discourages aggressive harvesting.
        * Lower tax encourages higher harvesting.

        Your goal is to:

        * Maintain field health at a sustainable level
        * While also keeping agent rewards reasonably high

        Avoid:

        * Field collapse (very low health)
        * Killing incentives (very low rewards)

        ---

        ### HISTORY

        Each entry shows:
        Tax Rate -> Field Health, Average Reward, Average Harvest

        {history_block}

        ---

        ### INSTRUCTIONS

        Based on the history above:

        * Identify how tax rate affects field health and rewards
        * Adjust the tax rate to balance sustainability and reward

        ---

        ### OUTPUT FORMAT (STRICT)

        Return ONLY a number between 0 and 1.

        Do not explain.
        Do not write anything else.

        ---

        ### QUERY

        What should the next tax rate be?

        """
        return prompt

    # number extraction helper[in case LLM returns extra text]
    def extract_number(self, text):
        match = re.search(r"\d*\.?\d+", text)
        return float(match.group()) if match else 0.1

    # action is the tax rate to apply, between 0 and 1
    time.sleep(0.5)  # to avoid hitting rate limits
    def get_action(self):
        prompt = self.build_prompt()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=20
        )

        output = response.choices[0].message.content.strip()

        # token usage for cost estimation
        usage = response.usage
        
        self.total_input_tokens += usage.prompt_tokens
        self.total_output_tokens += usage.completion_tokens
        

        tax = self.extract_number(output)
        tax = max(0.0, min(1.0, tax)) # clamp between 0 and 1

        return tax

  
    def update(self, tax, obs):
        self.history.append({
            "tax": tax,
            "field": obs["field_health"],
            "reward": obs["avg_reward"],
            "harvest": obs["avg_harvest"]
        })

    


# Simulate the environment with the LLM policy maker

def simulate_with_LLM(policy, sim, rounds=150):

    obs = sim.reset()
    field_history = []
    reward_history = []
    tax_history = []
    for step in range(rounds):
        # get tax from LLM
        tax = policy.get_action()
        # environment step
        obs = sim.step(tax)
        # update LLM memory
        policy.update(tax, obs)

        # logging
        field_history.append(obs["field_health"])
        reward_history.append(obs["avg_reward"])
        tax_history.append(tax)

    return field_history, reward_history, tax_history, policy.total_input_tokens, policy.total_output_tokens

# all the simulations have been done in the notebook, checkout the notebook section for more