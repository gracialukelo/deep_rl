import gymnasium as gym
import numpy as np
from typing import Any

class Agent:
    def __init__(self, env: gym.Env, gamma: float = 0.99, theta: float = 1e-6) -> None:
        """
        Initializes the Agent class with the environment, discount factor (gamma),
        and convergence threshold (theta).

        :param env: The environment in which the agent operates.
        :param gamma: Discount factor for future rewards.
        :param theta: Threshold for convergence in value iteration.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_table = np.zeros(env.observation_space.n)  # Initialize value table with zeros
        self.policy = np.zeros(env.observation_space.n)  # Initialize policy table with zeros

    def value_iteration(self) -> np.ndarray:
        """
        Performs the Value Iteration algorithm to find the optimal value table.
        
        :return: The final value table after convergence.
        """
        iteration_count = 0  # Counter for the number of iterations

        while True:
            delta = 0  # Delta to check for convergence

            # Loop through all states
            for state in range(self.env.observation_space.n):
                v_old = self.value_table[state]  # Store the old value of the state

                # Calculate the new value of the state
                action_values = []

                for action in range(self.env.action_space.n):
                    action_value = sum([prob * (reward + self.gamma * self.value_table[next_state])
                                        for prob, next_state, reward, _ in self.env.P[state][action]])
                    action_values.append(action_value)
                best_action_value = max(action_values)
                self.value_table[state] = best_action_value
                delta = max(delta, abs(v_old - best_action_value))

            iteration_count += 1  # Increment the iteration counter
            print(f"Iteration {iteration_count}: Value Table:\n{self.value_table.reshape((4, 4))}\n")

            # Check for convergence after updating all states
            if delta < self.theta:
                print(f"Optimal value table found after {iteration_count} iterations.\n")
                break

        return self.value_table  

    def extract_policy(self) -> Any: # kann auch np.ndarray sein
        """
        Extracts the optimal policy based on the value table.
        
        :return: The optimal policy.
        """
        for state in range(self.env.observation_space.n):
            action_values = []
            for action in range(self.env.action_space.n):
                action_value = sum([prob * (reward + self.gamma * self.value_table[next_state])
                                    for prob, next_state, reward, _ in self.env.P[state][action]])
                action_values.append(action_value)
            self.policy[state] = np.argmax(action_values)

        return self.policy
    
    def play_game(self) -> float:
        """
        Plays a game in the environment based on the extracted policy.

        :return: The total reward obtained during the game.
        """
        state, _ = self.env.reset()  # Reset the environment to the initial state
        total_reward = 0

        while True:
            action = int(self.policy[state])  # Get the action from the policy
            state, reward, done, _, _ = self.env.step(action)  # Take the action and observe the result
            total_reward += reward  # Accumulate the reward
            self.env.render()  # Render the environment

            if done:  # If the game is over, break the loop
                break

        return total_reward


if __name__ == "__main__":
    # Initialize the environment
    env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human")

    # Create and train the agent
    agent = Agent(env)
    value_table = agent.value_iteration()  # Perform value iteration to find the optimal value table
    policy = agent.extract_policy()  # Extract the optimal policy based on the value table

    print("Optimal Value Table:")
    print(value_table.reshape((4, 4)))

    print("\nOptimal Policy:")
    print(policy.reshape((4, 4)))

    # Display the directions for the policy
    actions = ["Left", "Down", "Right", "Up"]
    policy_actions = np.array([actions[int(action)] for action in policy])
    print("\nOptimal Policy (Actions):")
    print(policy_actions.reshape((4, 4)))

    # Play the game
    print("\nStarting game...")
    total_reward = agent.play_game()
    print(f"Total Reward: {total_reward}")
