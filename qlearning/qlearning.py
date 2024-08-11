import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Q-Learning Parameter
alpha = 0.1  # Lernrate
gamma = 0.99  # Diskontfaktor
epsilon = 1.0  # Exploitationsrate (wird mit der Zeit verringert)
epsilon_decay = 0.995  # Epsilon decay rate per episode
epsilon_min = 0.01  # Minimaler Wert für Epsilon
num_episodes = 500  # Anzahl der Episoden

# Umgebung initialisieren
env = gym.make("Taxi-v3")
state_size = env.observation_space.n
action_size = env.action_space.n

# Q-Tabelle initialisieren
Q = np.zeros((state_size, action_size))

# Funktion zur Auswahl der Aktion basierend auf Epsilon-Greedy-Politik
def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.randint(action_size)  # Zufällige Aktion
    return np.argmax(Q[state, :])  # Aktion mit höchstem Q-Wert

# Hauptschleife für das Training
rewards = []
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        # Q-Learning Update Regel
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)
    print(f"Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# Nach dem Training wird das Q-Table verwendet, um die beste Politik zu spielen
def play_game():
    state, _ = env.reset()
    total_reward = 0
    while True:
        action = np.argmax(Q[state, :])
        state, reward, done, _, _ = env.step(action)
        env.render()
        total_reward += reward
        if done:
            break
    print(f"Total Reward: {total_reward}")

# Spiel spielen
play_game()

# Plot der Belohnungen über die Episoden
plt.plot(range(num_episodes), rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Performance")
plt.show()

env.close()
