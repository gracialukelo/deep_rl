from typing import Any, List, Tuple
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Activation, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical

class Agent:
    """
    Ein Agent, der das CartPole-Spiel mit Hilfe des Kreuzentropie-Verfahrens (Cross-Entropy Method, CEM) spielt.
    Der Agent verwendet ein neuronales Netzwerk, um die optimale Aktion für jeden Zustand zu erlernen.

    Attribute:
        env (gym.Env): Die CartPole-Umgebung von Gym.
        observations (int): Die Dimension der Zustandsbeobachtungen in der Umgebung.
        actions (int): Die Anzahl der möglichen Aktionen in der Umgebung.
        model_dir (str): Das Verzeichnis, in dem das Modell gespeichert wird.
        model_path (str): Der vollständige Pfad, unter dem das Modell gespeichert oder geladen wird.
        model (Sequential): Das neuronale Netzwerk-Modell, das verwendet wird, um die Aktionen zu bestimmen.
    """
    def __init__(self, env: gym.Env, model_dir: str) -> None:
        """
        Initialisiert den Agenten mit der angegebenen Umgebung und dem Modellverzeichnis.
        
        Args:
            env (gym.Env): Die Gym-Umgebung.
            model_dir (str): Das Verzeichnis, in dem das Modell gespeichert wird.
        """
        self.env = env
        self.observations: int = self.env.observation_space.shape[0]
        self.actions: int = self.env.action_space.n
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "cartpole_model.keras")
        self.model = self.get_model()

    def get_model(self) -> Sequential:
        """
        Erstellt oder lädt das neuronale Netzwerk-Modell.
        Das Modell wird geladen, wenn es bereits existiert, andernfalls wird ein neues Modell erstellt.

        Returns:
            Sequential: Das neuronale Netzwerk-Modell.
        """
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            model = load_model(self.model_path)
        else:
            model = Sequential()
            model.add(Dense(units=128, input_dim=self.observations, activation="relu"))
            model.add(Dense(units=128, activation="relu"))
            model.add(Dense(units=self.actions, activation="softmax"))
            model.compile(optimizer=Adam(learning_rate=0.005), loss="categorical_crossentropy")
        return model

    def save_model(self) -> None:
        """
        Speichert das trainierte Modell im angegebenen Verzeichnis.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def get_action(self, state: np.ndarray) -> int:
        """
        Wählt eine Aktion basierend auf den Ausgabe-Wahrscheinlichkeiten des Modells aus.

        Args:
            state (np.ndarray): Der aktuelle Zustand der Umgebung.

        Returns:
            int: Die gewählte Aktion.
        """
        state = state.reshape(1, -1)
        policy = self.model(state, training=False).numpy()[0]
        return np.random.choice(self.actions, p=policy)

    def get_samples(self, num_episodes: int) -> Tuple[List[float], List[List[Tuple[np.ndarray, int]]]]:
        """
        Generiert Samples (Episoden), indem der Agent mit der Umgebung interagiert.

        Args:
            num_episodes (int): Die Anzahl der Episoden, die generiert werden sollen.

        Returns:
            Tuple[List[float], List[List[Tuple[np.ndarray, int]]]]:
                Eine Liste von Belohnungen und eine Liste von Episoden, die jeweils die Zustände und Aktionen enthalten.
        """
        rewards = [0.0 for _ in range(num_episodes)]
        episodes: List[List[Tuple[np.ndarray, int]]] = [[] for _ in range(num_episodes)]

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, episodes

    def filter_episodes(
        self,
        rewards: List[float],
        episodes: List[List[Tuple[np.ndarray, int]]],
        percentile: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Filtert die besten Episoden basierend auf den Belohnungen und bereitet die Trainingsdaten vor.

        Args:
            rewards (List[float]): Eine Liste von Belohnungen.
            episodes (List[List[Tuple[np.ndarray, int]]]): Eine Liste von Episoden.
            percentile (float): Der Prozentsatz, der bestimmt, welche Episoden ausgewählt werden.

        Returns:
            Tuple[np.ndarray, np.ndarray, float]:
                x_train (np.ndarray): Die Zustandsbeobachtungen für das Training.
                y_train (np.ndarray): Die Aktionen für das Training.
                reward_bound (float): Die Belohnungsgrenze, die zur Auswahl der besten Episoden verwendet wurde.
        """
        reward_bound = float(np.percentile(rewards, percentile))
        x_train_: List[np.ndarray] = []
        y_train_: List[int] = []
        for reward, episode in zip(rewards, episodes):
            if reward >= reward_bound:
                observation = [step[0] for step in episode]
                action = [step[1] for step in episode]
                x_train_.extend(observation)
                y_train_.extend(action)
        x_train = np.asarray(x_train_)
        y_train = to_categorical(y_train_, num_classes=self.actions)
        return x_train, y_train, reward_bound

    def train(
        self, percentile: float, num_iterations: int, num_episodes: int
    ) -> Tuple[List[float], List[float]]:
        """
        Trainiert das Modell mit dem Kreuzentropie-Verfahren.

        Args:
            percentile (float): Der Prozentsatz, der bestimmt, welche Episoden für das Training verwendet werden.
            num_iterations (int): Die Anzahl der Iterationen, die trainiert werden sollen.
            num_episodes (int): Die Anzahl der Episoden pro Iteration.

        Returns:
            Tuple[List[float], List[float]]:
                Eine Liste von durchschnittlichen Belohnungen und eine Liste von Belohnungsgrenzen für jede Iteration.
        """
        reward_means: List[float] = []
        reward_bounds: List[float] = []
        for it in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            self.model.train_on_batch(x=x_train, y=y_train)
            reward_mean = float(np.mean(rewards))
            print(
                f"Iteration: {it:2d} "
                f"Reward Mean: {reward_mean:.4f} "
                f"Reward Bound: {reward_bound:.4f}"
            )
            reward_bounds.append(reward_bound)
            reward_means.append(reward_mean)
        return reward_means, reward_bounds

    def play(self, episodes: int, render: bool = True) -> None:
        """
        Spielt das Spiel mit dem trainierten Modell.

        Args:
            episodes (int): Die Anzahl der Episoden, die gespielt werden sollen.
            render (bool): Ob die Umgebung während des Spiels gerendert werden soll.
        """
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                if done:
                    break

            print(f"Episode: {episode} Total Reward: {total_reward}")
        self.env.close()


def main() -> None:
    """
    Hauptfunktion, um den Agenten zu trainieren und zu evaluieren.
    Wenn das Modell bereits existiert, wird es geladen. Andernfalls wird der Agent trainiert und das Modell gespeichert.
    Nach dem Training wird das Spiel gespielt und der Trainingsplot wird gespeichert.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model_dir = "cross_entropy/saved_model"
    plot_path = os.path.join(model_dir, "training_performance.png")
    agent = Agent(env, model_dir)
    
    if not os.path.exists(agent.model_path):
        # Train the agent using Cross-Entropy Method
        reward_means, reward_bounds = agent.train(
            percentile=70.0, num_iterations=21, num_episodes=100
        )
        agent.save_model()

        # Plot the training performance and save the plot
        plt.title("Training Performance")
        plt.plot(range(len(reward_means)), reward_means, color="red")
        plt.plot(range(len(reward_bounds)), reward_bounds, color="blue")
        plt.legend(["Reward Means", "Reward Bounds"])
        plt.savefig(plot_path)
        print(f"Training plot saved to {plot_path}")
    
    # Play the game using the trained model
    input("Training complete. Press Enter to play...")
    agent.play(episodes=10)


if __name__ == "__main__":
    main()
