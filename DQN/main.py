import numpy as np
import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Maze Environment
class Maze:
    def __init__(self, size=4):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.board[self.goal_pos[0], self.goal_pos[1]] = 2  # Mark the goal
        return self.get_state()

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        if action == 1:
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        if action == 2:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        if action == 3:
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)

        reward = 0
        done = False
        if self.agent_pos == self.goal_pos:
            reward = 1
            done = True
        return self.get_state(), reward, done

    def get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.agent_pos[0], self.agent_pos[1]] = 1
        return state.reshape((1, -1))

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Main loop with Pygame
def main():
    pygame.init()
    font = pygame.font.Font(None, 36)
    env = Maze(size=4)
    state_size = env.size * env.size
    action_size = 4  # Up, Down, Left, Right
    agent = DQNAgent(state_size, action_size)

    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    for e in range(1000):  # episodes
        state = env.reset()
        for time in range(2000):  # time steps
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            screen.fill((0, 0, 0))

            # Render the episode counter
            episode_text = font.render(f'Episode: {e + 1}', True, (255, 255, 255))
            screen.blit(episode_text, (10, 10))  # Adjust position as needed

            for i in range(env.size):
                for j in range(env.size):
                    rect = pygame.Rect(j * 100, i * 100, 100, 100)
                    if [i, j] == env.agent_pos:
                        pygame.draw.rect(screen, (0, 0, 255), rect)
                    elif [i, j] == env.goal_pos:
                        pygame.draw.rect(screen, (0, 255, 0), rect)
                    else:
                        pygame.draw.rect(screen, (255, 255, 255), rect, 1)
            pygame.display.flip()

            if done:
                break

            agent.replay(32)
            clock.tick(60)

if __name__ == "__main__":
    main()
