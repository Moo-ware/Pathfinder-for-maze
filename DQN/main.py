import numpy as np
import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the DQN (Neural Network) in PyTorch
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


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
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        if action == 1 and self.agent_pos[0] < self.size - 1:  # Down
            self.agent_pos[0] += 1
        if action == 2 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1
        if action == 3 and self.agent_pos[1] < self.size - 1:  # Right
            self.agent_pos[1] += 1

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
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float()
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float()
            next_state = torch.from_numpy(next_state).float()
            reward = torch.tensor(reward)
            action = torch.tensor(action)
            done = torch.tensor(done)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            output = self.model(state)
            target_f = output.clone()
            target_f[0][action] = target
            loss = nn.MSELoss()(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def visualize_path(screen, path, env):
    visited = set()  # Keep track of visited positions
    font = pygame.font.Font(None, 36)
    for pos, action in path:
        visited.add(tuple(pos))  # Add position to visited set
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        screen.fill((0, 0, 0))
        episode_text = font.render("Best Path", True, (255, 255, 255))
        screen.blit(episode_text, (10, 10))  # Adjust position as needed

        for i in range(env.size):
            for j in range(env.size):
                rect = pygame.Rect(j * 100, i * 100, 100, 100)
                if (i, j) in visited:
                    pygame.draw.rect(screen, (255, 200, 0), rect)  # Draw visited path in a different color
                elif [i, j] == env.goal_pos:
                    pygame.draw.rect(screen, (0, 255, 0), rect)
                else:
                    pygame.draw.rect(screen, (255, 255, 255), rect, 1)

        pygame.display.flip()
        pygame.time.wait(500)  # Time delay for each step

    pygame.time.wait(20000)

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
    best_path = []
    best_length = float('inf')

    for e in range(50):  # episodes
        state = env.reset()
        current_path = []
        for time in range(2000):  # time steps
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            current_path.append((env.agent_pos.copy(), action))  # Store position and action
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

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
                if time < best_length:
                    best_length = time
                    best_path = current_path.copy()
                break

            agent.replay(32)
            clock.tick(60)
        
    return best_path

if __name__ == "__main__":
    best_path = main()
    print(best_path)
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    visualize_path(screen, best_path, Maze(size=4))
    
