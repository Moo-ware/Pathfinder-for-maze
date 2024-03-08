import gym
import pygame
import numpy as np
from gym import spaces

class ContinuousMazeEnv(gym.Env):
    def __init__(self):
        super(ContinuousMazeEnv, self).__init__()

        self.width, self.height = 600, 400
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=max(self.width, self.height), shape=(2,), dtype=np.float32)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Agent settings
        self.agent_radius = 10
        self.agent_pos = np.array([self.width / 4, self.height / 2])
        self.agent_speed = 5.0

        # Walls (represented as rectangles)
        self.walls = [
            pygame.Rect(100, 50, 150, 10),
            pygame.Rect(100, 250, 200, 10),
            pygame.Rect(100, 50, 10, 200),
            pygame.Rect(290, 60, 10, 190)
            # Add more walls as needed
        ]

    def step(self, action):
        new_pos = self.agent_pos + action * self.agent_speed
        if not self.check_collision(new_pos):
            self.agent_pos = new_pos

        # Boundary conditions
        self.agent_pos = np.clip(self.agent_pos, self.agent_radius, np.array([self.width, self.height]) - self.agent_radius)

        observation = self.agent_pos
        reward = 0
        done = False

        return observation, reward, done, {}

    def reset(self):
        self.agent_pos = np.array([self.width / 4, self.height / 2])
        return self.agent_pos

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, (255, 255, 255), wall)

        # Draw agent
        pygame.draw.circle(self.screen, (255, 0, 0), self.agent_pos.astype(int), self.agent_radius)
        pygame.display.flip()
        self.clock.tick(60)

    def check_collision(self, new_pos):
        agent_rect = pygame.Rect(new_pos[0] - self.agent_radius, new_pos[1] - self.agent_radius,
                                 self.agent_radius * 2, self.agent_radius * 2)
        return any(agent_rect.colliderect(wall) for wall in self.walls)

    def close(self):
        pygame.quit()

# Testing the environment
env = ContinuousMazeEnv()

for _ in range(5000):
    env.render()
    action = env.action_space.sample()  # Replace this with your algorithm's action
    env.step(action)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()
