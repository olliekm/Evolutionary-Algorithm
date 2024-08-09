

import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn

# ... (Previous code for the worm game)
pygame.init()

WIDTH, HEIGHT = 800, 600
FPS = 60
# size = 20
WORM_SIZE = 20
WORM_SPEED = 18

WHITE = (0,0,0)
RED = (255, 0, 0)
UP = (0,-1)
DOWN = (0,1)
LEFT = (-1,0)
RIGHT = (1,0)
# Neural network configuration
input_size = 4 # Inputs: x position, y position, current direction x, current direction y, n food x, n food y, grow, shrink
hidden_size = 14
hidden_size2 = 14
output_size = 6  # Outputs: LEFT, RIGHT, UP, DOWN

# Create a genetic algorithm with a population of worms
population_size = 25

class Food:
    def __init__(self):
        self.position = (random.randint(0, WIDTH - WORM_SIZE), random.randint(0, HEIGHT-WORM_SIZE))
        self.color = (0, 255, 0)
    def spawn(self):
        self.position = (random.randint(0, WIDTH - WORM_SIZE), random.randint(0, HEIGHT-WORM_SIZE))
    def render(self, surface):
        pygame.draw.rect(surface, self.color, (self.position[0], self.position[1], WORM_SIZE, WORM_SIZE))

# Define a simple neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_size2):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Worm class
class Worm:
    def __init__(self, neural_network):
        self.length = 1
        self.position = (0, random.randint(0, HEIGHT))
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = (random.randint(0,255),random.randint(0,255), random.randint(0,255) )
        self.fitness = 0
        self.neural_network = neural_network
        self.speed = (WORM_SPEED*8)/(WORM_SIZE - 8+1)
        self.size = WORM_SIZE + 8
    def get_nearest_food_info(self, foods):
        # Find the nearest food item and return its x and y positions
        head_position = self.get_head_position()
        nearest_food = min(foods, key=lambda food: np.linalg.norm(np.array(head_position) - np.array(food.position)))
        x_distance = nearest_food.position[0] - head_position[0]
        y_distance = nearest_food.position[1] - head_position[1]
        return x_distance, y_distance
    
    def get_head_position(self):
        return self.position

    def move(self, foods):
        # Use the neural network to make a decision based on sensory inputs
        x_distance, y_distance = self.get_nearest_food_info(foods)
        head_position = self.get_head_position()
        inputs = [head_position[0] / WIDTH, head_position[1] / HEIGHT, self.direction[0], self.direction[1], x_distance, y_distance, self.size ]
        inputs = torch.tensor(inputs, dtype=torch.float32).view(1, -1)
        outputs = self.neural_network(inputs).detach().numpy()
        decision = np.argmax(outputs)

            
        # Update the direction based on the decision
        if decision == 0 and not np.array_equal(self.direction, RIGHT):
            self.direction = LEFT
        elif decision == 1 and not np.array_equal(self.direction, LEFT):
            self.direction = RIGHT
        elif decision == 2 and not np.array_equal(self.direction, DOWN):
            self.direction = UP
        elif decision == 3 and not np.array_equal(self.direction, UP):
            self.direction = DOWN
        elif decision == 4:
            if self.size < 25:
                self.size += 0.1
        elif decision == 5:
            if self.size > 2:
                self.size -= 0.1
        # Move the worm
        cur = self.get_head_position()
        x, y = self.direction
        self.position = (((cur[0] + (x * self.speed)) % WIDTH), (cur[1] + (y * self.speed)) % HEIGHT)

        # if len(self.positions) > 2 and new in self.positions[2:]:
        #     self.reset()
        # else:
        #     self.positions.insert(0, new)
        #     if len(self.positions) > self.length:
        #         self.positions.pop()

    def reset(self):
        self.length = 1
        self.position = ((WIDTH // 2), (HEIGHT // 2))
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.fitness = 0

    def render(self, surface):
        pygame.draw.rect(surface, self.color, (self.position[0], self.position[1], self.size, self.size))

# Genetic Algorithm class (modified to use PyTorch)
class GeneticAlgorithm:
    def __init__(self, population_size, input_size, hidden_size, output_size, hidden_size2):
        self.population_size = population_size
        self.population = [Worm(NeuralNetwork(input_size, hidden_size, output_size, hidden_size2)) for _ in range(population_size)]
        self.generation = 1

    def select_parents(self):
        # Select parents based on the worms' fitness
        sorted_population = sorted(self.population, key=lambda worm: worm.fitness, reverse=True)

        # Select the top two individuals as parents
        parents = sorted_population[:2]

        return parents

    def crossover(self, parent1, parent2):
        # Perform crossover to create a new child worm
        child = Worm(NeuralNetwork(input_size, hidden_size, output_size, hidden_size2), (parent1.length + parent2.length)//2)
        child.neural_network.load_state_dict(parent1.neural_network.state_dict())
        for param_name, param in child.neural_network.named_parameters():
            if random.random() < 0.5:
                param.data.copy_(parent2.neural_network.state_dict()[param_name].data)
        return child

    def mutate(self, worm):
        # Perform mutation on the neural network parameters
        mutation_rate = 0.1
        for param in worm.neural_network.parameters():
            if random.random() < mutation_rate:
                param.data += torch.randn_like(param.data)

    def evolve(self):
        # Evaluate fitness, select parents, perform crossover, and mutate

        # Select the top two parents based on fitness
        parents = self.select_parents()

        # Create a new population with the same size
        new_population = [Worm(NeuralNetwork(input_size, hidden_size, output_size, hidden_size2), 0) for _ in range(self.population_size)]

        # Generate offspring through crossover and mutation
        for i in range(len(new_population)-1):
            child1 = self.crossover(parents[0], parents[1])
            child2 = self.crossover(parents[1], parents[0])
            self.mutate(child1)
            self.mutate(child2)
            new_population[i] = child1
            new_population[i+1] = child2

        self.population = new_population
        self.generation += 1


# Main function
def main():
    # ... (Previous code for setting up Pygame)
    population_size = 25

    foods = [Food() for _ in range(5)]
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    genetic_algorithm = GeneticAlgorithm(population_size, input_size, hidden_size, output_size, hidden_size2)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for worm in genetic_algorithm.population:
            worm.move(foods)
            for food in foods:
                head_position = worm.get_head_position()
                if (
                    food.position[0] <= head_position[0] <= food.position[0] + worm.size
                    and food.position[1] <= head_position[1] <= food.position[1] + worm.size
                ):
                    # worm.length += 1
                    worm.fitness += 3  # Increase fitness when eating food
                    food.spawn()

        # for food in foods:
        #     food.spawn()
        surface.fill(WHITE)

        # Render and update worms
        for worm in genetic_algorithm.population:
            worm.render(surface)
        # Render and update foods
        for food in foods:
            food.render(surface)
        font = pygame.font.Font('freesansbold.ttf', 32)

        # create a text surface object,
        generation_text = font.render(f"Generation: {genetic_algorithm.generation}", True, (255, 0, 0))
        surface.blit(generation_text, (10, 10))

        screen.blit(surface, (0, 0))
        pygame.display.update()

        # Control the speed of the simulation
        pygame.time.Clock().tick(100)

        # Uncomment the next two lines if you want to see each generation in the console

        # Evolve the population after a certain number of frames
        if pygame.time.get_ticks() % (FPS*10) == 0:
            print(f"Generation: {genetic_algorithm.generation}, Max Fitness: {max([worm.length for worm in genetic_algorithm.population])}")
            genetic_algorithm.evolve()
        if genetic_algorithm.generation % 100 == 0 or genetic_algorithm.generation == 1:
            torch.save(genetic_algorithm.population[0].neural_network.state_dict(), 'generations/gen' + str(genetic_algorithm.generation) + '.pth')
        if genetic_algorithm.generation == 500:
            population_size = 100
if __name__ == "__main__":
    main()
