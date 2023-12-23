import pygame
import random
import neat
import os
import pickle

# Definição de constantes para o jogo
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
DINO_WIDTH, DINO_HEIGHT = 30, 50
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 25, 25
WHITE, ORANGE, BLACK = (255, 255, 255), (255, 165, 0), (0, 0, 0)
FPS = 60

# Inicialização do Pygame
pygame.init()

# Classe para gerenciar o ranking dos jogadores
class RankingSystem:
    def __init__(self, max_ranking_size=10):
        self.max_ranking_size = max_ranking_size
        self.ranking = []

    def update_ranking(self, genome, score):
        self.ranking.append((genome, score))
        self.ranking.sort(key=lambda x: x[1], reverse=True)
        self.ranking = self.ranking[:self.max_ranking_size]

    def save_ranking(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.ranking, file)

    def load_ranking(self, file_path):
        with open(file_path, 'rb') as file:
            self.ranking = pickle.load(file)

    def get_ranking_data(self):
        return self.ranking.copy()

# Classe do dinossauro do jogo
class Dinosaur:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT - DINO_HEIGHT
        self.width, self.height = DINO_WIDTH, DINO_HEIGHT
        self.jump = False
        self.jump_height = 20
        self.gravity = 5
        self.score = 0

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and not self.jump:
            self.jump = True

    def move(self):
        if self.jump:
            self.y -= self.jump_height
            self.jump_height -= 1
            if self.jump_height < -20:
                self.jump = False
                self.jump_height = 20
        else:
            self.y += self.gravity
            if self.y > SCREEN_HEIGHT - self.height:
                self.y = SCREEN_HEIGHT - self.height

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, (self.x, self.y, self.width, self.height))

    def get_inputs(self, obstacles):
        if obstacles:
            obstacle = obstacles[0]
            return [self.x, self.y, obstacle.x - self.x, obstacle.height]
        return [self.x, self.y, SCREEN_WIDTH, 0]

# Classe do obstáculo do jogo
class Obstacle:
    def __init__(self, x, y, width=OBSTACLE_WIDTH, height=OBSTACLE_HEIGHT, speed=10):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.speed = speed

    def move(self):
        self.x -= self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, ORANGE, (self.x, self.y, self.width, self.height))

    def collides_with(self, dino):
        return (dino.x + dino.width > self.x and dino.x < self.x + self.width and
                dino.y + dino.height > self.y and dino.y < self.y + self.height)

# Função para desenhar o jogo na tela
def draw_game(screen, dinos, obstacles, score, font, ranking_system):
    screen.fill(BLACK)
    for dino in dinos:
        dino.draw(screen)
    for obstacle in obstacles:
        obstacle.draw(screen)

    text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(text, (10, 10))
    display_ranking(screen, ranking_system, font)
    pygame.display.update()

# Função para exibir o ranking
def display_ranking(screen, ranking_system, font):
    rankings = ranking_system.get_ranking_data()
    for i, (genome_id, score) in enumerate(rankings):
        text = font.render(f"Genoma {genome_id}: Score {score}", True, WHITE)
        screen.blit(text, (10, 30 * i + 40))

# Função principal do jogo
def main(genomes, config, use_neat=False, font=None, ranking_system=None):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    obstacles, nets, dinos = [], [], []
    score = 0

    if use_neat:
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            dinos.append(Dinosaur())

    running = True
    while running and dinos:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        for obstacle in obstacles[:]:
            obstacle.move()
            if obstacle.x < -obstacle.width:
                obstacles.remove(obstacle)

        remove_indices = []
        for i, dino in enumerate(dinos):
            if use_neat:
                net = nets[i]
                inputs = dino.get_inputs(obstacles)
                action = net.activate(inputs)
                if action[0] > 0.5:
                    dino.jump = True
            else:
                dino.handle_keys()

            dino.move()

            for obstacle in obstacles:
                if obstacle.collides_with(dino):
                    remove_indices.append(i)
                    break
            dino.score += 1

        for idx in sorted(remove_indices, reverse=True):
            if use_neat:
                genomes[idx][1].fitness = dinos[idx].score
                ranking_system.update_ranking(genomes[idx][1], dinos[idx].score)
            del dinos[idx]
            del nets[idx]

        score = max([dino.score for dino in dinos], default=0)

        if random.randint(0, 100) < 1.2:
            obstacle_height = random.choice([20, 30, 40])  # Variação na altura
            obstacle_speed = random.choice([8, 10, 12])   # Variação na velocidade
            obstacles.append(Obstacle(SCREEN_WIDTH, SCREEN_HEIGHT - obstacle_height, height=obstacle_height, speed=obstacle_speed))

        draw_game(screen, dinos, obstacles, score // FPS, font, ranking_system)
        clock.tick(FPS)

# Função para executar o algoritmo NEAT
def run_neat(config_file, font, ranking_system):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    for gen in range(50): # Número de gerações
        p.run(lambda genomes, config: main(genomes, config, use_neat=True, font=font, ranking_system=ranking_system), 1)

    ranking_system.save_ranking('ranking_file.pkl')

# Execução principal
if __name__ == "__main__":
    ranking_system = RankingSystem()
    font = pygame.font.SysFont(None, 36)
    use_neat_mode = True

    if use_neat_mode:
        config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
        run_neat(config_path, font, ranking_system)
    else:
        main(None, None, use_neat=False, font=font, ranking_system=ranking_system)
