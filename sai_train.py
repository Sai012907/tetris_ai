from os import pardir
import numpy as np
from game import Game
from sai import Sai_AI
import random
import pandas as pd

def cross(a1, a2):

    new_weights = []
    
    prop = a1.fit_rel / a2.fit_rel
    
    for i in range(len(a1.weights)):
        rand = random.uniform(0, 1)
        if rand > prop:
            new_weights.append(a1.weights[i])
        else:
            new_weights.append(a2.weights[i])

    return Sai_AI(weights = np.array(new_weights), mutate = True)

def compute_fitness(agent, trials):

    fitness = []
    
    for i in range(trials):
        game = Game('genetic', agent = agent)
        peices_dropped, rows_cleared = game.run_no_visual()
        fitness.append(peices_dropped)
     
    return np.average(np.array(fitness))

def run_X_epochs(num_epochs = 5, num_trials = 5, pop_size = 50, num_elite = 5, survival_rate = .2, logging_file = 'data.csv'):

    data = [[np.ones(3)]]
    df = pd.DataFrame(data, columns = ['top_weight'])
    df.to_csv(f'data/{logging_file}', index = False)

    population = [Sai_AI() for k in range(pop_size)]

    for epoch in range(num_epochs):

        total_fitness = 0
        top_agent = 0
        weights = np.zeros(3)

        for n in range(pop_size):
 
            agent = population[n]
            agent.fit_score = compute_fitness(agent, num_trials)
            total_fitness += agent.fit_score 
            weights += agent.weights

        for agent in population:
            agent.fit_rel = agent.fit_score / total_fitness

        next_gen = []

        sorted_pop = sorted(population, reverse = True)

        elite_fit_score = 0
        elite_genes = np.zeros(3)
        top_agent = sorted_pop[0]

        for i in range(num_elite):
            elite_fit_score += sorted_pop[i].fit_score
            elite_genes += sorted_pop[i].weights
            next_gen.append(Sai_AI(weights = sorted_pop[i].weights, mutate = False))

        num_parents = round(pop_size * survival_rate)
        parents = sorted_pop[:num_parents]
        
        for k in range(pop_size - num_elite):

            parents = random.sample(parents, 2)
            next_gen.append(cross(parents[0], parents[1]))

        data = [[top_agent.weights]]
        df = pd.DataFrame(data, columns = ["top_weight"])
        df.to_csv(f'data/{logging_file}', mode = 'a', index = False, header = False)

        population = next_gen

    return data

run_X_epochs()
