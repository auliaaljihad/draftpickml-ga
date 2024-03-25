import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

def calculate_fitness(individual, pick_kawan, pick_musuh):
    fitness_value = 0
    count = 0
    hero_1 = "kosong"
    for index in individual:
        if count == 0:
            if hero_1 != index:
                hero_1 = index

        sinergis = 0
        counter = 0

        for kawan in pick_kawan_nama:
            sinergis = sinergis + hero_data_clean[index]['s_' + str(kawan)]
        if count == 1:
            sinergis = sinergis + hero_data_clean[index]['s_' + str(hero_data_clean[hero_1]['Hero'])]
        for musuh in pick_musuh_nama:
            counter = counter + hero_data_clean[index]['c_' + str(musuh)]
        count += 1

        fitness_value += ((sinergis + counter) + 1) * (hero_data_clean[index]['win_rate'] + hero_data_clean[index]['ban_rate'] + hero_data_clean[index]['pick_rate'])
    return fitness_value

def tournament_selection(population, fitness_values, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament_candidates_indices = random.sample(range(len(population)), tournament_size)
        tournament_candidates_fitness = [fitness_values[i] for i in tournament_candidates_indices]
        winner_index = tournament_candidates_indices[np.argmax(tournament_candidates_fitness)]
        selected_parents.append(population[winner_index])
    return selected_parents

def crossover(parent1, parent2):
    # crossover_point = random.randint(0, len(parent1) - 1)
    crossover_point = 1
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, hero_data, mutation_rate):
    for i in range(len(individual)):
        if random.uniform(0, 1) < mutation_rate:
            new_hero_index = random.choice(list(hero_data_clean.keys()))
            individual[i] = new_hero_index

    if len(individual) == 2:
        while individual[0] == individual[1]:
            random_hero = random.randint(0,1)
            new_hero_index = random.choice(list(hero_data_clean.keys()))
            individual[random_hero] = new_hero_index
    return individual

def genetic_algorithm(population, hero_data, generations, tournament_size, crossover_rate, mutation_rate, pop_size, pick_kawan, pick_musuh):
    temp_result = []
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitness_values = [calculate_fitness(individual, pick_kawan, pick_musuh) for individual in population]

        # Keep track of the best individual in the current generation
        max_fitness = max(fitness_values)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[np.argmax(fitness_values)]

        # Tournament selection
        parents = tournament_selection(population, fitness_values, tournament_size)

        # Crossover
        children = []
        for parent1, parent2 in zip(parents[::2], parents[1::2] + [None]):
            if parent2 is not None:
                if random.uniform(0, 1) < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                    children.append(child1)
                    children.append(child2)
                else:
                    # print("diatas crossover rate")
                    children.append(parent1)
                    children.append(parent2)
            else:
                # Handle the case where the number of parents is odd
                children.append(parent1)

        # Mutation
        mutated_children = [mutate(child, hero_data, mutation_rate) for child in children]

        # Combine old and new populations, then select the best individuals
        combined_population = population + mutated_children
        fitness_values_combined = [calculate_fitness(individual, pick_kawan, pick_musuh) for individual in combined_population]
        best_indices_combined = np.argsort(fitness_values_combined)[-pop_size:]
        population = [combined_population[i] for i in best_indices_combined]

        # Print results for each generation
        temp_result.append(best_fitness)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    # Pilih individu terbaik dari populasi terakhir
    fitness_values = [calculate_fitness(individual, pick_kawan, pick_musuh) for individual in population]
    best_individual_index = np.argmax(fitness_values)
    best_individual = population[best_individual_index]

    return best_individual, temp_result


data=pd.read_csv('dataset/dataset.csv')
hero_data = data.to_dict(orient='index')
hero_data_clean = data.to_dict(orient='index')

index_to_delete = ['Zilong', 'Zhask', 'Belerick', 'Karina', 'Uranus', 'Paquito', 'Kagura', 'Diggie', 'Hayabusa', 'Lapu-Lapu']
ban_hero = [index for index, hero in hero_data_clean.items() if hero['Hero'] in index_to_delete]

for index in ban_hero:
    if index in hero_data_clean:
        del hero_data_clean[index]
        # print("Entri telah dihapus.".format(index_to_delete))
    else:
        print('gagal')


pick_kawan_nama = ['Angela', 'Alice']
# pick_kawan_nama = []
pick_kawan = [index for index, hero in hero_data_clean.items() if hero['Hero'] in pick_kawan_nama]

for index in pick_kawan:
    if index in hero_data_clean:
        del hero_data_clean[index]
        # print("Entri telah dihapus.".format(index_to_delete))
    else:
        print('gagal')

pick_musuh_nama = ['Akai', 'Aldous']
# pick_musuh_nama = []
pick_musuh = [index for index, hero in hero_data_clean.items() if hero['Hero'] in pick_musuh_nama]

for index in pick_musuh:
    if index in hero_data_clean:
        del hero_data_clean[index]
        # print("Entri telah dihapus.".format(index_to_delete))
    else:
        print('gagal')
        
df = pd.DataFrame(data)
total_win_rate = df['win_rate'].sum()
total_ban_rate = df['ban_rate'].sum()
total_pick_rate = df['pick_rate'].sum()
total = total_win_rate

pop_size = 30
population = [random.sample(list(hero_data_clean.keys())[:-1], 2) for _ in range(pop_size)]

# Set algorithm parameters
generations = 30
tournament_size = 3
crossover_rate = 0.8
mutation_rate = 0.1

# Run genetic algorithm for hard engage
result, temp_result = genetic_algorithm(population, hero_data_clean, generations, tournament_size, crossover_rate, mutation_rate, pop_size, pick_kawan, pick_musuh)

# Print and visualize results
print("Individu Terbaik:")
for hero in result:
    print(f"Name: {hero_data_clean[hero]['Hero']}")

print(result)
plt.plot(temp_result, label="Komposisi")

plt.title("Fitness Progress")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()  # Add legend to differentiate the lines
plt.show()