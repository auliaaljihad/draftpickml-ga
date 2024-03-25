import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np

def calculate_fitness(individual, pick_kawan, pick_musuh):
    # print("s_" + str(pick_kawan[0]))
    # print(individual)
    fitness_value = 0
    count = 0
    for index in individual:
        # print("Hero Random: ")
        # print(hero_data_clean[index]['Hero'])

        sinergis = 0
        counter = 0

        for kawan in pick_kawan_nama:
            sinergis = sinergis + hero_data_clean[index]['s_' + str(kawan)]
        for musuh in pick_musuh_nama:
            counter = counter + hero_data_clean[index]['s_' + str(musuh)]

        # print("Total Sinergis hero: "+ str(sinergis))
        # print("Total Counter hero: "+ str(counter))
        count += 1

        # print(hero_data_clean[index]['win_rate'])
        # print(hero_data_clean[index]['pick_rate'])
        # print(hero_data_clean[index]['ban_rate'])

        fitness_value += ((sinergis + counter) + 1) * (hero_data_clean[index]['win_rate'] + hero_data_clean[index]['ban_rate'] + hero_data_clean[index]['pick_rate'])
        # print(fitness_value)
    return fitness_value

def tournament_selection(population, fitness_values, tournament_size):
    # print(len(population))
    # print(population)
    selected_parents = []
    for _ in range(len(population)):
        tournament_candidates_indices = random.sample(range(len(population)), tournament_size)
        # print(tournament_candidates_indices)
        tournament_candidates_fitness = [fitness_values[i] for i in tournament_candidates_indices]
        # print(tournament_candidates_fitness)
        winner_index = tournament_candidates_indices[np.argmax(tournament_candidates_fitness)]
        # print(winner_index)
        selected_parents.append(population[winner_index])
    return selected_parents

def crossover(parent1, parent2):
    # print("hahaha")
    # print(len(parent1))
    crossover_point = random.randint(0, len(parent1) - 1)
    # print(crossover_point)
    # print("Parent 1: " + str(parent1))
    # print("Parent 2: " + str(parent2))
    # print("parent1[:crossover_point]: " + str(parent1[:crossover_point]))
    # print("parent1[crossover_point:]: " + str(parent1[crossover_point:]))
    # print("parent2[:crossover_point]: " + str(parent2[:crossover_point]))
    # print("parent2[crossover_point:]: " + str(parent2[crossover_point:]))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    # print("Child 1: " + str(child1))
    # print("Child 2: " + str(child2))
    return child1, child2

def mutate(individual, hero_data, mutation_rate):
    for i in range(len(individual)):
        if random.uniform(0, 1) < mutation_rate:
            # print("Lebih kecil dari 0.1")
            # print(individual)
            # print(individual[i])
            new_hero_index = random.choice(list(hero_data_clean.keys()))
            individual[i] = new_hero_index
            # individual[i] = individual[i]
            # print("hero baru:")
            # print(individual[i])

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
        # print("Fitness 1st: ")
        # print(fitness_values)
        # print("###########")

        # Keep track of the best individual in the current generation
        max_fitness = max(fitness_values)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_individual = population[np.argmax(fitness_values)]

        # Tournament selection
        parents = tournament_selection(population, fitness_values, tournament_size)
        # print("Pemenang Turnamen")
        # print(parents)
        # print("**************")
        # print(parents[::2])
        # print(parents[1::2])
        # Crossover
        children = []
        titik = 0
        for parent1, parent2 in zip(parents[::2], parents[1::2] + [None]):
            # print("Parent 1: Titik " + str(titik))
            # print(parent1)
            # print("Parent 2: Titik " + str(titik))
            # print(parent2)
            if parent2 is not None:
                if random.uniform(0, 1) < crossover_rate:
                    # print("dibawah crossover rate")
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
            titik += 1

        # print("\n&&&&&&&&&&&&&&")
        # print("Children: ")
        # print(children)

        # print("\n&&&&&&&&&&&&&&")
        # print("Child-child: ")
        # for child in children:
        #     print(child)

        # Mutation
        # print("\n#############")
        mutated_children = [mutate(child, hero_data, mutation_rate) for child in children]
        # print("\n#############")
        # print(mutated_children)

        # Combine old and new populations, then select the best individuals
        combined_population = population + mutated_children
        # print("\n******************")
        # print(combined_population)
        fitness_values_combined = [calculate_fitness(individual, pick_kawan, pick_musuh) for individual in combined_population]
        # print("\n******************")
        # print(fitness_values_combined)
        best_indices_combined = np.argsort(fitness_values_combined)[-pop_size:]
        # print("\n******************")
        # print(best_indices_combined)
        population = [combined_population[i] for i in best_indices_combined]
        # print("\n******************")
        # print(population)

        # Print results for each generation
        temp_result.append(best_fitness)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
        # print(temp_result)

    # Pilih individu terbaik dari populasi terakhir
    # print("\n&&&&&&&&&&&&&")
    # print(population)
    fitness_values = [calculate_fitness(individual, pick_kawan, pick_musuh) for individual in population]
    # print("\n&&&&&&&&&&&&&")
    # print(fitness_values)
    best_individual_index = np.argmax(fitness_values)
    # print("\n&&&&&&&&&&&&&")
    # print(best_individual_index)
    best_individual = population[best_individual_index]
    # print("\n&&&&&&&&&&&&&")
    # print(best_individual)


    return best_individual, temp_result


data=pd.read_csv('dataset/coba2.csv')
hero_data = data.to_dict(orient='index')
hero_data_clean = data.to_dict(orient='index')

# jumlah_pick = 1
# if jumlah_pick == 1:
#     data=pd.read_csv('/content/coba.csv')
#     hero_data = data.to_dict(orient='index')
#     hero_data_clean = data.to_dict(orient='index')
# else:
#     data=pd.read_csv('/content/coba2.csv')
#     hero_data = data.to_dict(orient='index')
#     hero_data_clean = data.to_dict(orient='index')

# print(len(hero_data))

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


# print(len(hero_data_clean))
        
df = pd.DataFrame(data)
total_win_rate = df['win_rate'].sum()
total_ban_rate = df['ban_rate'].sum()
total_pick_rate = df['pick_rate'].sum()
# print(total_win_rate)
# print(total_ban_rate)
# print(total_pick_rate)

# total = total_win_rate / (total_win_rate + total_ban_rate + total_pick_rate)
total = total_win_rate
# print(total)

# transposed_df = df.transpose()
# # total_win_rate = total_win_rate * transposed_df['Granger'].sum()


# print(transposed_df)

pop_size = 30
population = [random.sample(list(hero_data_clean.keys())[:-1], 2) for _ in range(pop_size)]

# population

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

# Plot fitness progress for both strategies on the same graph
# print(total)
# print(temp_result)
plt.plot((temp_result / total) * 100, label="Komposisi")
# plt.plot(temp_result * 100, label="Komposisi")

plt.title("Fitness Progress")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()  # Add legend to differentiate the lines
plt.show()