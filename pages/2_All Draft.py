# streamlit_app.py
import streamlit as st
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from collections import defaultdict
import streamlit.components.v1 as components

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

        for hit in individual:
            sinergis = sinergis + hero_data_clean[index]['s_' + str(hero_data_clean[hit]['Hero'])]
        # sinergis = 0
        counter = 0

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
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, hero_data, mutation_rate):
    for i in range(len(individual)):
        if random.uniform(0, 1) < mutation_rate:
            new_hero_index = random.choice(list(hero_data_clean.keys()))
            individual[i] = new_hero_index

    unique_heroes = []
    for hero in individual:
        while hero in unique_heroes:
            hero = random.choice(list(hero_data_clean.keys()))
        unique_heroes.append(hero)

    individual = unique_heroes
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

st.set_page_config(
    page_title="Draft Pick Mythic Rank",
    page_icon="🧊",
    layout="centered", 
    )

data=pd.read_csv('dataset/dataset.csv')
hero_data = data.to_dict(orient='index')
hero_data_clean = data.to_dict(orient='index')

st.subheader('Dataframe')
st.dataframe(data, use_container_width=True)

# Set algorithm parameters
index_to_delete = st.multiselect(
    'Masukan hero yang diban',
    ['Aamon', 'Akai', 'Aldous', 'Alice', 'Alpha', 'Alucard', 
     'Angela', 'Argus', 'Arlott', 'Atlas', 'Aulus', 'Aurora', 
     'Badang', 'Balmond', 'Bane', 'Barats', 'Baxia', 'Beatrix', 
     'Belerick', 'Benedetta', 'Brody', 'Bruno', 'Carmilla', 'Cecilion', 
     'Chang’e', 'Chou', 'Cici', 'Claude', 'Clint', 'Cyclops', 'Diggie', 
     'Dyrroth', 'Edith', 'Esmeralda', 'Estes', 'Eudora', 'Fanny', 'Faramis', 
     'Floryn', 'Franco', 'Fredrinn', 'Freya', 'Gatotkaca', 'Gloo', 'Gord', 
     'Granger', 'Grock', 'Guinevere', 'Gusion', 'Hanabi', 'Hanzo', 'Harith', 
     'Harley', 'Hayabusa', 'Helcurt', 'Hilda', 'Hylos', 'Irithel', 'Ixia', 
     'Jawhead', 'Johnson', 'Joy', 'Julian', 'Kadita', 'Kagura', 'Kaja', 'Karina', 
     'Karrie', 'Khaleed', 'Khufra', 'Kimmy', 'Lancelot', 'Lapu-Lapu', 'Layla', 'Leomord', 
     'Lesley', 'Ling', 'Lolita', 'Lunox', 'Luo Yi', 'Lylia', 'Martis', 'Masha', 'Mathilda', 
     'Melissa', 'Minotaur', 'Minsitthar', 'Miya', 'Moskov', 'Nana', 'Natalia', 'Natan', 'Nolan', 
     'Novaria', 'Odette', 'Paquito', 'Pharsa', 'Phoveus', 'Popol and Kupa', 'Rafaela', 'Roger', 
     'Ruby', 'Saber', 'Selena', 'Silvanna', 'Sun', 'Terizla', 'Thamuz', 'Tigreal', 'Uranus', 'Vale', 
     'Valentina', 'Valir', 'Vexana', 'Wanwan', 'X.Borg', 'Xavier', 'Yi Sun-shin', 'Yin', 'Yu Zhong', 
     'Yve', 'Zhask', 'Zilong'],
    ['Aamon', 'Fredrinn', 'Guinevere', 'Moskov', 'Roger', 'Thamuz', 'Yin', 'Vexana', 'Khaleed', 'Lolita']
    )

ban_hero = [index for index, hero in hero_data_clean.items() if hero['Hero'] in index_to_delete]

for index in ban_hero:
    if index in hero_data_clean:
        del hero_data_clean[index]
        # print("Entri telah dihapus.".format(index_to_delete))
    else:
        print('gagal')

# Initialize input
pop_size = st.number_input('Ukuran Populasi', min_value=1, value=3)
population = [random.sample(list(hero_data_clean.keys())[:-1], 5) for _ in range(pop_size)]

generations = st.number_input('Jumlah Generasi', min_value=1, value=15)
tournament_size = st.number_input('Tournament Size', min_value=1, value=3)
crossover_rate = st.number_input('Crossover Rate', value=0.8, max_value=1.00)
mutation_rate = st.number_input('Mutation Rate', value=0.10, max_value=1.00)

pick_kawan_nama = []
pick_kawan = [index for index, hero in hero_data_clean.items() if hero['Hero'] in pick_kawan_nama]

pick_musuh_nama = []
pick_musuh = [index for index, hero in hero_data_clean.items() if hero['Hero'] in pick_musuh_nama]

# Run genetic algorithm
result, temp_result = genetic_algorithm(population, hero_data_clean, generations, tournament_size, crossover_rate, mutation_rate, pop_size, pick_kawan, pick_musuh)

df = pd.DataFrame(data)
total_win_rate = df['win_rate'].sum()
total_ban_rate = df['ban_rate'].sum()
total_pick_rate = df['pick_rate'].sum()
total = total_win_rate

chart_data = pd.DataFrame({
    "Generations": list(range(1, len(temp_result)+1)),
    "Fitness": (temp_result) 
})

st.subheader('Fitness Value')
st.line_chart(chart_data, x='Generations', y=["Fitness"], color=["#dc143c"], width=0, height=0, use_container_width=True)

col1 = st.columns(1)
st.subheader("Rekomenasi draft pick:")

result_df = pd.DataFrame(data)
result_df = result_df.drop(result_df.iloc[:, 1:247], axis=1)
hasil = []
for i in result:
    hasil.append(result_df.iloc[i])

st.dataframe(hasil, width=0, height=0, use_container_width=True)
fitness_tf = ((temp_result[-1]))
st.write(f"Fitness score: {fitness_tf:.2f}")