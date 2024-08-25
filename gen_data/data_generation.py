import igraph as ig
from igraph import Graph
from typing import List
import os
from pathlib import Path
from math import log2
from random import randint
from copy import deepcopy
from comparison.sat import solve_sat
from numpy.random import permutation

MAX_NODES = 100
MIN_NODES = 10

def generate_random_data(n : int, sizes : List[int], sparsity_levels : List[float]):
    batch = int(n / len(sizes) / len(sparsity_levels))

    return [Graph.Erdos_Renyi(n = size, p = (0.1 if size > 50 else sparsity)) for size in sizes for sparsity in sparsity_levels for _ in range(batch)]


def generate_clique_graph(n : int, sparsity : float):
    max_edges = int(1.1 * sparsity * n * (n - 1) / 2)
    min_edges = int(0.9 * sparsity * n * (n - 1) / 2)

    max_clique_size = int(log2(n))

    graph = Graph(n=n, directed=False)

    while True:
        clique_size = randint(1, max_clique_size)
        nodes = list(permutation(n))[0:clique_size]

        new_graph = deepcopy(graph)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not new_graph.are_adjacent(nodes[i], nodes[j]):
                    new_graph.add_edge(nodes[i], nodes[j])
        
        if new_graph.ecount() >= min_edges and new_graph.ecount() <= max_edges:
            return new_graph
        
        if new_graph.ecount() < min_edges:
            graph = new_graph
            continue

def generate_clique_data(n : int, sizes : List[int], sparsity_levels : List[float]):
    batch = int(n / len(sizes) / len(sparsity_levels))

    return [generate_clique_graph(size, 0.1 if size > 50 else sparsity) for size in sizes for sparsity in sparsity_levels for _ in range(batch)]

def graph_to_compressed(graph : Graph):
    adjacency : List[List[int]] = graph.get_adjacency()

    n = len(adjacency)

    adjacency = list(map(lambda adj : adj + [0 for _ in range(MAX_NODES - n)], adjacency))

    if n < MAX_NODES:
        adjacency += [[0 for _ in range(MAX_NODES)] for _ in range(MAX_NODES - n)]

    transform = lambda l: int("".join(list(map(str, l))), 2)

    adjacency = list(map(transform, adjacency))

    return adjacency

def to_one_hot(n : int):
    result = []

    while n > 0:
        result.append(n % 2)
        n = int(n / 2)

    l = len(result)

    result += [0 for _ in range(MAX_NODES - l)]
    return list(reversed(result))

def compressed_to_adj(l : List[int]):
    return [to_one_hot(elem) for elem in l]

def save_graphs(graphs : List[Graph], directory : str):
    for index, graph in enumerate(graphs):
        _, colors = solve_sat(graph)

        colors = [colors.get(i, 0) for i in range(MAX_NODES)]
        adjacency = graph_to_compressed(graph)

        filename = os.path.join('..', directory, f"graph{index}.txt")

        output_file = Path(filename)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with output_file.open('w') as fout:
            for line in adjacency:
                fout.write(f"{line} ")
            
            fout.write("\n")
            for color in colors:
                fout.write(f"{color} ")
        
        if index % 100 == 0:
            print(f"Completed graph with index {index}")

def load_graphs(directory : str, n : int):
    data = []
    colorings = []

    for index in range(n):
        adj = []

        filename = os.path.join('.', directory, f"graph{index}.txt")

        read_file = Path(filename)

        with read_file.open('r') as fin:
            adj = list(map(int, fin.readline().strip().split(' ')))
            colors = list(map(int, fin.readline().strip().split(' ')))

            data.append(compressed_to_adj(adj))
            colorings.append(colors)
    
    return data, colorings