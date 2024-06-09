from igraph import Graph

from pysat.solvers import Solver
from pysat.formula import CNF
from typing import List

def encode(k : int, index : int, color : int) -> int:
    return index * k + color + 1

def decode(k : int, encoding : int):
    return int((encoding - 1) / k), (encoding - 1) % k

def encode_graph(adj : List[List[int]], k : int):
    cnf = CNF()

    n = len(adj)

    # 3 types of constraints:

    # 1. A node must have at least one color
    # Clauses of the form xi0 V xi1 V ... V xi{k-1}

    for node in range(n):
        clause = [encode(k, node, color) for color in range(k)]

        cnf.append(clause)

    # 2. A node must have at most one color
    # Clauses of the form ~xik1 V ~xik2 for every k1 != k2

    for node in range(n):
        for color1 in range(k):
            for color2 in range(color1 + 1, k):
                cnf.append([-encode(k, node, color1), -encode(k, node, color2)])
    

    # 3. For each edge (i1, i2), there is the constraint of color(i1) != color(i2).
    # As ~x(i1, c) V ~x(i2, c) for every c for every i1 != i2 with edge(i1, i2)

    for i1 in range(n):
        for i2 in adj[i1]:
            if i2 <= i1:
                continue
                
            for color in range(k):
                cnf.append([-encode(k, i1, color), -encode(k, i2, color)])
    
    return cnf

def decode_coloring(n : int, k : int, encoding : List[int]):
    colors = {}

    for enc in encoding:
        if enc > 0:
            node, color = decode(k, enc)
            colors[node] = color

    return colors

def get_coloring(adj : List[List[int]], k : int):
    n = len(adj)
    cnf = encode_graph(adj, k)

    solver = Solver()
    solver.append_formula(cnf)

    if solver.solve():
        colors = decode_coloring(n, k, solver.get_model())
        return True, colors
    
    return False, None

def solve_sat(graph : Graph):
    n = graph.vcount()

    adjacency_list = [graph.neighbors(index) for index in range(n)]
    start = 1
    stop = n

    optimal_coloring = None
    optimal_count = None

    while start <= stop:
        med = int((start + stop) / 2)

        ok, coloring = get_coloring(adjacency_list, med)

        if ok:
            optimal_coloring = coloring
            optimal_count = med
        
            stop = med - 1
        else:
            start = med + 1
    
    return optimal_count, optimal_coloring