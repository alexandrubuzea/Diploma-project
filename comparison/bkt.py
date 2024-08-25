import igraph
from igraph import Graph
from typing import List

def check(adj : List[List[int]], colors) -> bool:
    if len(colors) == 0:
        return True

    current = len(colors) - 1

    color = colors[current]

    for nei in adj[current]:
        if nei not in colors:
            continue
        
        if colors[nei] == color:
            return False
    
    return True

def bkt(adj : List[List[int]], colors, k : int):
    node = len(colors)
    total_nodes = len(adj)

    ok = check(adj, colors)

    if ok and node == total_nodes:
        return True, colors
    
    if not ok:
        return False, {}
    
    for color in range(k):
        colors[node] = color
        ok, result = bkt(adj, colors, k)

        if ok:
            return True, result

        colors.pop(node)

    return False, {}

def solve_bkt(graph : Graph):
    adjacency_list = [graph.neighbors(index) for index in range(graph.vcount())]

    start = 1
    stop = len(adjacency_list)

    optimal_count = None
    optimal_coloring = None

    while start <= stop:
        med = int((start + stop) / 2)

        colors = {}
        ok, coloring = bkt(adjacency_list, colors, med)

        if ok:
            optimal_coloring = coloring
            optimal_count = med
            stop = med - 1
        else:
            start = med + 1
    
    return optimal_count, optimal_coloring

# graph = Graph.Erdos_Renyi(n=60, m=150)

# count, colors = solve_bkt(graph)

# print(graph)
# print(colors)
