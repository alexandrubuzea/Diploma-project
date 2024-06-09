from igraph import Graph
from typing import List

def get_coloring(adj : List[List[int]], k : int):
    n = len(adj)

    remaining_nodes = set(range(n))

    stack = []

    while len(remaining_nodes) > 0:
        found = False

        for node in remaining_nodes:
            if len(list(filter(lambda x: x not in stack, adj[node]))) < k:
                remaining_nodes.remove(node)
                stack.append(node)

                found = True
                break
        
        if not found:
            # heuristic: choose the node with the most restrictions
            node = max(list(map(lambda node: (len(list(filter(lambda nei: nei in remaining_nodes, adj[node]))), node), list(remaining_nodes))))[1]
            remaining_nodes.remove(node)
            stack.append(node)
    
    colors = {}

    added_nodes = set()

    for node in reversed(stack):
        neighbours_colors = set(map(lambda nei : colors[nei], list(filter(lambda x: x in adj[node], list(added_nodes)))))

        found = False
    
        for color in range(k):
            if color not in neighbours_colors:
                added_nodes.add(node)
                colors[node] = color
                found = True
                break
        
        if not found:
            return False, None

    return True, colors

def solve_heuristic(graph : Graph):
    n = graph.vcount()

    adjacency_list = [graph.neighbors(index) for index in range(n)]

    optimal_coloring = None
    optimal_count = None

    for k in range(1, n + 1):
        ok, coloring = get_coloring(adjacency_list, k)

        if ok:
            optimal_coloring = coloring
            optimal_count = k
            break
    
    return optimal_count, optimal_coloring