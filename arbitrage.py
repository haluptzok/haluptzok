"""
Assume currency exchange is organized as an adjacency matrix conversion rate from i -> j = matrix[i][j]

Given a table of currency conversions - find out if there is an arbitrage opportunity.
Bellman-Ford can detect if there is an arbitrage opportunity
But you might want to know you are getting the best arbitrage opportunity - independent of path length
Or the best opportunity for each path length. Like 0->4->0 might be less return than 0->1->2->3->0, 
but should 0->4->0->4->0 considered in comparison?

I find the best non-looping path for each path length.  I give more than just the best, but not all the paths.

For simplicity of names use N,S,E,W

    N       S       E       W
N   1.0     2.0     4.0     8.0
S   0.5     1.0     0.333   5.0  
E   0.25    3.0     1.0     0.10
W   0.125   0.20    10.0    1.0

Essentially we have a graph with edges - we want to see if there is a cycle
that could get us a positive return.

A depth first search - where you don't repeat any nodes on the path- and check at each point
if you can return to start node works.  But it's (n!)

Bellman-Ford is V*E, E=V*V, so it's V^3 - but it seems to detect positive loops exist - not the exact best loop.
myArbitrage is V^4 - essentially Bellman-Ford for each node as the source - finding the best loop for each path length
Lot's of optimization not done - like curently pushing from all nodes 
instead of only nodes that are updated, early out check, etc.
"""
import sys
print("max float", sys.float_info.max)

def myArbitrage(matrix):
    num_currency = len(matrix)
    arbitrage_paths = []  # put shortest non-looping arbitrage paths in here, start and end in the same currency

    # Special case self-loop check for arbitrage
    # can diagonal be better than 1.0? - assuming diagonal <=1.0 so not checking.

    for iNodeStart in range(num_currency):  # starting at each node see if you can find arbitrage loops
        # initialize search from iNodeStart
        next_max_currency = [0.0] * num_currency  # Every currency has $0
        next_max_path = [[]] * num_currency       # Every currency has the empty path
        next_max_currency[iNodeStart] = 1.0       # Except currency[iNodeStart] = 1.00
        next_max_path[iNodeStart] = [iNodeStart]  # And it's path is just itself
        print(f"S1 iStep=  {next_max_currency=}")

        for iStep in range(num_currency - 1):  # Do it (num_currency - 1) steps so a complete traversal of all nodes is possible
            cur_max_currency = next_max_currency
            cur_max_path     = next_max_path
            next_max_currency = [0.0] * num_currency  # Every currency has $0
            next_max_path = [[]] * num_currency       # Every currency has the empty path

            for iNodeOut in range(num_currency):  # node to push out of
                for iNodeIn in range(num_currency): # node to push into
                    amt_currency = cur_max_currency[iNodeOut] * matrix[iNodeOut][iNodeIn]
                    # If we find a better path to this node, remember it.
                    if amt_currency > next_max_currency[iNodeIn]:
                        # But we are looking for unique paths - not looping back on ourself in smaller positive loops
                        if iNodeIn not in cur_max_path[iNodeOut]:
                            next_max_currency[iNodeIn] = amt_currency
                            next_max_path[iNodeIn] = cur_max_path[iNodeOut] + [iNodeIn]
            print(f"S1 {iStep=} {next_max_currency=}")
            # Do we have an arbitrage path in this step?
            for iNodeIn in range(num_currency): # node to push into
                amt_arbitrage = next_max_currency[iNodeIn] * matrix[iNodeIn][iNodeStart]
                if amt_arbitrage > 1.0:
                    new_path = next_max_path[iNodeIn] + [iNodeStart]
                    print(f"S4 {iStep=} {amt_arbitrage=} {next_max_currency=} {new_path=}")
                    if new_path[0] == min(new_path):  # On any loop - we only want loop once
                        # so only add the one where the first node is the minimum node
                        arbitrage_paths.append([len(new_path), amt_arbitrage, new_path])

    arbitrage_paths.sort(reverse=False)
    for path in arbitrage_paths:
        print("fnd_path:", path)
    
    arbitrage_paths.sort(reverse=False, key=lambda x: x[1])
    print(" ")

    for path in arbitrage_paths:
        print("bst_path:", path)

    return len(arbitrage_paths)

matrix = [
[1.0,     2.0,     4.0,     8.0],
[0.5,     1.0,     0.333,   5.0],
[0.25,    3.0,     1.0,     0.10],
[0.125,   0.20,    10.0,    1.0]
]

rates = [
    [1, 0.23, 0.25, 16.43, 18.21, 4.94],
    [4.34, 1, 1.11, 71.40, 79.09, 21.44],
    [3.93, 0.90, 1, 64.52, 71.48, 19.37],
    [0.061, 0.014, 0.015, 1, 1.11, 0.30],
    [0.055, 0.013, 0.014, 0.90, 1, 0.27],
    [0.20, 0.047, 0.052, 3.33, 3.69, 1],
]

rates_test1 = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]

rates_test2 = [
    [1.0, 1.1, 0.0],
    [0.0, 1.0, 1.1],
    [1.1, 0.0, 1.0]
]

rates_test3 = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 1.1],
    [0.0, 1.1, 1.0]
]

# This below is from https://anilpai.medium.com/currency-arbitrage-using-bellman-ford-algorithm-8938dcea56ea
# it doesn't find the best arbitrage path, it does detect if arbitrage is possible
# Doing the log for addition - doesn't handle 0 exchange rate transitions - I hacked a max in the log to make it sorta work
# I compared to mine to see if I had bugs - but it doesn't give best paths - but very similar approach.

from typing import Tuple, List
from math import log

# currencies = ('PLN', 'EUR', 'USD', 'RUB', 'INR', 'MXN')

def negate_logarithm_convertor(graph: Tuple[Tuple[float]]) -> List[List[float]]:
    ''' log of each rate in graph and negate it'''
    result = [[-log(max(edge,0.00000001)) for edge in row] for row in graph]
    return result

def arbitrage(rates_matrix: Tuple[Tuple[float, ...]]):
    ''' Calculates arbitrage situations and prints out the details of this calculations'''

    trans_graph = negate_logarithm_convertor(rates_matrix)

    # Pick any source vertex -- we can run Bellman-Ford from any vertex and get the right result

    n = len(trans_graph)
    min_dist = [float('inf')] * n  # This represents the smallest amount -log(0 + epsilon) = infinity (as epsilon goes to 0)
    pre = [-1] * n     # path is empty initially
    source = 0 # Start searching from the index 0 security
    min_dist[source] = source   # log(1) = 0, so start with $1 and push out.  We negated the log so finding the smallest is finding the biggest

    # 'Relax edges |V-1| times'
    for _ in range(n-1):
        for source_curr in range(n):
            for dest_curr in range(n):
                if min_dist[dest_curr] > min_dist[source_curr] + trans_graph[source_curr][dest_curr]:
                    min_dist[dest_curr] = min_dist[source_curr] + trans_graph[source_curr][dest_curr]
                    pre[dest_curr] = source_curr

    # if we can still relax edges, then we have a negative cycle
    for source_curr in range(n):
        for dest_curr in range(n):
            if min_dist[dest_curr] > min_dist[source_curr] + trans_graph[source_curr][dest_curr]:
                # negative cycle exists, and use the predecessor chain to print the cycle
                print_cycle = [dest_curr, source_curr]
                # Start from the source and go backwards until you see the source vertex again or any vertex that already exists in print_cycle array
                while pre[source_curr] not in  print_cycle:
                    print_cycle.append(pre[source_curr])
                    source_curr = pre[source_curr]
                print_cycle.append(pre[source_curr])
                print("Arbitrage Opportunity: \n")
                print(" --> ".join([str(p) for p in print_cycle[::-1]]))

print(myArbitrage(rates_test1))
print(myArbitrage(rates_test2))
print(myArbitrage(rates_test3))
print(myArbitrage(matrix))
print(myArbitrage(rates))

arbitrage(rates_test1)
arbitrage(rates_test2)
arbitrage(rates_test3)
arbitrage(matrix)
arbitrage(rates)

# Time Complexity: O(N^3)
# Space Complexity: O(N^2)
