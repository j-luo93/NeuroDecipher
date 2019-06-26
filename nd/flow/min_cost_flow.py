import itertools
import logging
import random

import numpy as np
import torch
from cvxopt import matrix, solvers, spmatrix
from ortools.graph import pywrapgraph

from dev_misc import Map


def min_cost_flow(dists, demand, n_similar=None, capacity=1):
    '''
    Modified from https://developers.google.com/optimization/flow/mincostflow.

    ``capacity`` controls how many lost tokens can be mapped to the same known token.
    If it is set to -1, then there is no constraint at all, otherwise use its value.
    '''
    logging.debug('Solving flow')
    dists = (dists * 100.0).astype('int64')
    max_demand = min(dists.shape[0], dists.shape[1])
    if demand > max_demand:
        logging.warning('demand too big, set to %d instead' % (max_demand))
        demand = max_demand
    # between each pair. For instance, the arc from node 0 to node 1 has a
    # capacity of 15 and a unit cost of 4.
    nt, ns = dists.shape
    start_nodes = list()
    end_nodes = list()
    unit_costs = list()
    capacities = list()
    # source to c_t
    for t in range(nt):
        start_nodes.append(0)
        end_nodes.append(t + 2)  # NOTE 0 is reserved for source, and 1 for sink
        unit_costs.append(0)
        capacities.append(1)
    # c_s to sink
    for s in range(ns):
        start_nodes.append(s + 2 + nt)
        end_nodes.append(1)
        unit_costs.append(0)
        if capacity == -1:
            capacities.append(nt + ns)  # NOTE Ignore capacity constraint.
        else:
            capacities.append(capacity)
    # c_t to c_s
    if n_similar:  # and False:
        idx = dists.argpartition(n_similar - 1, axis=1)[:, :n_similar]
        all_words = set()
        for t in range(nt):
            all_s = idx[t]
            all_words.update(all_s)
        #    for s in all_s:
        #        start_nodes.append(t + 2)
        #        end_nodes.append(s + 2 + nt)
        #        unit_costs.append(dists[t, s])
        if len(all_words) < demand:
            logging.warning('pruned too many words, adding some more')
            added = random.sample(set(range(ns)) - all_words, demand - len(all_words))
            all_words.update(added)
        #    for s in added:
        #        for t in range(nt):
        #            start_nodes.append(t + 2)
        #            end_nodes.append(s + 2 + nt)
        #            unit_costs.append(dists[t, s])
        for t, s in itertools.product(range(nt), all_words):
            start_nodes.append(t + 2)
            end_nodes.append(s + 2 + nt)
            unit_costs.append(dists[t, s])
            capacities.append(1)

    else:
        for t, s in itertools.product(range(nt), range(ns)):
            start_nodes.append(t + 2)
            end_nodes.append(s + 2 + nt)
            unit_costs.append(dists[t, s])
            capacities.append(1)

    # Define an array of supplies at each node.
    supplies = [demand, -demand]  # + [0] * (nt + ns)

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(
            int(start_nodes[i]),
            int(end_nodes[i]),
            int(capacities[i]),
            int(unit_costs[i]))

    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow between node 0 and node 4.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        cost = min_cost_flow.OptimalCost()
        flow = np.zeros([nt, ns])
        for i in range(min_cost_flow.NumArcs()):
            t = min_cost_flow.Tail(i)
            s = min_cost_flow.Head(i)
            if t > 1 and s > 1 + nt:
                flow[t - 2, s - 2 - nt] = min_cost_flow.Flow(i)
        return flow, cost
    else:
        logging.error('There was an issue with the min cost flow input.')
        raise RuntimeError('Min cost flow solver error')
