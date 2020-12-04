from math import exp, ceil
import numpy as np
from numpy import random as npr
import networkx as nx
import random as rn
import itertools
from bisect import bisect


def get_householdsizes(n, p=0.5, dist=None):
    """
    Return a sample of the sizes of n households, either from a geometric distribution with parameter p, or from a fixed
    frequency distribution dist.
    :param n: int, number of households
    :param p: parameter of geometric distribution
    :param dist: frequencies of household sizes
    :return:
    """
    if dist is None:
        households = npr.geometric(p, size=n)
        return households

    else:
        dist = np.array(dist)
        try:
            households = npr.choice(np.arange(1, len(dist)+1), size=n, p=dist)

        except ValueError:
            dist = dist / dist.sum()
            households = npr.choice(np.arange(1, len(dist)+1), size=n, p=dist)

        return households

def get_companysizes(a, n, max=35000):
    """
    Draw the size of n companies from a Zipf distribution
    :param n: int, number of companies
    :param a: float, coefficient of zipf distribution
    :param max: float, maximum size, for Munich equal to BMW with roughly 35000 employees

    :return:
    """
    return npr.zipf(a, size=n)


def update_weights(degrees, kcrit):
    degrees = np.array(degrees)
    weights = degrees * np.exp(-degrees / kcrit)
    return weights



def watts_strogatz_scalefree(n, k, p, kcrit):
    """Returns a Watts–Strogatz small-world graph with preferential attachment.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node forms k new contacts. The average number of contacts is 2k.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    newman_watts_strogatz_graph()
    connected_watts_strogatz_graph()
    """
    if k > n:
        raise nx.NetworkXError("k > n, choose smaller k or larger n")

    #If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.Graph()
    nodes = np.arange(n)
    # weights = [deg * exp(-deg / kcrit) for _, deg in G.degree()]
    # weights = np.array(weights + [0.] * (n - k), dtype=float)
    weights = np.zeros(n)
    n_nonlocs = np.minimum(k, np.arange(n) - k)
    n_nonlocs[:k] = 0
    n_nonlocs = npr.binomial(n_nonlocs, p)   # get number of non-local edges
    for source in nodes:
        n_nonloc = n_nonlocs[source]
        n_loc = k - n_nonloc
        if n_loc > 0:
            localnodes = [i % n for i in range(source - k, source)] if source < k else range(source - k, source)
            loctargets = rn.sample(localnodes, n_loc)  # draw local targets
            G.add_edges_from(itertools.product([source], loctargets))  # add local edges
            weights[loctargets] = [deg * exp(-deg / kcrit) for _, deg in G.degree(loctargets)]

        while n_nonloc > 0:
            target = rn.choices(nodes, weights=weights)[0]
            while G.has_edge(source, target) or source == target:
                if G.degree(source) >= n - 1:
                    break

                target = rn.choices(nodes, weights=weights)[0]

            else:
                G.add_edge(source, target)
                keff = G.degree(target)
                weights[target] = keff * exp(-keff / kcrit)

            n_nonloc -= 1

        sourcedeg = G.degree(source)
        weights[source] = sourcedeg * exp(-sourcedeg / kcrit)
        source += 1

    return G


def clustered_scalefree(n, k, m, kcrit):
    """Returns a Watts–Strogatz small-world graph with preferential attachment.

    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is connected to its k nearest neighbors.
    m : int
        The number of out-of-household contacts

    See Also
    --------
    newman_watts_strogatz_graph()
    connected_watts_strogatz_graph()
    """
    if k > n:
        raise nx.NetworkXError("k > n, choose smaller k or larger n")


    # If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    m0 = int(ceil(m))
    G = nx.empty_graph(m0)
    nodes = list(range(n))
    weights = np.ones(n)
    rng = npr.default_rng()
    contacts = rng.poisson(lam=m, size=n)
    for source in range(m0, n):
        if contacts[source] > 0:
            ms = min((contacts[source], source))
            p = weights[:source] / weights[:source].sum()
            targets = npr.choice(nodes[:source], p=p, replace=False, size=ms)
            G.add_edges_from(zip([source] * ms, targets))

            weights[source] = ms + 1
            weights[targets] = [1 + deg * exp(-deg / kcrit) for _, deg in G.degree(targets)]

    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))

    return G

def household_scalefree(hhsizes, lam, kcrit):
    """
    Build a scalefree network with cutoff, based on fully connected cliques (households), where every node has k out-of-
    household contacts, where k is Poisson distributed with parameter p. The out-of-household contacts are grown
    sequentially, with an adapted Barabasi-Albert rule with exponential cut-off with parameter kcrit.
    :param hhsizes: list of ints, sizes of the fully-connected households
    :param lam: expected value of Poisson distribution for number of out-of-household contacts
    :param kcrit: critical number of contacts for cut-off
    :return:
    """
    G = nx.random_partition_graph(hhsizes, 1, 0)
    hhsizescumsum = np.array(hhsizes).cumsum()
    nx.set_node_attributes(G, {node: hhsizes[bisect(hhsizescumsum, node)] for node in G}, name='householdsize')
    n = G.number_of_nodes()

    rng = npr.default_rng()
    ks = rng.poisson(lam=lam, size=n)
    weights = np.ones(n, dtype=float)
    nodes = np.arange(n)
    for i, k in enumerate(ks):
        while k > 0:
            target = rn.choices(nodes, weights=weights)[0]
            while G.has_edge(i, target) or i == target:
                if G.degree(i) >= n - 1:
                    break

                target = rn.choices(nodes, weights=weights)[0]

            else:
                G.add_edge(i, target)
                hhsize = G.nodes[i]['householdsize']
                keff = G.degree(i) + 2 - hhsize
                weights[i] = keff * exp(-keff / kcrit)
                hhsize = G.nodes[target]['householdsize']
                keff = G.degree(target) + 2 - hhsize
                weights[target] = keff * exp(-keff / kcrit)

            k -= 1

    return G



