import numpy as np
from numpy import random as npr
from matplotlib import pyplot as plt
from random import random
from base import *
import networkx as nx
from network_setup import *


class CovidSimulator(DiseaseSimulator):
    """
    More sophisticated COVID-19-specific SEIR model.
    """
    statenames = ['susceptible', 'exposed', 'quarantine', 'no symptoms', 'symptoms', 'hospital', 'icu', 'dead',
                  'recovered']
    colortable = {'susceptible': 'xkcd:grey', 'contact': 'xkcd:teal', 'exposed': 'xkcd:brown',
                  'quarantine': 'xkcd:violet', 'no symptoms': 'xkcd:dark yellow', 'symptoms': 'xkcd:bright green',
                  'hospital': 'xkcd:orange', 'icu': 'xkcd:red', 'dead': 'xkcd:black', 'recovered': 'xkcd:green'}
    altcolortable = {state: 'C{}'.format(i) for i, state in enumerate(statenames)}  # alternative colortable
    standardparameters = {'p_inf': 0.1,  # infection prob
                          't_i': 3,  # progression time to infectious/no symptoms
                          't_s': 2,  # progr. time to symptoms
                          't_sr': 9,  # progr. t. to recovered
                          'p_sr': 0.955,  # prob for recovery from symptoms
                          't_sh': 4,  # p. time symptoms -> hospital
                          't_hr': 14,  # p. time hospital -> recovered
                          't_hicu': 2,  # p. time -> icu
                          'p_icu': 0.25,  # prob. hospital -> icu
                          't_icu': 10,  # time in icu
                          'p_d': 0.5,  # prob of death
                          'p_eq': 0.,  # probability of external quarantine
                          'p_sq': 0.25}  # probability of self-quarantine after symptoms

    def set_attributes(self):
        nx.set_node_attributes(self.G, 'susceptible', name='state')
        nx.set_node_attributes(self.G, 0, name='counter')
        nx.set_node_attributes(self.G, False, name='quarantine')

    def stopcondition(self, n_t):
        n = n_t[-1]
        s = n[1] + sum(n[3:7])
        return bool(s)

    def spread(self, i, newstates):
        for j in self.G[i]:
            if self.G.nodes[j]['state'] is 'susceptible' and not self.G.nodes[j]['quarantine']:
                if random() < self.params['p_inf']:
                    newstates[j] = 'exposed'

    def timestep(self):
        newstates = self.get_states().copy()
        newquarantines = nx.get_node_attributes(self.G, 'quarantine').copy()
        for node, data in self.G.nodes(data=True):
            state = data['state']
            if state is 'exposed':
                data['counter'] += 1
                if data['counter'] >= self.params['t_i']:
                    newstates[node] = 'no symptoms'
                    data['counter'] = 0

            elif state is 'no symptoms':
                data['counter'] += 1
                if not data['quarantine']:
                    self.spread(node, newstates)
                    # if random() < self.params['p_eq']:  # external quarantine, leave out for now
                    #     newquarantines[node] = True

                if data['counter'] >= self.params['t_s']:
                    newstates[node] = 'symptoms'
                    data['counter'] = 0
                    data['nextstate'] = 'recovered' if random() < self.params['p_sr'] else 'hospital'

            elif state is 'symptoms':
                data['counter'] += 1
                if not data['quarantine']:
                    self.spread(node, newstates)
                    if random() < self.params['p_sq']:
                        newquarantines[node] = True

                if data['nextstate'] is 'recovered':
                    if data['counter'] >= self.params['t_sr']:
                        newstates[node] = 'recovered'
                        data['nextstate'] = None
                        data['counter'] = 0
                        newquarantines[node] = False

                elif data['counter'] >= self.params['t_sh']:
                    newstates[node] = 'hospital'
                    data['counter'] = 0
                    newquarantines[node] = False
                    data['nextstate'] = 'icu' if random() < self.params['p_icu'] else 'recovered'

            elif state is 'hospital':
                data['counter'] += 1
                if data['nextstate'] is 'recovered':
                    if data['counter'] >= self.params['t_hr']:
                        newstates[node] = 'recovered'
                        data['counter'] = 0
                        data['nextstate'] = None
                        newquarantines[node] = False

                elif data['counter'] >= self.params['t_hicu']:
                    newstates[node] = 'icu'
                    data['counter'] = 0
                    data['nextstate'] = 'dead' if random() < self.params['p_d'] else 'recovered'
                    newquarantines[node] = False

            elif state is 'icu':
                data['counter'] += 1
                if data['counter'] >= self.params['t_icu']:
                    newstates[node] = data['nextstate']
                    data['counter'] = 0
                    data['nextstate'] = None
                    newquarantines[node] = False

        nx.set_node_attributes(self.G, newstates, name='state')
        nx.set_node_attributes(self.G, newquarantines, name='quarantine')

    def get_total_numbers(self):
        numbers = []
        states = self.get_states().values()
        for statename in self.statenames:
            if statename is 'quarantine':
                numbers.append(sum(nx.get_node_attributes(self.G, 'quarantine').values()))

            else:
                numbers.append(sum(state == statename for state in states))

        return numbers


if __name__ == '__main__':
    n = 1000
    househouldsizes = [17333,	13983,	4923,	3748,	1390]  # households of different sizes in Munich
    sizes = get_householdsizes(n, dist=househouldsizes)
    n = sizes.sum()
    betten = 34 * n // 100000
    k_normal = 12  # mean number of next-neighbor contacts
    R0 = 2.5
    Rnew = 1.1
    t_inf = 11
    p_inf = R0 / t_inf / k_normal  # infection prob R_0 / infectious time / contacts
    k_new = Rnew / t_inf / p_inf
    k_outofhh = 2
    p_edge = 0.1  # prob of rewiring
    p_offblock = k_outofhh / n
    p = np.ones((len(sizes), len(sizes))) * p_offblock
    p[np.diag_indices(len(p))] = 1

    nw = CovidSimulator(sizes=sizes, pmat=p, p_inf=p_inf, t_r=10, graphtype='random blocks', n=n, k=k_normal, p_edge=p_edge,
                        p_sq=0)

    nw.set_states_random(n, 'exposed')

    mdeg = get_meandegree(nw.G)
    nw.set_parameter('p_inf', R0 / mdeg / (nw.params['t_s'] + nw.params['t_sr']))
    nw.timeevo(recordfull=True)
    nw.draw_network(legend=True)
    plt.show()
    tmax = len(nw.n_t)
    #colors = [nw.colortable[state] for state in nw.s.values()]
    # nx.draw_networkx(nw.G, pos=pos, node_color=colors, with_labels=False)
    plt.style.use('seaborn-bright')
    nw.plot_n_t(altcolors=False)
    plt.gca().set_yscale('log')
    plt.hlines(betten, 0, tmax, colors='r', ls='--', label='ICU capacity')
    # plt.title('Watts-Strogatz small-world graph, $n = {}, k = {}, p = {}$'.format(nw.ntot, k, p_edge))
    plt.title('Random block graph of households, $n = {}$'.format(nw.ntot))
    plt.legend()
    plt.show()
    # ani = nw.animate_network(node_size=20, drawedges=True)
    # ani.save('small world sir animation p=1.mp4')
    # plt.show()