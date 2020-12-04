import numpy as np
from numpy import random as npr
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import networkx as nx
from random import random
import matplotlib.cm
import sys
from network_setup import *

def get_meandegree(graph):
    """
    Calculate the mean degree of 'graph'
    :param graph: Networkx Graph
    :return: float: Average degree (= average number of links)
    """
    degrees = dict(graph.degree)
    ntot = len(degrees)
    return sum(degrees.values()) / ntot

def update_progress(progress):
    """
    Simple progress bar update.
    :param progress: float. Fraction of the work done, to update bar.
    :return:
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rProgress: [{0}] {1}% {2}".format("#" * block + "-" * (barLength - block), round(progress * 100, 1),
                                               status)
    sys.stdout.write(text)
    sys.stdout.flush()


def get_cmap_string(domain, palette='hsv'):
    domain_unique = np.unique(domain)
    n_c = len(domain_unique)
    hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
    mpl_cmap = matplotlib.cm.get_cmap(palette, lut=n_c)

    def cmap_out(x, **kwargs):
        mpl_cmap(hash_table[x], **kwargs)

    return cmap_out


class DiseaseSimulator():
    """
    Class to simulate a simple SIR model on a graph.
    """
    statenames = ['susceptible', 'infected', 'recovered']
    colordict = {'susceptible': 'xkcd:steel grey', 'infected': 'xkcd:red', 'recovered': 'xkcd:green'}
    colorarray = np.array(colordict.items)
    altcolortable = {state: 'C{}'.format(i) for i, state in enumerate(statenames)}  # alternative
    standardparameters = {'p_inf': 0.1, 't_r': 10}

    def __init__(self, printing=True, graph=None, graphtype='complete', graphparams={'n': 1000}, **kwparams):
        if graph is None:
            self.set_graph(graphtype, graphparams)

        else:
            self.G = graph
        self.printing = printing
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.ntot = self.G.number_of_nodes()
        self.states = np.zeros(self.ntot, dtype=np.uint8)
        self.counter = np.zeros(self.ntot)
        self.params = kwparams
        self.set_params()

    def set_graph(self, graphtype, graphparams):
        if graphtype == 'hexagonal':
            self.G = nx.triangular_lattice_graph(**graphparams)

        elif graphtype == 'watts strogatz':
            self.G = nx.watts_strogatz_graph(**graphparams)

        elif graphtype == 'newman watts strogatz':
            self.G = nx.newman_watts_strogatz_graph(**graphparams)

        elif graphtype == 'square':
            self.G = nx.grid_2d_graph(**graphparams)

        elif graphtype == 'random blocks':
            self.G = nx.stochastic_block_model(**graphparams)

        elif graphtype == 'powerlaw cluster':
            self.G = nx.powerlaw_cluster_graph(**graphparams)

        elif graphtype == 'scale-free small world':
            self.G = clustered_scalefree(**graphparams)

        elif graphtype == 'barabasi-albert':
            self.G = nx.barabasi_albert_graph(**graphparams)

        else:
            print('Using complete graph')
            self.G = nx.complete_graph(**graphparams)

    def set_parameter(self, param, value):
        if self.printing:
            print('Setting parameter ' + param + ' to {}'.format(value))
        self.params[param] = value

    def set_params(self):
        for p in self.standardparameters:
            if p not in self.params:
                self.set_parameter(p, self.standardparameters[p])

    def get_total_numbers(self):
        return np.bincount(self.states, minlength=len(self.statenames))

    def print_progress(self, t):
        text = '\rTime step {}: '.format(t)
        for statename, number in zip(self.statenames, self.n_t[-1]):
            text += statename + ': {}, '.format(number)
        text += 'of {} people'.format(self.ntot)
        sys.stdout.write(text)
        sys.stdout.flush()

    def stopcondition(self, n_t):
        return n_t[-1][1] != 0

    def timeevo(self, tmax=None, recordfull=False, printprogress=True):
        assert self.G.number_of_nodes() == self.ntot
        # self.update_fields()
        if tmax is None:
            self.n_t = []
            self.n_t.append(self.get_total_numbers())
            if recordfull:
                self.s_t = []
                self.s_t.append(self.get_states())
            t = 0
            while self.stopcondition(self.n_t):
                t += 1
                self.timestep()
                # self.update_fields()
                self.n_t.append(self.get_total_numbers())
                if printprogress:
                    self.print_progress(t)

                if recordfull:
                    self.s_t.append(self.states)
                    
            self.n_t = np.vstack(self.n_t)

        else:
            self.n_t = np.empty((tmax+1, len(self.statenames)), dtype=np.uint)
            self.n_t[0] = self.get_total_numbers()
            if recordfull:
                self.s_t = np.empty((tmax+1,)+self.states.shape, dtype=self.states.dtype)
                self.s_t[0] = self.states
            for t in range(1, tmax + 1):
                self.timestep()
                self.n_t[t] = self.get_total_numbers()
                if printprogress:
                    self.print_progress(t)
                if recordfull:
                    self.s_t[t] = self.states

    def get_nt(self):
        try:
            return self.n_t

        except:
            print('Not possible! Did you run timeevo?')

    def get_states(self):
        return self.states

    def set_states_random(self, n, state):
        """
        Set the state of 'n' randomly chosen individuals to 'state'
        :param n: int, number of people, whose state is to be changed
        :param statename: New state, should be one of self.statenames or an int in range(len(self.statenames))
        :return:
        """
        inds = npr.choice(self.ntot, size=n, replace=False)
        if isinstance(state, str):
            self.states[inds] = self.statenames.index(state)

        else:
            self.states[inds] = state

    def timestep(self):
        newstates = self.states.copy()
        for node in self.G.nodes():
            if self.states[node] == 1:
                self.spread(node, newstates)
                self.counter[node] += 1
                if self.counter[node] >= self.params['t_r']:
                    newstates[node] = 2
                    self.counter[node] = 0

        self.states = newstates

    def spread(self, i, newstates):
        for j in self.G[i]:
            if self.states[j] == 0:
                if random() < self.params['p_inf']:
                    newstates[j] = 1

    def plot_n_t(self, legend=True, altcolors=False, logy=True):
        n_t = self.get_nt()

        plots = []
        if altcolors:
            colors = self.altcolortable

        else:
            colors = self.colordict

        for i, statename in enumerate(self.statenames):
            plots.append(plt.plot(n_t[:, i], label=statename.title(), c=colors[statename]))

        plt.xlabel('Time (days)')
        plt.ylabel('Cases')
        if legend:
            plt.legend()
        if logy:
            plt.gca().set_yscale('log')

        return plots

    def draw_network(self, G=None, states=None, drawlabels=False, drawedges=True, fig=None, node_size=10,
                        layout='spring', legend=False, edgewidth=1, edgealpha=0.5, legendmarkersize=10,
                     legendmarkeredgewidth=2):
        if fig is None:
            fig = plt.figure()
        if G is None:
            G = self.G
        if states is None:
            try:
                states = self.s_t[-1]

            except AttributeError:
                states = self.get_states(graph=G)

        pos = nx.get_node_attributes(G, 'pos')
        d = dict(G.degree)

        if pos == {}:
            if layout == 'spring':
                pos = nx.spring_layout(G)

            elif layout == 'circular':
                pos = nx.circular_layout(G)

            else:
                pos = nx.random_layout(G)

        if drawlabels:
            nx.draw_networkx_labels(G, pos)

        if drawedges:
            nx.draw_networkx_edges(G, pos, width=edgewidth, alpha=edgealpha)

        if legend:
            custom_lines = [Line2D([0], [0], marker='o', color='none', markersize=legendmarkersize, markeredgecolor='k',
                                   markerfacecolor=self.colordict[state], lw=0, markeredgewidth=legendmarkeredgewidth)
                            for state in self.statenames]
            plt.legend(custom_lines, [name.title() for name in self.statenames])

        colors = [self.colordict[self.statenames[s]] for s in states]
        nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=[(s+1) * node_size for s in d.values()],
                                       edgecolors='k')
        plt.tight_layout(pad=0)
        return fig, nodes

    def animate_network(self, **kwargs):
        colors = [[self.colordict[self.statenames[s]] for s in states] for states in self.s_t]


        fig, nodes = self.draw_network(**kwargs)
        tmax = len(colors)
        title = plt.title('Time = 0 (d)')

        def update(step):
            title.set_text('Time = {} (d)'.format(step))
            nodes.set(facecolor=colors[step])
            return nodes, title

        return animation.FuncAnimation(fig, update, frames=range(1, tmax), interval=50, blit=False)


if __name__ == '__main__':
    tmax = 100
    n = 100
    p = 0.1
    k = 6
    graphparams = {'n': n, 'k': k, 'p': p}

    nw = DiseaseSimulator(graphtype='watts strogatz', graphparams=graphparams)
    nw.set_states_random(10, 'infected')
    nw.timeevo(tmax=20, recordfull=True)
    # nx.draw_networkx(nw.G, with_labels=False)

    plt.style.use('seaborn-bright')
    nw.plot_n_t()
    plt.title('Watts-Strogatz small-world graph, $n = {}, k = {}, p = {}$'.format(nw.ntot, k, p))
    plt.show()
    ani = nw.animate_network(legend=True)
    # ani.save('small world sir animation p=1.mp4')
    plt.show()
