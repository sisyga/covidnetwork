from base import *
import numpy as np
from scipy.stats import gamma

class SEIR_Simulator(DiseaseSimulator):
    statenames = ['susceptible', 'exposed', 'infected', 'recovered']
    colordict = {'susceptible': 'xkcd:grey',  'exposed': 'xkcd:dark yellow', 'infected': 'xkcd:red',
                  'recovered': 'xkcd:black'}
    altcolortable = {state: 'C{}'.format(i) for i, state in enumerate(statenames)}  # alternative colortable
    standardparameters = {'p_inf': 0.02,  # infection prob
                          't_i': 3,  # mean progression time to infectious
                          's_i': 1,  # std of progr. time to infectious
                          't_r': 10,  # mean progr. t. to recovered
                          's_r': 3  # std of progr. time to recovered
                            }

    def __init__(self, **params):
        DiseaseSimulator.__init__(self, **params)
        self.progtime = np.zeros(self.ntot)
        mti = self.params['t_i']
        varti = self.params['s_i']**2
        mtr = self.params['t_r']
        vartr = self.params['s_r']**2
        self.t_i_d = gamma(a=mti**2/varti, scale=varti/mti)
        self.t_r_d = gamma(a=mtr**2/vartr, scale=vartr/mtr)


    def set_states_random(self, n, state, randomize_waitingtime=False):
        """
        Set the state of 'n' randomly chosen individuals to 'statename'
        :param n: int, number of people, whose state is to be changed
        :param statename: New state, should be one of self.statenames
        :return:
        """
        inds = npr.choice(self.ntot, size=n, replace=False)
        if isinstance(state, str):
            state = self.statenames.index(state)

        self.states[inds] = state

        if state == 1:
            tis = self.t_i_d.rvs(n)
            if randomize_waitingtime:
                tis -= np.random.uniform(low=0, high=self.params['t_i']+self.params['s_i'], size=n)
            self.progtime[inds] = tis

        elif state == 2:
            tis = self.t_r_d.rvs(n)
            if randomize_waitingtime:
                tis -= np.random.uniform(low=0, high=self.params['t_r'], size=n)
            self.progtime[inds] = tis

    def stopcondition(self, n_t):
        n = n_t[-1]
        s = n[1] + n[2]
        return bool(s)

    def spread(self, i, newstates):
        for j in self.G[i]:
            if self.states[j] == 0:
                if random() < self.params['p_inf']:
                    newstates[j] = 1
                    self.progtime[j] = self.t_i_d.rvs()

    def timestep(self):
        newstates = self.states.copy()
        infected = self.states == 2
        exposed = self.states == 1
        for node in infected.nonzero()[0]:
            self.spread(node, newstates)
            self.counter[node] += 1
            if self.counter[node] >= self.progtime[node]:
                newstates[node] = 3
                self.counter[node] = 0
                self.progtime[node] = 0

        for node in exposed.nonzero()[0]:
            self.counter[node] += 1
            if self.counter[node] >= self.progtime[node]:
                newstates[node] = 2
                self.counter[node] = 0
                self.progtime[node] = self.t_r_d.rvs()

        self.states = newstates

def newcases_seir(n_t):
    I = n_t[:, 2]
    R = n_t[:, 3]
    cases = I[1:] - I[:-1] + R[1:] - R[:-1]
    return cases

if __name__ == '__main__':
    tmax = 100
    n = int(1e6)

    k = 6
    # p_inf = 2.5 / 10 / k
    # # p_edge = 0.75
    #
    # househouldsizes = [17333,	13983,	4923,	3748,	1390]  # household frequences in Munich
    # sizes = get_householdsizes(n, dist=househouldsizes)
    # n = sizes.sum()
    # k_outofhh = 9  # contacts out of the households
    # p_offblock = k_outofhh / n
    # p = np.ones((len(sizes), len(sizes)), dtype=np.float32) * p_offblock
    # p[np.diag_indices(len(p))] = 1
    graphparams = {'n': n, 'k': k, 'p': 1e-4}


    nw = SEIR_Simulator(graphtype='watts strogatz', graphparams=graphparams, p_inf=0.025)
    # nw.set_states_random(np.ceil(nw.ntot/1000).astype(int), 'infected')
    nw.set_states_random(635, 'infected')
    nw.set_states_random(60, 'exposed')
    # print(nx.algorithms.average_clustering(nw.G), get_meandegree(nw.G))
    nw.timeevo(tmax=100, recordfull=False)
    # plt.style.use('seaborn-bright')
    data = np.load('casespm_de_mar16_jun6.npy')
    plt.plot(data)
    plt.plot(newcases_seir(nw.n_t))
    nw.plot_n_t()
    # plt.title('Clustered powerlaw graph, $n = {}, k = {}, p = {}$'.format(*graphparams.values()))
    plt.show()
    # nw.draw_network(legend=True)
    # plt.savefig('model snapshot.pdf', dpi=600)
    # ani = nw.animate_network(legend=True)
    # ani.save('Clustered powerlaw graph k={}.mp4'.format(k))
    # plt.show()
