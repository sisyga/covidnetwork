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
        if n == 0:
            return

        else:
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
    n = round(3e5)

    k = 32
    graphparams = {'n': n, 'k': k, 'p': 1e-4}


    nw = SEIR_Simulator(graphtype='watts strogatz', graphparams=graphparams, p_inf=0.02)
    # nw.set_states_random(np.ceil(nw.ntot/1000).astype(int), 'infected')
    nw.set_states_random(81, 'infected')
    nw.set_states_random(33, 'exposed')
    nw.timeevo(tmax=100, recordfull=False)
    plt.plot(newcases_seir(nw.n_t), label='New cases')
    nw.plot_n_t()
    plt.show()
