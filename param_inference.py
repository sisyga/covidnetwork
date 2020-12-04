from pyabc.visualization import plot_kde_matrix
from pyabc import (ABCSMC, AdaptivePopulationSize,
                   RV, Distribution, AggregatedTransition, DiscreteJumpTransition, MultivariateNormalTransition,
                   IntegratedModel, ModelResult,
                   MedianEpsilon)
from pyabc.sampler import SingleCoreSampler, MappingSampler
import matplotlib.pyplot as plt
import os
import math
import tempfile
import pandas as pd
import numpy as np
from multiprocessing import Pool

import scipy.stats as st
from seir_model import SEIR_Simulator
from base import DiseaseSimulator

import os
#

class MyStochasticProcess(IntegratedModel):
    def __init__(self, n, tmax, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.tmax = tmax
        self.data = data

    def init_model(self, pars):
        graphparams = {'n': self.n, 'k': 2*round(pars['k']), 'p': 10 ** (-pars['log_p'])}
        # print(pars['n01'], pars['n02'], graphparams, pars['p_inf'])
        nw = SEIR_Simulator(graphtype='watts strogatz', graphparams=graphparams, p_inf=pars['p_inf'], printing=False)
        nw.set_states_random(round(pars['n02']), 'infected')
        nw.set_states_random(round(pars['n01']), 'exposed')
        return nw

    def integrated_simulate(self, pars, eps):
        if 'delay' in pars:
            delay = round(pars['delay'])
        else:
            delay = 0
        data = self.data[delay:self.tmax+delay]
        nw = self.init_model(pars)
        cumsum = 0
        history = np.zeros((self.tmax+1, 4), dtype=int)
        history[0] = nw.get_total_numbers()
        # casehistory = np.zeros(self.tmax+1)
        for t in range(1, self.tmax+1):
            nw.timestep()
            history[t] = nw.get_total_numbers()
            cases = history[t, 2] - history[t-1, 2] + history[t, 3] - history[t-1, 3]
            # casehistory[t] = cases
            cumsum += abs(cases - data[t-1])
            # cumsum = math.sqrt(np.sum((casehistory[1:t+1] - data[:t])**2))
            if cumsum > eps:
                # print('Run stopped at t = {} '.format(t) + 'with a distance of {}'.format(cumsum/t * self.tmax))
                return ModelResult(accepted=False)

        # print('run accepted with {}'.format(cumsum))
        return ModelResult(accepted=True,
                           distance=cumsum,
                           sum_stats={'cases': newcases_seir(history), 'delay': delay})
    def sample(self, pars):
        nw = self.init_model(pars)
        nw.timeevo(tmax=self.tmax, printprogress=False)
        return {'cases': newcases_seir(nw.n_t)}#, 'delay': round(pars['delay'])}


def newcases_seir(n_t):
    I = n_t[:, 2]
    R = n_t[:, 3]
    cases = I[1:] - I[:-1] + R[1:] - R[:-1]
    return cases

def newcases_sir(n_t):
    I = n_t[:, 1]
    R = n_t[:, 2]
    cases = I[1:] - I[:-1] + R[1:] - R[:-1]
    return cases

def seirrun(params):
    graphparams = {'n': n, 'k': 2*int(params['k']), 'p': 10 ** (-params['log_p'])}
    print(params['n01'], params['n02'], graphparams, params['p_inf'])
    nw = SEIR_Simulator(graphtype='watts strogatz', graphparams=graphparams, p_inf=params['p_inf'], printing=False)
    nw.set_states_random(int(params['n02']), 'infected')
    nw.set_states_random(int(params['n01']), 'exposed')
    nw.timeevo(tmax=tmax, recordfull=False, printprogress=False)
    return {'cases': newcases_seir(nw.n_t)}

def sirrun(params):
    graphparams = {'n': n, 'k': 2*int(params['k']), 'p': 10 ** (-params['log_p'])}
    print(params['n01'], graphparams, params['p_inf'])
    nw = DiseaseSimulator(graphtype='watts strogatz', graphparams=graphparams, p_inf=params['p_inf'], printing=False)
    nw.set_states_random(int(params['n01']), 'infected')
    nw.timeevo(tmax=tmax, recordfull=False, printprogress=False)
    return {'cases': newcases_sir(nw.n_t)}

def distance(x,y):
    return np.sum(np.abs(x['cases'] - y['cases']))

def distance2(x, y):
    return math.sqrt(np.sum((x['cases'] - y['cases'])**2))


def distance_delay(x, y):
    delay = round(x['delay'])
    tmax = len(x['cases'])
    data = y['cases']
    data = data[delay:tmax + delay]

    return np.sum(np.abs(x['cases'] - data))

if __name__ == '__main__':
    # os.environ['NUMEXPR_MAX_THREADS'] = '12'
    n = int(3e5)
    k_domain = np.arange(1, 11)
    data = np.load('casespm_de_mar16_jun6.npy') * n / 1e6
    tmax = len(data)

    prior = Distribution(n01=RV('uniform', 0, int(data.max())),
                         n02=RV('uniform', 0, 10 * int(data.max())),
                         k=RV('randint', 1, 7),
                         log_p=RV('uniform', 0, 6),
                         p_inf=RV('uniform', 0.01, 0.03))
    model = MyStochasticProcess(n, tmax, data)
    transition = AggregatedTransition(mapping={
        # 'n01': DiscreteJumpTransition(domain=np.arange(int(data.max()))),
        # 'n02': DiscreteJumpTransition(domain=np.arange(10 * int(data.max()))),
        'k': DiscreteJumpTransition(domain=k_domain, p_stay=0.7),
        ('n01', 'n02', 'log_p', 'p_inf'): MultivariateNormalTransition(scaling=0.8)})

    db = "sqlite:///" + os.path.join(os.getcwd(), "n=3e5_new.db")
    with Pool(processes=5) as pool:
        abc = ABCSMC(model, prior,
                     distance, transitions=transition,
                     sampler=MappingSampler(pool.map),  # SingleCoreSampler(),
                     # population_size=AdaptivePopulationSize(100, max_population_size=500),
                     # population_size=25
                     )
        # abc.load(db, np.load('run_id.npy'))
        abc.load(db, 10)
        # history = abc.new(db, {'cases': data})
        abc.run(max_nr_populations=10)
        pool.close()
        pool.join()

    history = abc.history
    np.save('run_id_n=3e5_new.npy', history.id)
    # print(history.max_t)
    df, w = history.get_distribution()
    plot_kde_matrix(df, w)
    plt.show()
