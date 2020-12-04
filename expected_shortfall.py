import numpy as np
import os
import matplotlib.pyplot as plt
from param_inference import MyStochasticProcess, distance
from pyabc import ABCSMC, RV, Distribution, AggregatedTransition, DiscreteJumpTransition, MultivariateNormalTransition
from pyabc.visualization import plot_kde_matrix
from pyabc.transition.grid_search import GridSearchCV
from mp_test import *

def run(pars):
    index, pars = pars
    graphparams = {'n': int(pars['n']), 'k': 2 * int(pars['k']), 'p': 10 ** (-pars['log_p'])}
    # print(pars['n01'], pars['n02'], graphparams, pars['p_inf'])
    nw = SEIR_Simulator(graphtype='watts strogatz', graphparams=graphparams, p_inf=pars['p_inf'], printing=False)
    nw.set_states_random(int(pars['n02']), 'infected')
    nw.set_states_random(int(pars['n01']), 'exposed')
    nw.timeevo(recordfull=False, printprogress=False)
    return index, nw.get_nt()

if __name__ == '__main__':
    n = int(3e5)
    k_domain = np.arange(1, 11)
    data = np.load('casespm_de_mar16_jun6.npy') * n / 1e6
    tmax = len(data)
    prior = Distribution(n01=RV('uniform', 0, int(data.max())),
                         n02=RV('uniform', 0, 10 * int(data.max())),
                         k=RV('randint', 1, 11),
                         log_p=RV('uniform', 0, 6),
                         p_inf=RV('uniform', 0.01, 0.03))
    model = MyStochasticProcess(n, tmax, data)
    transition = AggregatedTransition(mapping={
        # 'n01': DiscreteJumpTransition(domain=np.arange(int(data.max()))),
        # 'n02': DiscreteJumpTransition(domain=np.arange(10 * int(data.max()))),
        'k': DiscreteJumpTransition(domain=k_domain, p_stay=0.7),
        ('n01', 'n02', 'log_p', 'p_inf'): MultivariateNormalTransition(scaling=0.7)})

    db = "sqlite:///" + os.path.join(os.getcwd(), "n=3e5_new.db")

    abc = ABCSMC(model, prior, distance, transitions=transition)
    abc.load(db, int(np.load('run_id_n=3e5_new.npy')))
    # abc.load(db, 10)
    # history = abc.new(db, {'cases': data})
    # abc.run(max_nr_populations=2)


    history = abc.history
    # np.save('run_id_n=3e5_new.npy', history.id)
    # print(history.max_t)
    df, w = history.get_distribution()
    kde = GridSearchCV().fit(df, w)
    params = kde.rvs(size=100)
    # for p in params['k']:
    #     print(round(p+0.5))
    plot_kde_matrix(df, w, kde=kde)
    plt.show()