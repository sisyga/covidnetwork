import numpy as np
from matplotlib import pyplot as plt
from pyabc.visualization import plot_kde_matrix
from param_inference import MyStochasticProcess, newcases_seir
from pyabc import (ABCSMC, RV, Distribution, AggregatedTransition, DiscreteJumpTransition, MultivariateNormalTransition,
                   ModelResult, IntegratedModel)
from pyabc.transition import GridSearchCV
from seir_model import SEIR_Simulator
from networkx import watts_strogatz_graph
import os
from param_inference import distance


def stopcondition(n):
    s = n[1] + n[2]
    return bool(s)

class ContinuedSpread(IntegratedModel):
    def __init__(self, n, t1, t2, data, history, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n
        self.t1 = t1  # check this again
        self.t2 = t2
        self.tmax = t2
        self.data = data
        df, w = history.get_distribution()
        self.kde = GridSearchCV().fit(df, w)

    def init_model(self, pars):
        graphparams = {'n': self.n, 'k': int(2*round(pars['k'])), 'p': 10 ** (-pars['log_p'])}
        nw = SEIR_Simulator(graphtype='watts strogatz', graphparams=graphparams, p_inf=pars['p_inf'], printing=False)
        nw.set_states_random(int(round(pars['n02'])), 'infected')
        nw.set_states_random(int(round(pars['n01'])), 'exposed')
        return nw

    def integrated_simulate(self, pars, eps):
        data = self.data
        nw = self.init_model(self.kde.rvs())
        nw.timeevo(tmax=self.t1, printprogress=False)
        firstcases = newcases_seir(nw.n_t)
        nw.G = watts_strogatz_graph(self.n, 2*round(pars['k']), 10 ** (-pars['log_p']))
        nw.set_parameter('p_inf', pars['p_inf'])
        cumsum = 0
        history = np.zeros((self.tmax+1, 4), dtype=int)
        history[0] = nw.get_total_numbers()
        for t in range(1, self.tmax+1):
            nw.timestep()
            history[t] = nw.get_total_numbers()
            if not stopcondition(history[t]):
                return ModelResult(accepted=False)

            cases = history[t, 2] - history[t-1, 2] + history[t, 3] - history[t-1, 3]
            cumsum += abs(cases - data[t-1])
            if cumsum > eps:
                # print('Run stopped at t = {} '.format(t) + 'with a distance of {}'.format(cumsum/t * self.tmax))
                return ModelResult(accepted=False)

        # print('run accepted with {}'.format(cumsum))
        return ModelResult(accepted=True,
                           distance=cumsum,
                           sum_stats={'cases': newcases_seir(history), 'firstcases': firstcases})

    def sample(self, pars):
        nw = self.init_model(self.kde.rvs())
        nw.timeevo(tmax=self.t1, printprogress=False)
        nw.G = watts_strogatz_graph(self.n, 2*round(pars['k']), 10 ** (-pars['log_p']))
        nw.set_parameter('p_inf', pars['p_inf'])
        nw.timeevo(tmax=self.tmax, printprogress=False)
        return {'cases': newcases_seir(nw.n_t)}


if __name__ == '__main__':
    n = int(3e5)
    rel = n / 1e6
    data1 = np.load('casespm_de_mar16_jun6.npy') * rel
    data2 = np.load('casespm_de_jun7_sep15.npy') * rel
    t1 = len(data1)
    print(t1)
    assert t1 == 0
    t2 = len(data2)
    kmin, kmax = 1, 11
    k_domain = np.arange(kmin, kmax+1)
    # print(len(data_ext[delaymax:tmax+delaymax]), tmax)
    prior = Distribution(k=RV('randint', kmin, kmax),
                         log_p=RV('uniform', 0, 7),
                         p_inf=RV('uniform', 0.01, 0.03))
    model = MyStochasticProcess(n, t1, data1)
    transition = AggregatedTransition(mapping={
        'k': DiscreteJumpTransition(domain=k_domain),
        ('log_p', 'p_inf'): GridSearchCV()
    })
    id_old = 'n=3e5_new'
    db_old = "sqlite:///" + os.path.join(os.getcwd(), id_old+".db")
    abc_old = ABCSMC(model, prior, distance)
    abc_old.load(db_old, int(np.load('run_id_'+id_old+'.npy')))
    model = ContinuedSpread(n, t1, t2, data2, abc_old.history)
    id = 'n=3e5_late'
    db = 'sqlite:///' + os.path.join(os.getcwd(), id+'.db')
    abc = ABCSMC(model, prior, distance, transitions=transition, population_size=200)
    # abc.load(db, int(np.load('run_id_'+id+'.npy')))
    # print(int(np.load('run_id_'+id+'.npy')))
    # abc.load(db, 3)
    history = abc.new(db,  {'cases': data2, 'firstcases': data1})
    # history = abc.history
    np.save('run_id_'+id+'.npy', history.id)

    abc.run(max_nr_populations=10)



    print(history.max_t)
    df, w = history.get_distribution()
    # kde = GridSearchCV().fit(df, w)
    # print(kde.rvs()['log_p'])
    # df['p'] = 10 ** (-df['log_p'])
    plot_kde_matrix(df, w)
    plt.show()