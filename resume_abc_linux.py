import numpy as np
import os
import matplotlib.pyplot as plt
from param_inference import MyStochasticProcess, distance
from pyabc import ABCSMC, RV, Distribution, AggregatedTransition, DiscreteJumpTransition, MultivariateNormalTransition, MedianEpsilon
from pyabc.visualization import plot_kde_matrix
from pyabc.transition import GridSearchCV

if __name__ == '__main__':
    n = int(3e5)
    data_prev = np.load('casespm_de_feb26_mar16.npy') * n / 1e6
    data = np.load('casespm_de_mar16_jun6.npy') * n / 1e6
    data_ext = np.load('casespm_de_mar16_jun21.npy') * n / 1e6
    tmax = len(data)
    kmin, kmax = 1, 11
    k_domain = np.arange(kmin, kmax+1)
    n01min, n01max = round(data[0]), round(data[:4].sum())

    n01_domain = np.arange(n01min, n01max+1)
    n02min, n02max = round(50 * data[0] / 10), round(50 * data[2])
    n02_domain = np.arange(n02min, n02max+1)
    delaymin, delaymax = 0, 10
    delay_domain = np.arange(delaymin, delaymax+1)
    prior = Distribution(n01=RV('uniform', n01min, n01max+0.5),
                         n02=RV('uniform', n02min, n02max+0.5),
                         k=RV('randint', kmin, kmax),
                         # delay=RV('randint', delaymin, delaymax+1),
                         log_p=RV('uniform', 0, 7),
                         p_inf=RV('uniform', 0.01, 0.03))
    model = MyStochasticProcess(n, tmax, data)
    transition = AggregatedTransition(mapping={
        # 'delay': DiscreteJumpTransition(domain=delay_domain),
        # 'n01': DiscreteJumpTransition(domain=n01_domain),
        # 'n02': DiscreteJumpTransition(domain=n02_domain),
        'k': DiscreteJumpTransition(domain=k_domain, p_stay=.8),
        ('n01', 'n02', 'log_p', 'p_inf'): GridSearchCV()
    })
    id = 'n=3e5_2'
    db = "sqlite:///" + os.path.join(os.getcwd(), id+".db")

    abc = ABCSMC(model, prior, distance,
                 transitions=transition, population_size=200,
                 )
    # abc.load(db, int(np.load('run_id_'+id+'.npy')))#, {'cases': data})
    abc.load(db, 1)
    # history = abc.new(db,  {'cases': data})
    np.save('run_id_'+id+'.npy', abc.history.id)

    abc.run(max_nr_populations=3)


    history = abc.history
    print(history.max_t)
    df, w = history.get_distribution()
    # kde = GridSearchCV().fit(df, w)
    # print(kde.rvs()['log_p'])
    # df['p'] = 10 ** (-df['log_p'])
    plot_kde_matrix(df, w)
    plt.show()