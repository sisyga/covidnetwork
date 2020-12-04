import numpy as np
import os
import matplotlib.pyplot as plt
from param_inference import MyStochasticProcess, distance, distance2
from pyabc import ABCSMC, RV, Distribution, AggregatedTransition, DiscreteJumpTransition, MultivariateNormalTransition
from pyabc.visualization import plot_kde_matrix
from pyabc.transition import GridSearchCV

if __name__ == '__main__':
    n = int(3e5)
    k_domain = np.arange(1, 17)
    data = np.load('cases_de_feb26_mar16.npy') #* n / 1e6
    # print(data[0])
    # assert data < 3
    tmax = len(data)
    prior = Distribution(n01=RV('uniform', 0, 3 * round(data[0])),
                         n02=RV('uniform', 0, 10 * round(data[0])),
                         k=RV('randint', k_domain.min(), k_domain.max()+1),
                         log_p=RV('uniform', 0, 6),
                         p_inf=RV('uniform', 0.01, 0.07))
    model = MyStochasticProcess(n, tmax, data)
    transition = AggregatedTransition(mapping={
        # 'n01': DiscreteJumpTransition(domain=np.arange(int(data.max()))),
        # 'n02': DiscreteJumpTransition(domain=np.arange(10 * int(data.max()))),
        'k': DiscreteJumpTransition(domain=k_domain, p_stay=0.7),
        ('n01', 'n02', 'log_p', 'p_inf'): GridSearchCV()
    })

    db = "sqlite:///" + os.path.join(os.getcwd(), "early_growth.db")

    abc = ABCSMC(model, prior, distance, transitions=transition)
    abc.load(db, int(np.load('run_id_early_growth.npy')), {'cases': data})
    # history = abc.new(db, {'cases': data})
    abc.run(max_nr_populations=1)


    history = abc.history
    np.save('run_id_early_growth.npy', history.id)
    # print(history.max_t)
    df, w = history.get_distribution()
    plot_kde_matrix(df, w)
    plt.show()