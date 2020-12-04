import os
import multiprocessing as mp
from itertools import product
import numpy as np
from seir_model import SEIR_Simulator
from base import *
from networkx import to_numpy_array
from tqdm import tqdm

# PATH = 'D:\\covid-modeling\\seir\\scalefree smallworld\\'
# PATH = 'D:\\covid-modeling\\seir\\watts strogatz 3\\'
# PATH = 'D:\\covid-modeling\\seir\\random graph\\'
PATH = './data/watts strogatz 3/'

def iteration(args):
    index, kwargs = args
    nw = SEIR_Simulator(printing=False, **kwargs)
    nw.set_states_random(1, 'infected')  # choose 1 agent of the population to be infected at t = 0
    nw.timeevo(tmax=365, recordfull=False, printprogress=False)
    # np.save(PATH + 'network_array_{}.npy'.format(index), to_numpy_array(nw.G, dtype=bool, weight=None))
    return index, nw.get_nt()


def preprocess(variablearray, reps, **constparams):
    params = {**constparams}
    paramstobeiterated = [(i + (j,), dict(params, **p)) for (i, p), j in product(np.ndenumerate(variablearray),
                                                                                 range(reps))]
    return paramstobeiterated


def multiprocess(function, iterator, **poolkwargs):
    with mp.Pool(**poolkwargs) as pool:
        results = list(tqdm(pool.imap_unordered(function, iterator), total=len(iterator)))
        pool.close()
        pool.join()

    return results


def postprocess(result, arr):
    for index, n_t in result:
        arr[index] = n_t

    return arr


if __name__ == '__main__':
    # househouldfreq = np.array([17333,	13983,	4923,	3748,	1390])  # number of households of different sizes
    # n_hh = 5000  # number of households
    # hh_sizes = get_householdsizes(n_hh, dist=househouldfreq)
    # n_tot = hh_sizes.sum()
    n = 100000
    # R0 = 2.5  # estimated basic reproduction number of COVID-19
    # t_r = 10
    # k = 8  # average number of daily contacts w/o restrictions [Mossong et al, 2008, PloS Medicine]
    # p_inf = R0 / t_r / k  # estimate of infection probability
    p_inf = 0.02
    ks = np.arange(2, 33, 2)  # average number of contacts outside of household
    # ps = k_offblock / n_tot  # probability to have edges between different households
    # ps = np.linspace(0, .5, num=11)  # expected number of non-local edges per node
    ps = np.logspace(-5, 0, num=11, endpoint=True)
    # ps = [1]
    lp = len(ps)
    lk = len(ks)
    reps = 20  # number of repetitions of for each parameter
    graphtype = 'watts strogatz'
    np.savez(PATH+'params.npz', ps=ps, ks=ks, n=n, reps=reps, p_i=p_inf)
    # pmats = []
    # for p in ps:
    #     pmat = np.ones((n_hh, n_hh)) * p
    #     pmat[np.diag_indices(n_hh)] = 1
    #     pmats.append(pmat)
    graphparams = {'n': n}#, 'kcrit': np.inf}
    params = np.empty((lp, lk), dtype=object)
    for i, p in enumerate(ps):
        for j, k in enumerate(ks):
            params[i, j] = {'graphparams': {**graphparams}}
            params[i, j]['graphparams']['p'] = p
            params[i, j]['graphparams']['k'] = k

    # params = {'p_inf': p_inf, 'graph_type': 'random blocks', 'sizes': hh_sizes}
    paramstobeiterated = preprocess(params, reps, graphtype=graphtype, p_inf=p_inf)
    result = multiprocess(iteration, paramstobeiterated, processes=4)
    n_pr = np.empty(params.shape + (reps,), dtype=object)
    n_pr = postprocess(result, n_pr)
    np.save(PATH+'n_pr.npy', n_pr)
