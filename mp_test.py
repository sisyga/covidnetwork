import os
import multiprocessing as mp
from itertools import product
import numpy as np
from seir_model import SEIR_Simulator
from base import *
from networkx import to_numpy_array
from tqdm import tqdm

PATH = '.\\data\\SIR\\'

def iteration(args):
    index, kwargs = args
    # nw = SEIR_Simulator(printing=False, **kwargs)
    nw = DiseaseSimulator(printing=False, **kwargs)
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
    n = 100000
    p_inf = 0.02
    # ks = np.arange(2, 32, 2)  # average number of contacts
    # ps = np.logspace(-5, 0, num=11, endpoint=True)
    ks = np.logspace(np.log10(3), np.log10(100), dtype=int, num=14) * 2  # vary k on the log scale to show polynomial growth
    ps = [0]
    lp = len(ps)
    lk = len(ks)
    reps = 20  # number of repetitions of for each parameter
    graphtype = 'watts strogatz'
    np.savez(PATH+'params.npz', ps=ps, ks=ks, n=n, reps=reps, p_i=p_inf)
    graphparams = {'n': n}#, 'kcrit': np.inf}
    params = np.empty((lp, lk), dtype=object)
    for i, p in enumerate(ps):
        for j, k in enumerate(ks):
            params[i, j] = {'graphparams': {**graphparams}}
            params[i, j]['graphparams']['p'] = p
            params[i, j]['graphparams']['k'] = k

    paramstobeiterated = preprocess(params, reps, graphtype=graphtype, p_inf=p_inf)
    result = multiprocess(iteration, paramstobeiterated, processes=3)
    n_pr = np.empty(params.shape + (reps,), dtype=object)
    n_pr = postprocess(result, n_pr)
    np.save(PATH+'n_pr.npy', n_pr)
