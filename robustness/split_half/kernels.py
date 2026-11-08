"""Parallel cell accumulation; fixed split labels, no fastmath or model work."""
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=False)
def accumulate(x, labels, units, strata, target, sums, counts, ntc_sums, ntc_counts, centroids, pass_number):
    # Split ownership makes every write race-free. Source rows remain ordered.
    for s in prange(labels.shape[0]):
        for i in range(x.shape[0]):
            h = labels[s, i]
            sid = strata[i]
            if pass_number == 0:
                if not target[i]:
                    ntc_counts[s,h,sid] += 1
                    for d in range(x.shape[1]):
                        ntc_sums[s,h,sid,d] += np.float64(x[i,d])
            elif target[i]:
                uid = units[i]
                counts[s,h,uid] += 1
                for d in range(x.shape[1]):
                    sums[s,h,uid,d] += np.float64(x[i,d]) - centroids[s,h,sid,d]
