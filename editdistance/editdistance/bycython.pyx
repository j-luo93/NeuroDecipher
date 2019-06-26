# distutils: language = c++
# distutils: sources = editdistance/_editdistance.cpp

from libc.stdlib cimport malloc, free
# from libc.stdint cimport int64_t

from cython.parallel import prange
cimport numpy as np
import numpy as np
from cachetools import LRUCache

cdef extern from "./_editdistance.h" nogil:
    ctypedef int int64_t
    unsigned int edit_distance(const int64_t *a, const unsigned int asize, const int64_t *b, const unsigned int bsize)

cdef inline int64_t* hash_object(object x, unsigned int l):
    cdef int64_t *xl = <int64_t *>malloc(l * sizeof(int64_t))
    for i in range(l):
        xl[i] = hash(x[i])
    return xl

_DIST_CACHE = LRUCache(10000)
cpdef unsigned int eval(object a, object b):
    key = (a, b)
    if key in _DIST_CACHE:
        return _DIST_CACHE[key]

    cdef unsigned int i, dist
    cdef unsigned int len_a = len(a)
    cdef unsigned int len_b = len(b)
    cdef int64_t *al = hash_object(a, len_a)
    cdef int64_t *bl = hash_object(b, len_b)
    dist = edit_distance(al, len_a, bl, len_b)
    free(al)
    free(bl)
    _DIST_CACHE[key] = dist
    return dist

# DEF MAX_LEN = 1000000
cpdef object eval_all(object a_list, object b_list):
    cdef unsigned int i, j, l
    cdef unsigned int na = len(a_list)
    cdef unsigned int nb = len(b_list)
    # assert na <= MAX_LEN and nb <= MAX_LEN, 'na=%d, nb=%d' %(na, nb)
    dists_storage = np.zeros([na, nb], dtype='uint32')
    cdef np.uint32_t[:, ::1] dists = dists_storage
    # cdef unsigned int a_lens[MAX_LEN]
    # cdef unsigned int b_lens[MAX_LEN]
    # cdef int64_t *al[MAX_LEN]
    # cdef int64_t *bl[MAX_LEN]
    
    cdef unsigned int *a_lens = <unsigned int *>malloc(na * sizeof(unsigned int))
    cdef unsigned int *b_lens = <unsigned int *>malloc(nb * sizeof(unsigned int))
    cdef int64_t **al = <int64_t **>malloc(na * sizeof(int64_t))
    cdef int64_t **bl = <int64_t **>malloc(nb * sizeof(int64_t))
    for i in range(na):
        a = a_list[i]
        l = len(a)
        a_lens[i] = l
        al[i] = hash_object(a, l)
    for i in range(nb):
        b = b_list[i]
        l = len(b)
        b_lens[i] = l
        bl[i] = hash_object(b, l)
    with nogil:
        for i in prange(na, num_threads=12):
            for j in range(nb):
                dists[i, j] = edit_distance(al[i], a_lens[i], bl[j], b_lens[j])
    #free(al)
    #free(bl)
    return np.asarray(dists_storage, dtype='int64')

