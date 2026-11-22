# fastsum.pyx
# cython: language_level=3

def sum_sq_cython(list lst):
    cdef double s = 0.0
    cdef double x
    for x in lst:          # типизированный цикл -> чистый C-код
        s += x * x
    return s
