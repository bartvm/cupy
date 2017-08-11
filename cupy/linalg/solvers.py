import cupy
from cupy.cuda import cublas
from cupy.cuda import device


# TODO(okuta): Implement solve
def _getrs(a, b, OP):
    OPS = ('T', 'N')
    if OP not in OPS:
        raise ValueError

    # dtype checking
    if a.dtype != b.dtype:
        raise ValueError
    if a.dtype == 'f' or a.dtype == 'e':
        dtype = 'f'
        ret_type = a.dtype
    else:
        dtype = 'd'
        ret_type = 'd'

    # If a is row order, just apply transpose
    if a.flags.c_contiguous:
        OP = OPS[1 - OPS.index(OP)]
    elif not a.flags.f_contiguous:
        raise ValueError
    a = a.astype(dtype)

    # We need to copy b to column-order
    b = b.astype(dtype, order='F', copy=True)

    n, m = a.shape
    if n != m:
        raise ValueError
    k, nrhs = b.shape
    if n != k:
        raise ValueError
    lda = ldb = n

    if dtype == 'f':
        buffer_size = cupy.cuda.cusolver.sgetrf_bufferSize
        getrf = cupy.cuda.cusolver.sgetrf
        getrs = cupy.cuda.cusolver.sgetrs
    else:
        buffer_size = cupy.cuda.cusolver.dgetrf_bufferSize
        getrf = cupy.cuda.cusolver.dgetrf
        getrs = cupy.cuda.cusolver.dgetrs

    handle = device.Device().cusolver_handle

    if OP == 'N':
        op = cublas.CUBLAS_OP_N
    else:
        op = cublas.CUBLAS_OP_T

    info = cupy.empty((), 'i')
    work_size = buffer_size(handle, n, n, a.data.ptr, lda)
    work = cupy.empty(work_size, dtype)

    ipiv = cupy.empty(n, 'i')

    getrf(handle, n, n, a.data.ptr, lda, work.data.ptr,
          ipiv.data.ptr, info.data.ptr)
    getrs(handle, op, n, nrhs, a.data.ptr, lda,
          ipiv.data.ptr, b.data.ptr, ldb, info.data.ptr)

    return b.astype(ret_type)


def solve(a, b):
    return _getrs(a, b, 'N')


# TODO(okuta): Implement tensorsolve


# TODO(okuta): Implement lstsq


# TODO(okuta): Implement inv


# TODO(okuta): Implement pinv


# TODO(okuta): Implement tensorinv
