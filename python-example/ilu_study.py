"""
Script to compare performance of LU and ILU factorization using SciPy.
Input matrix can be loaded as an .npz file. Otherwise, a 2D disrete Laplacian
matrix is generated.

For usage help run: python3 ilu_study.py -h
"""

import os
import argparse
import math
import time

import numpy as np
import scipy as sp

def make_system_matrix(nrows_block):
    """
    Generate a linear system matrix based on a 2D Laplacian operator
    which has a block tridiagonal structure.

    Parameters
    ----------
    nrows_block : int
        The number of rows in each matrix block

    Returns
    -------
    A : scipy.sparse.csc_array
        The system matrix in CSC format
    """

    # 1D Laplacian operator matrix (Toeplitz matrix)
    K = sp.sparse.diags_array(
        [-1.0, 2.0, -1.0],
        offsets = [-1, 0, 1],
        shape=(nrows_block, nrows_block),
        format="csc"
    )

    # Kronecker sum of two 1D Laplacian operator matrices
    K2D = sp.sparse.csc_array(sp.sparse.kronsum(K, K))

    return K2D

def factorize(A, method, method_name=None, drop_tol=None, fill_factor=None):
    """
    Factorize a sparse matrix with a specified factorization method
    and print performance information.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        The sparse matrix to factorize in CSC format
    method : function
        The factorization method
    method_name : str
        The name of the factorization method
    drop_tol : float, optional
        The drop tolerance for ILU
    fill_factor : float, optional
        The fill ratio upper bound for ILU 
    """

    print('\n------------------------------')
    if method_name is not None:
        print(f'{method_name}:')
    else:
        print(f'{method.__name__}')
    print('------------------------------')

    t0 = time.perf_counter()

    if method == sp.sparse.linalg.spilu:
        lu = method(A, drop_tol, fill_factor)
    else:
        lu = method(A)

    t1 = time.perf_counter()

    t_factor = t1 - t0

    nrows = A.shape[0]

    t0 = time.perf_counter()

    Pr = sp.sparse.csc_array((np.ones(nrows), (lu.perm_r, np.arange(nrows))))
    Pc = sp.sparse.csc_array((np.ones(nrows), (np.arange(nrows), lu.perm_c)))

    error_norm = sp.sparse.linalg.norm(A - Pr.T @ (lu.L @ lu.U) @ Pc.T)

    t1 = time.perf_counter()

    t_norm = t1 - t0

    # Memory usage is estimated as the size of the double precision
    # floating point nonzero entries. There is other memory involved in
    # storing this matrix, but this should be a good approximation.
    mem_usage = ((lu.L.nnz + lu.U.nnz) * 8.0) / 1024.0**2

    print(f'Error norm   = {error_norm:.5e}')
    print(f'Factor time  = {t_factor:.5e} s')
    print(f'Norm time    = {t_norm:.5e} s')
    print(f'Memory usage = {mem_usage:.1f} MB\n')      


if __name__ == "__main__":

    # Argument handling
    parser = argparse.ArgumentParser(
        prog='python ilu_study.py',
        description='Test case comparing LU and ILU decomposition using SciPy',
    )

    parser.add_argument(
        '-b', '--blocks',
        default='400',
        help='Number of matrix blocks (total rows are the square of this).' +
             ' The default is 400.'
    )

    parser.add_argument(
        '--from-file',
        action='store_true',
        help='Read matrix from file instead of generating it.' +
             ' If both --blocks and --from-file are passed, the value of' +
             ' --blocks will be ignored.'
    )

    args = parser.parse_args()

    # Read or generate system matrix depending on command line argument 
    if args.from_file:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        A = sp.sparse.load_npz(
            os.path.join(script_dir, 'laplacian_2d_matrix.npz')
        )
        print('\nMatrix read from file. If you passed a value for --blocks,' +
              ' it will be ignored.')
    else:
        # Number of rows in a block is the same as the number of blocks
        nrows_block = int(args.blocks)
        A = make_system_matrix(nrows_block)
        print('\nGenerated system matrix.')

    # Print information about the system matrix
    mem_usage = (A.nnz * 8.0) / 1024.0**2
    print('\n------------------------------')
    print(f'System matrix info:')
    print('------------------------------')
    print(f'Dimension    = {A.shape[0]}')
    print(f'Nonzeros     = {A.nnz}')
    print(f'Memory usage = {mem_usage:.1f} MB\n')      

    # LU
    factorize(A, sp.sparse.linalg.splu, "Exact LU")

    # Incomplete LU
    drop_tol = 1e-4     # default is 1e-4
    fill_factor = 10    # default is 10
    factorize(A, sp.sparse.linalg.spilu, "Incomplete LU", drop_tol, fill_factor)
