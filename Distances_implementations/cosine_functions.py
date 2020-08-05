"""
This file contains different approaches to cosine distance and similarity.
The implementations are done for numpy, pandas or sparse matrices.
"""
""" Common DS libraries """
import numpy as np

""" Matrices libraries """
from scipy.spatial.distance import squareform, pdist
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import sklearn.preprocessing as sklpp
from sparse_dot_topn import awesome_cossim_topn

def cosine_similarity_on_rows_numpy_1(matrix):
    """
    This function computes cosine simmilarity for numpy arrays; not for sparse matrices
    Input:
        - matrix (np.array): matrix to compute the cosine distance in the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    row_norm_mat = sklpp.normalize(matrix, axis=1)
    return np.matmul(row_norm_mat, row_norm_mat.T)

# The following implementation is pretty slow
def cosine_similarity_on_rows_numpy_2(matrix):
    """
    Cosine simmilarity for numpy arrays. It does not work for sparse matrices formats.
    Input:
        - matrix (np.array): matrix to compute the cosine distance in the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    return 1 - squareform(pdist(matrix, metric='cosine'))


def cosine_dist_on_rows_numpy_3(matrix):
    """
    Cosine simmilarity for numpy arrays. It does not work for sparse matrices formats.
    Input:
        - matrix (np.array): matrix to compute the cosine distance in the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    norm = matrix / np.linalg.norm(matrix, axis=-1)[:, np.newaxis]
    return np.dot(norm, norm.T)


def cosine_similarity_on_rows_numpy_4(matrix):
    """
    Cosine simmilarity for numpy arrays. It does not work for sparse matrices formats.
    Input:
        - matrix (np.array): matrix to compute the cosine distance in the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    similarity = np.dot(matrix, matrix.T)
    # Number of occurrences
    square_mag = np.diag(similarity)
    # Inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # Inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # Cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return cosine.T * inv_mag


def cosine_similarity_on_rows_numpy_5(matrix):
    """
    Cosine simmilarity for numpy arrays.
    Input:
        - matrix (np.array): matrix to compute the cosine distance in the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    Anorm = matrix / np.linalg.norm(matrix, axis=-1)[:, np.newaxis]
    return linear_kernel(Anorm)

# It turns out to be faster for np.arrays, at least in low dimension.
def cosine_similarity_on_rows_numpy_and_csr_1(matrix):
    """
    Cosine simmilarity for numpy arrays and sparse matrices formats.
    Input:
        - matrix (np.array/sparse): matrix to compute the cosine distance in the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    return cosine_similarity(matrix)


# The following implementation use Dask and Cython in the back
def cosine_similarity_on_rows_numpy_and_csr_2(matrix, ntop=10):
    """
    Cosine similarity for numpy arrays and sparse matrices.
    It use Dask as well as Cython in the back, being faster than the others.
    Input:
        - matrix (np.array/sparse): matrix to compute the cosine distance in the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    mat = matrix.astype(np.float, copy=True)
    return awesome_cossim_topn(A=mat, B=mat.transpose(), ntop=ntop)


def cosine_similarity_on_rows_sparse_1(matrix):
    """
    Cosine similarities on rows for sparse matrices.
    Input:
        - matrix (np.array): matrix to compute the cosine distance in the rows
    Output:
        - matrix (sparse scipy): cosine distance matrix
    """
    similarity = matrix * matrix.T
    square_mag = np.array(matrix.sum(axis=1))
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitudes
    inv_mag = np.sqrt(inv_square_mag).T
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = np.array(similarity.multiply(inv_mag))
    return cosine * inv_mag.T


def cosine_similarity_cols_np(matrix):
    """
    Cosine distance for numpy arrays on columnss.
    Input:
        - matrix (np.array): matrix to compute the cosine distance in the columns
    Output:
        - matrix (np.array): cosine distance matrix
    """
    cols_norm_mat = sklpp.normalize(matrix, axis=0)
    return np.matmul(cols_norm_mat.T, cols_norm_mat)