""" Common DS libraries """
import numpy as np


def jaccard_sim_cols_sparse(matrix):
    """
    This fuction computes Jaccard similarity among the columns of a csr matrix.
    It is faster for csc than csr
    Input:
        - matrix (sparse): matrix to compute the cosine distance in the columns
    Output:
        - matrix (sparse scipy): cosine distance matrix
    """
    # We use (a.b)/(a.a + b.b - a.b)
    matrix.astype(bool).astype(int)
    cols_sum = matrix.getnnz(axis=0)
    ab = matrix.transpose() * matrix
    aa = np.repeat(cols_sum, ab.getnnz(axis=0))
    bb = cols_sum[ab.indices]
    similarities = ab.copy()
    similarities.data = ab.data / (aa + bb - ab.data)
    return similarities


def jaccard_sim_rows_sparse(matrix):
    """
    This fuction computes Jaccard similarity among the rows of a mcsr matrix.
    Input:
        - matrix (sparse scipy): matrix to compute the cosine distance in the columns
    Output:
        - matrix (sparse scipy): cosine distance matrix
    """
    # We use (a.b)/(a.a + b.b - a.b)
    matrix.astype(bool).astype(int)
    rows_sum = matrix.getnnz(axis=1)
    ab = matrix * matrix.T
    aa = np.repeat(rows_sum, ab.getnnz(axis=1))
    bb = rows_sum[ab.indices]
    similarities = ab.copy()
    similarities.data = ab.data / (aa + bb - ab.data)
    return similarities


def jaccard_dist_rows_sparse(matrix):
    """
    This fuction computes Jaccard similarity among the rows of a mcsr matrix.
    It works for dense matrices from numpy in the case of all the entries having int as dtype.
    Input:
        - matrix (sparse scipy): matrix to compute the jaccard distance among the rows
    Output:
        - matrix (np.array): cosine distance matrix
    """
    # We use (a.b)/(a.a + b.b - a.b)
    matrix = matrix.astype(bool).astype(int)
    intersection = matrix.dot(matrix.T)
    row_sums = intersection.diagonal()
    unions = row_sums[:, None] + row_sums - intersection
    dist = 1.0 - intersection / unions
    return dist
