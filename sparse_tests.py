from Sparse_matrices.Utils_sparse_matrices import *
from scipy.sparse import csr_matrix
from scipy.sparse import rand
import os

mat = rand(10000, 900, density=0.1, format='csr')
print("\n")

print(density_checker(mat))

path = os.path.join(_download.LOCAL_CACHE_DIR, filename)