import numpy as np
from sklearn.decomposition import TruncatedSVD
import numpy as np



class SVD:
    def __init__(self, n_components):
        self.n_components = n_components


    def fit(self, X):

        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)

        # Truncar las matrices U, Sigma y Vt según el número de componentes especificado
        U_trunc = U[:, :self.n_components]
        Sigma_trunc = np.diag(Sigma[:self.n_components])
        Vt_trunc = Vt[:self.n_components, :]

        self.components_ = Vt_trunc

        X_transformed = np.dot(U_trunc,Sigma_trunc)

        return X_transformed

    def transform(self, X):

        return np.dot(X,self.components_.T)

    def inverse_transform(self, X):
        return np.dot(X, self.components_)



"""
# Crear una matriz de ejemplo
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Especificar el número de componentes singulares a retener (ajusta según sea necesario)
n_componentes = 2

# Crear un objeto TruncatedSVD
svd = SVD(n_components=n_componentes)

# Aplicar TruncatedSVD a la matriz A
A_transformada = svd.fit_transform(A)
# Reconstruir la matriz original desde la transformación
A_reconstruida = svd.inverse_transform(A_transformada)

print("Matriz original:")
print(A)
print("\nMatriz transformada:")
print(A_transformada)
print("\nMatriz reconstruida:")
print(A_reconstruida)
print("\nsvd components:")
print(svd1.components_)
"""