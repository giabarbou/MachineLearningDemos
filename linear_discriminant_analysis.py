import numpy as np

class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.n_components_ = 0
        self.lambdas = None
        self.w = None
    
    @staticmethod
    def __solve_optimization_problem(Sw, Sb, n_components):

        # if Sw is singular it will return the pseudo-inverse
        Sw_inv = np.linalg.pinv(Sw)
        
        # solve the eigenvalue problem: Sw⁻¹Sb w = λ w
        eigenvalues, eigenvectors = np.linalg.eigh(Sw_inv @ Sb)
        
        idx = np.argsort(eigenvalues)[::-1]
        vals_sorted = eigenvalues[idx]
        vecs_sorted = eigenvectors[:, idx]

        lambdas = vals_sorted[:n_components]
        w = vecs_sorted[:, :n_components]

        return lambdas, w

    @staticmethod
    def __calculate_between_and_within_matrices(X, y):

        n_features = X.shape[1]
        classes = np.unique(y)

        overall_mean = np.mean(X, axis=0)
        
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))
        
        for c in classes:
            
            # within scatter matrix
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            Sw += (X_c - mean_c).T @ (X_c - mean_c)
            
            # between scatter matrix
            n_c = X_c.shape[0]
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            Sb += n_c * (mean_diff @ mean_diff.T)

        return Sb, Sw
    

    def fit(self, X, y):

        Sb, Sw = LDA.__calculate_between_and_within_matrices(X, y)
        self.lambdas, self.w = LDA.__solve_optimization_problem(Sw, Sb, self.n_components)
        self.n_components_ = self.lambdas.shape[0]

    
    def transform(self, X):
        return X @ self.w
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
