import numpy as np

class KPCA:

    def __init__(self, n_components, kernel='rbf', coef0=1, gamma=1.0, degree = 2):

        self.n_components = n_components
        self.n_components_ = 0 # actual num of components

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        self.K_grand_mean = None
        self.K_row_means = None
        
        self.X_centered_train = None
        self.X_mean_train = None

        self.alphas = None

        self.kernel = kernel
        self.coef0 = coef0

    @staticmethod
    def __kernel_matrix_rbf(X, X_train = None, gamma=1.0):

        X_squared = np.sum(X**2, axis=1)
        
        if X_train is not None:
            X_squared_other = np.sum(X_train**2, axis=1)
        else:
            X_train = X
            X_squared_other = X_squared

        # Efficient computation of: ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2·x_i·x_j
        distances = X_squared[:, None] + X_squared_other[None, :] - 2 * X @ X_train.T 
        
        # exp(-γ ||x_i - x_j||²)
        K = np.exp(-gamma * distances)
        
        return K
    
    @staticmethod
    def __kernel_matrix_poly(X, X_train = None, degree=2, gamma=1.0, coef0=1):
        
        if X_train is None:
            X_train = X
        
        K = np.dot(X, X_train.T)
        K = (gamma * K + coef0) ** degree
        
        return K
    
    @staticmethod
    def __kernel_matrix_linear(X, X_train = None):

        if X_train is None:
            X_train = X
        
        K = np.dot(X, X_train.T)

        return K

    
    def __kernel_matrix(self, X, X_train = None):

        if self.kernel == 'poly':
            K = KPCA.__kernel_matrix_poly(X, X_train=X_train, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'linear':
            K = KPCA.__kernel_matrix_linear(X, X_train)
        else:
            K = KPCA.__kernel_matrix_rbf(X, X_train=X_train, gamma=self.gamma)

        return K
    
    @staticmethod
    def __find_num_components_based_on_desired_variance(eigenvalues, desired_variance):

        explained_variance = eigenvalues / eigenvalues.sum()
        cumulative_variance = np.cumsum(explained_variance)

        n_components = np.argmax(cumulative_variance >= desired_variance) + 1

        return n_components

    @staticmethod
    def __compute_coefficients(K_centered_train, n_components):

        # Solve the eigenvalue problem: Kv = λₖv
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered_train)

        # order by most to least significant
        idx = eigenvalues.argsort()[::-1]
        vals_sorted = eigenvalues[idx]
        vecs_sorted = eigenvectors[:, idx]

        if n_components < 1:
            n_components = KPCA.__find_num_components_based_on_desired_variance(eigenvalues, desired_variance=n_components)

        # keep n components
        lambdas = vals_sorted[:n_components]
        vecs = vecs_sorted[:, :n_components]

        # discard zero or negative values
        idx_positive = (lambdas > 0)
        lambdas = lambdas[idx_positive]
        vecs = vecs[:, idx_positive]

        # scale to get the actual coefficients:
        # ||v||² =  λₖ  ||α||² = 1 
        # ||α||² = 1/λₖ ||v||² 
        #   α    = 1/√λₖ  v
        alphas = np.sqrt(1 / lambdas) * vecs

        return alphas
    
    @staticmethod
    def __center_kernel_matrix_inplace(K):
        
        row_means = np.mean(K, axis=1)
        grand_mean = np.mean(K)
        
        K -= row_means
        K -= row_means[:, None]
        K += grand_mean
        
        return K, row_means, grand_mean
    
    def fit(self, X):

        # center the training data
        self.X_mean_train = np.mean(X, axis=0)
        self.X_centered_train = X - self.X_mean_train

        K = self.__kernel_matrix(self.X_centered_train)
        
        K_centered_train, self.K_row_means, self.K_grand_mean = KPCA.__center_kernel_matrix_inplace(K)

        self.alphas = KPCA.__compute_coefficients(K_centered_train, self.n_components)

        # calculate transformed samples
        self.Z_train = K_centered_train @ self.alphas

        self.n_components_ = self.Z_train.shape[1]

    def transform(self, X):

        X_centered_new = X - self.X_mean_train
        K = self.__kernel_matrix(X_centered_new, self.X_centered_train)

        K_row_means_new = np.mean(K, axis=1)
        K_centered_new = (K 
                          - K_row_means_new[:, None]
                          - self.K_row_means[None, :] 
                          + self.K_grand_mean)
        
        Z_new = K_centered_new @ self.alphas

        return Z_new

    def fit_transform(self, X):
        self.fit(X)
        return self.Z_train