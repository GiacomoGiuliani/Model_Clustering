import numpy as np

class LRM:
    def __init__(self, Y, K, seq, order):
        '''
        Builder of the class. 
        Parameters to specify: 
        - K: number of clusters
        - Y: (N x 2) matrix containing all the observations along longitude and latitude
        - seq: vector containing the index corresponding to each track in Y 
        - Ord: integer specifying the order of my polynomial, usually 2
        - minvar: minimum accepted value for variance (diag vals of Sigma), value taken from Gaffney
        Parameters built here: 
        - X: Vandermonde matrix, obtained through Y (see the func)
        
        '''
        assert Y.shape[1] == 2, "Assertion failed: Input matrix Y must have a second dimension of size 2."  # (Lat, Lon) check

        self.K = K
        self.Y = Y
        self.seq = seq
        self.order = order
        self.minvar = 1e-5

        # WLS parameters
        self.mu = None
        self.sigma = None
        
        # Other attributes of the class
        self.scale = None

        # Variables to check 
        #print('Pik_test here')
        #self.Pik_test = Pik_test

        # Params to build
        self.X = self._vnd_mat()
        self.Pik = self._init_e()

        

    def _init_seq(self):
        """
        Creates a sequence vector where for each element in `idx_seq`,
        the corresponding range 0 to value (inclusive) is appended to the result.
        For instance, given 2 tracks of length 3 and 2, the result will be: [0, 1, 2, 0, 1].
         To ensure reproducibility with Gaffney's code, the time range HAS to start from 0. 
        """
        idx_seq = self.seq
        time_seq = []
        for value in idx_seq:
            time_seq.extend(range(value)) 
        return time_seq

    def _vnd_mat(self): 
        '''
        This function builds the Vandermonde matrix. Using this implementation from Stack: https://stackoverflow.com/a/48246009 as
        it runs much faster (70%) than the np.vander() function.
        
        Params: 
        - X: vector of size (N x 1) in a form like this (0,...,n1,0,...n2,0...nt)
        - N: polynomial order
        
        '''
        N = self.order + 1
        X = self._init_seq()
        Vand_mtx = (X ** np.arange(N)[:, None]).T
        
        return Vand_mtx

    def _sprod(self, mat, j):
        """
        Change the format of the Pik matrix. Pik is defined with the size of the sequence vector.
        This function transforms it to the size of the Y matrix, by repeating the values of Pik for each track.
        Watch out: it does not operate on the original n x K matrix, but on the n. points (N) x K matrix.
        Thus, given the first f points of the first track, this function will calculate the probability associated to each track, 
        which will be the product of the probabilities of the points in the track (f = f1 + ... + fn - where fi is the length of the ith track).
        
        Parameters:
        - mat: Matrix of shape (N, K) - Posterior probabilities.
        - j: Index of the column to compute the product for.
        
        Returns:
        - s: Array of shape (n,) - Product of the j-th column for each sequence.
        """
        # Extract the column corresponding to index `j`
        x = mat[:, j]
        
        # Check that the sum of the track points matches the size of the matrix
        assert np.sum(self.seq) == len(x), (
        f"Assertion failed: Cumulative sum of sequence ({np.sum(self.seq)}) "
        f"does not match the length of x ({len(x)}).")

        # Get the sequence lengths from self.seq
        lens = self.seq  
        n = len(lens)

        # Initialize the result array to store products
        s = np.ones(n)

        # Iterate over each sequence and compute the product
        current_index = 0  # To track the current starting position in x
        for i, length in enumerate(lens):
            # Compute the product for the current sequence
            seq_x = x[current_index:current_index + length]
            s[i] = np.prod(seq_x)
            

            # Update the starting index for the next track
            current_index += length

        return s


    def _weighted_least_squares(self, j):
        """
        Perform Weighted Least Squares (WLS) regression. It solves the problem using the QR decomposition. X,Y corresponds to the time and the track matrices. 
        W is the vector of the weights, obtained from the matrix Pik.
        The computation of beta is compiuted using the QR decomposition as the (X'WX)^(-1) (X'WY) would be computationally expensive.
        Since the QR decomposition is not unique, inverted signs in both matrices do not represent an issue for following calculations.

        Parameters:
        - X: Predictor matrix of shape (n, p) - Time observations (n. timesteps x polynomial order).
        - Y: Dependent variable matrix of shape (n, d) - Curves (n. timesteps x n.dims).
        - W: Weight vector of shape (n,), default is all ones (OLS) - We get the weights from the posterior probability Pik

        Returns:
        - b: Regression coefficients of shape (p, d).
        - sigma: Weighted variance of shape (d, d).
        """

        #print(f"here in wls")
        X = self.X
        Y = self.Y
        n, p = self.X.shape
        yn, d = self.Y.shape
        # Given the length of each track, change the size of the Pik matrix (weights) to be consistent with X,Y (n. tracks instead of n.clusters)
        W = np.repeat(self.Pik[:, j], self.seq)
        test_idx = [0, 9, 19, 29, 39, 49, 59, 89, 99]

        #print(np.take(W, test_idx))


        # Check timestep dimension in X,Y
        if n != yn:
            raise ValueError("The number of rows in X and Y must be the same.")

        # Default weights to 1 (ordinary least squares)
        if W is None:
            W = np.ones(n)
            
        # Check dimensions for weights vector
        W = np.asarray(W).flatten()
        if W.ndim != 1 or W.size != n:
            raise ValueError("W must be a vector of length equal to the number of rows in X.")

        # Precompute sqrt(weights) for efficiency 
        sqrt_W = np.sqrt(W)

        # Apply weights to X and Y
        # Element-wise multiplication as the weight matrix is assumed diagonal
        rtWmatd = np.tile(sqrt_W[:, None], (1, d))
        rtWmatp = np.tile(sqrt_W[:, None], (1, p))
        VX = np.multiply(rtWmatp, X)
        VY = np.multiply(rtWmatd, Y)

        #print(f"Let's now check Y {np.sum(Y)}, X {np.sum(X)}")
        #print(f"Let's now check VX {np.sum(VX)}, VY {np.sum(VY)}, sqrtWp {np.sum(rtWmatp)} and  sqrtWd {np.sum(rtWmatd)}")



        # QR decomposition for solving weighted least squares
        Q, R = np.linalg.qr(VX, mode='reduced')  # Reduced mode for efficiency
        b = np.linalg.solve(R, Q.T@VY)  # Regression coefficients

        # Weighted residuals
        residuals = Y - X @ b
    
        # Normalize variance
        df = np.sum(W)

        residuals_weighted = W[:, np.newaxis] * residuals
        sigma = (residuals_weighted.T @ residuals) / df

        # Return regression coefficients vector and vartiance
        return b, sigma

    def _mvnormpdf(self, j):
        '''
        This function calculates the multivariate gaussian density for each Pik. 
        It takes as input Y, X, Mu, Sigma, as the argument of the exponential is the trace of [(Z - TMu) Sigma.I (Z-TMu)'].
        '''
        
        Z = self.Y
        X = self.X @ self.mu[:, :, j]
        S = self.sigma[:, :, j]
        
        # Get the number of dimensions (it should be 2, lat and lon)
        D = self.Y.shape[1]

        assert D == 2., "Assertion failed: shape of the input matrix must be 2 (lat, lon)"
        
        # Compute the inverse and the determinant of the covariance matrix
        isigma = np.linalg.inv(S)
        dsigma = np.linalg.det(S)
        
        # Remove the mean 
        Z_c = Z - X
        
        # Now compute the density
        expo = -0.5 * np.sum(Z_c @ isigma * Z_c, axis=1)
        y = np.exp(expo) / np.sqrt((2*np.pi)**D * dsigma)
        
        return y

    def _init_e(self, mu=0.5):
        '''
        This function initializes the posterior probability matrix with random values using an exponential distribution. 
        
        Params: 
        K: number of clusters
        n: number of curves
        mu: mean of the exponential distribution (set to 0.5)
        '''
        
        n = len(self.seq)
        Pik = np.random.exponential(mu, (n, self.K))
        #Pik = self.Pik_test
        Pik /= np.sum(Pik, axis=1, keepdims=True) # Normalization of each row -> probabilities

        return Pik

    def _E_step(self):
        """
        This function performs the E step of the EM algorithm. It is executed from the 2nd iteration. 
        """
        
        # Extract parameters size
        P, D, K = self.mu.shape  
        if K != self.K:
            raise ValueError("The second dimensions of the mu parameter must correspond to the number of clusters")
        N, D = self.Y.shape      
        n = len(self.seq)   
        mlen = max(self.seq)  # Maximum length of a track

        # Compute Piik (likelihoods for each cluster) -> n. track points x n. clusters
        Piik = np.zeros((N, K)) 
        for k in range(K):
            Piik[:, k] = self._mvnormpdf(k)

        # Scale to have data in a suitable range
        self.scale = np.mean(Piik)
        Piik /= self.scale

        # Update Pik (membership probabilities)
        for k in range(self.K):
            self.Pik[:, k] = self._sprod(Piik, k)
        self.Pik *= self.Alpha

    def _M_step(self):
        '''
        This function performs the the maximization step of the algorithm, by computing the two parameters. 
        The procedure is equivalent to a weighted least squares, but we just keep diagonal elements in the sigma matrix.
        '''
        #print(f"here in M")
        X = self.X
        Y = self.Y
        N, K = self.Pik.shape

        # k is created in _init_e() or in _E_step() -> check its size
        if K != self.K:
            raise ValueError("The second dimensions of the mu parameter must correspond to the number of clusters")

        if N == self.Y.shape[0]:
            raise ValueError("The first dimensions of K must be different from the number of data points")

        self.Alpha = np.sum(self.Pik, axis=0) / N 
        
        # Initialize Mu and Sigma arrays (allocate memory)
        d1 = X.shape[1]
        d2 = Y.shape[1]
        self.mu = np.zeros((d1, d2, K))
        self.sigma = np.zeros((d2, d2, K))
        
        # Obtain wls parameters
        for k in range(K):
            self.mu[:, :, k], self.sigma[:, :, k] = self._weighted_least_squares(k)
            # apply twice np.diag to obtain a diagonal matrix (extra-diag elements assumed to be 0)
            self.sigma[:, :, k] = np.diag(np.diag(self.sigma[:, :, k])) 

    def _check_minvar(self):
        '''
        This function ensures that all the diagonal elements of the covariance matrix are greater than a certain threshold,
        in order to prevent numerical instabilities. 
        Params: 
        - Sigma: covariance matrix coming from the Maximization step
        - minvar: minimum variance value
        '''
        D, _, K = self.sigma.shape

        # Vectorized approach to find diagonal values below the minimum variance for all clusters
        for k in range(K):
            # Get the diagonal values for the k-th cluster
            diag_vals = np.diag(self.sigma[:, :, k])

            # Create a mask for values below the minvar
            mask = diag_vals < self.minvar

            # Replace these values with minvar
            # Update diagonal values directly in the covariance matrix
            for i in range(D):
                if mask[i]:
                    self.sigma[i, i, k] = self.minvar  # Set the diagonal element to minvar
        
    def _calc_like(self):
        """
        Compute the log-likelihood and normalize membership probabilities.
         The conditional likelihood is calculated as: L = ∏ (Σ (α_k * f_k)).

        Parameters:
        - N: Number of data points (used in scale adjustment).
        - progname: Name of the program for debugging/logging purposes.

        Returns:
        - Lhood: The computed log-likelihood.
        """
        n, K = self.Pik.shape 
        N = len(self.Y)
        
        # Check N != n (n. points != n. tracks)
        if n == N:
            raise ValueError("The number of rows in Pik and Y must be different.")

        
        # Sum of a_k * f_k for each cluster
        s = np.sum(self.Pik, axis=1) 
        
        # Handle zero-sum rows efficiently -> Assign a minimum value not to affect the overall result
        # Boolean mask for zero-sum rows
        zero_mask = (s == 0)  
        if np.any(zero_mask):
            realmin = np.finfo(float).tiny  # Smallest positive number in float
            self.Pik[zero_mask, :] = realmin * 1e100 * self.Alpha  # Broadcast Alpha directly
            s[zero_mask] = np.sum(self.Pik[zero_mask, :], axis=1)  # Update sums only for affected rows

        # Compute log-likelihood -> the second term accounts for the scale
        Lhood = np.sum(np.log(s)) + N * np.log(self.scale)

        # Normalize Pik (Bayes normalization)
        self.Pik /= s[:, np.newaxis]

        return Lhood
    
    def _stopping_cond(self, Lhood, iter_i, limit=50, stopval=1e-5):
        """
        This function checks the stopping condition of the algorithm.
        
        Parameters:
            -Lhood (list or array): Log-likelihood values at each iteration.
            - num_iter (int): Current iteration number.
            - limit: Maximum number of iterations allowed.
            - stopval: Threshold for convergence based on log-likelihood change.
        
        Returns:
            bool: True if the algorithm should stop, False otherwise.
        """
        num_iter = iter_i - 1
        if num_iter >= limit:
            print("Maximum number of iterations reached.")
            return True

        if num_iter > 1:
            if np.isnan(Lhood[num_iter]):
                print("The log-likelihood is equal to NaN.")
                return True
            
            if Lhood[num_iter] < Lhood[num_iter - 1]:
                print("The log-likelihood appears to have decreased on this iteration.")
                return True

            abs_change = Lhood[num_iter] - Lhood[0]
            if abs_change == 0:
                print("The log-likelihood has not changed.")
                return True
            
            delta = (Lhood[num_iter] - Lhood[num_iter - 1]) / abs_change
            if abs(delta) < stopval:
                print("The algorithm has converged.")
                return True
        
        return False
    
    def _permute_mod(self):
        """
        Permutes the dimensions of self.mu from [P,D,K] to [P,K,D].
        """
        self.mu = np.transpose(self.mu, (0, 2, 1))

    def mean_reg_line(self):
        '''
        This function computes the mean regression line for each cluster, 
        as the matrix product between the Vandermonde matrix and the 
        coefficient matrix. 
        '''

        T, PX = self.X.shape
        P, K, D = self.mu.shape

        # Extract the maximum length
        L = np.max(self.seq)
        XL = np.arange(1, L+1, 1)

        N = self.order +1

        assert L < T, f"Assertion failed: this track has more time steps than the size of the time dataset"

        # Build a non-cyclic Vandermonde matrix
        XV = np.vander(XL, N, increasing=True)

        # Check the dimensions of Mu
        assert N == P, f"Assertion failed: {PX} and {P} should be of the same size"
        assert K == self.K, f"Assertion failed: {K} should be equal to the number of clusters"

        # Initialize the mean regression vector
        Y_mean = np.zeros((L, D, K))

        # Compute the mrv for each cluster
        for kk in range(K):
            Y_mean[:, :, kk] = XV @ self.mu[:, kk, :]

        return Y_mean 


    def lrm_run(self):
        
        '''
        This function executes the LRM model. Some notes in the following lines. 
        - Why the stopping condition is placed before the M step: In many E-M algorithm applications, 
        the primary goal is to maximize the likelihood. Once the likelihood stabilizes, there’s a reasonable assumption that
        further iterations won't yield significant parameter changes. Since the likelihood reflects the quality of the fit,
        there might be no need to refine the parameters further when the stopping condition is met.
        - To obtain a robust classification, the algorithm should be run multiple times with different initializations.
        This refers to the random order of the tracks.

        Output: 
        - TrainLhood: log-likelihood values at the last iteration
        - C: cluster membership for each track

        '''

        # Pik already initialized in the class constructor -> perform the M step maximize the first (rnd) params
        self._M_step()

        print("Random initialization and first optimization done!")
        
        # Now,start the loop
        # Initialize variables (log-likelihood to evaluate convergence)
        NumIter = 0
        Lhood = []  
        while True:  
            NumIter += 1  
            
            # Ensure diagonal values are above the minimum threshold
            self._check_minvar()
            
            # E-Step
            self._E_step()
            
            # Evaluation of the Log-likelihood
            Lhood_iter = self._calc_like()
            
            # Depending on this value, we might stop the loop
            Lhood.append(Lhood_iter)
            
            # Check stopping condition
            if self._stopping_cond(Lhood, NumIter):
                print(f"Stopping condition met at iteration {NumIter}.")
                break

            # M Step
            self._M_step()

            if (NumIter)%5==0: 
                print("Opt. iteration n."+str(NumIter))

        print("Optimization complete!")

        # Permute the dimensions of mu
        self._permute_mod()
    
        # Save Lhood values up to the current iteration
        self.Lhood = np.array(Lhood[:NumIter])
        
        # Assign the cluster with the maximum posterior probability for each data point
        # Maximized membership along each row
        self.C = np.argmax(self.Pik, axis=1)
        print("clusters assigned!")
        
        # Calculate the total number of data points
        self.NumPoints = self.Y.size  
        
        # Store the final log-likelihood and the log-likelihood per data point
        self.TrainLhood = self.Lhood[-1]
        self.TrainLhood_ppt = self.TrainLhood / self.NumPoints
        
        # Calculate the number of independent parameters
        P, K, D = self.mu.shape # (pol. order, dimensions, n. clusters)
        self.NumIndParams = (K - 1) + K * P * D + K * D  # alpha, mu, sigma

