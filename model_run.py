# Define a function to run the LRM  model with several initializations
def model_run(data, seq, K, order, num_inits, lhood=False):
    """
    Run the LRM model with multiple initializations.

    Parameters:
    - data: array of objects containing the tracks
    - seq: array of objects containing the track lengths
    - K: Number of clusters.
    - order: Integer specifying the order of the polynomial, typically 2.
    - num_inits: Number of initializations to perform.

    Returns:
    - List of LRM objects.
    """

    # Initialize the list of LRM objects
    best_score = float("-inf")
    best_mbs = None
    best_idx = None
    best_mean_reg = None
    best_iter = None
    best_mu = None
    best_sigma = None

    
     # Set the seed
    np.random.seed(42)

    for i in range(num_inits+1): 

        indices = np.arange(len(data))

        # Print the iteration number
        print(f"Running iteration {i+1}...")

        if i == 0: 
            # First iteration with the original order
            shuff_data = data[indices]
            shuff_seq = seq[indices]

        else: 
            # Shuffle the tracks
            np.random.shuffle(indices)
    
            shuff_data = data[indices]
            shuff_seq = seq[indices]

    
        # Unpack the shuffled data
        Hur_Data = np.vstack(shuff_data)


        # Initialize the LRM model
        lrm_mod = LRM(Hur_Data, K, shuff_seq, order)

        # Run the model
        lrm_mod.lrm_run()

        # Print the log likelihood
        print(f"Log-Likelihood: {lrm_mod.TrainLhood_ppt}")

        # Update the best model
        if lrm_mod.TrainLhood_ppt > best_score: 
            best_score = lrm_mod.TrainLhood_ppt
            best_mbs = lrm_mod.C
            best_idx = indices
            best_mean_reg = lrm_mod.mean_reg_line()
            best_iter = i
            best_mu = lrm_mod.mu
            best_sigma = lrm_mod.sigma

    print(f" The best score is: {best_score} \n")
    print(f"Best model found at iteration {best_iter + 1}.")
    if lhood != False: 
        return best_mbs, best_idx, best_mean_reg, best_score
    else:    
        return best_mbs, best_idx, best_mean_reg
