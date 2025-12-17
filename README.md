# Model_Clustering

A public repository containing a Python implementation of **Gaffney’s model-based clustering** algorithm, adapted for **tropical cyclone (TC) trajectory analysis**.

This implementation follows the methodology described in Camargo et al. (2007, 2008) and applies polynomial regression–based mixture models to clustered cyclone tracks.

---

## Repository Overview

The core script in this repository is:

- **`LRM2.py`** — performs model-based clustering of tropical cyclone tracks using polynomial regression models.

---

## How to Use

The script `LRM2.py` requires **four input arguments**:

1. **`K`**  
   Integer specifying the number of clusters to fit.

2. **`Y`**  
   A NumPy array containing the concatenated cyclone track coordinates.  
   Each row corresponds to a point in a track, formatted as: (lon, lat)
   All tracks must be stacked sequentially into a single array.

3. **`seq`**  
A vector indicating track membership for each row in `Y`.  
Each value in `seq` identifies which cyclone track a given `(lat, lon)` point belongs to.

4. **`order`**  
Integer specifying the polynomial regression order. Recommended value for tropical cyclones: **2**  (See Camargo et al. 2007, 2008)

