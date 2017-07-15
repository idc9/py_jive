jive
--------------------------------------

**author**: [Iain Carmichael](https://idc9.github.io/)

Additional documentation, examples and code revisions are coming soon. For questions, issues or feature requests please reach out to Iain: iain@unc.edu.

# Overview

Angle based Joint and Individual Variation Explained (AJIVE) is a dimensionality reduction algorithm for the multi-block setting i.e. K different data matrices, with the same set of observations and (possibly) different numbers of variables) **AJIVE finds *joint* modes of variation which are common to all K data blocks as well as modes of *individual* variation which are specific to each block.** For a detailed discussion of AJIVE see [Angle-Based Joint and Individual Variation Explained](https://arxiv.org/pdf/1704.02060.pdf). 

An R version of this package can be found [**here**](https://github.com/idc9/r_jive).

# Installation

Clone the repo:

```
git clone https://github.com/idc9/jive.git
python setup.py install
```
    
    
# Example

```python
import numpy as np
from jive.jive import jive
from jive.jive_visualization import plot_jive_full_estimates
	
# load some example data
X = np.load('data/toy_ajive_fig2_x.npy')
Y = np.load('data/toy_ajive_fig2_y.npy')
blocks = [X, Y]
	
# fit JIVE
jive = Jive(blocks, wedin_estimate=True)
jive.set_signal_ranks([2, 3]) # select signal ranks based on scree plot
	
# get all JIVE estimated data
block_estimates = jive.get_block_estimates()
	
# plot full JIVE estimates
plot_jive_full_estimates(jive)
```
For some more example code see [this notebook](doc/jive_demo.ipynb). 

# Help and Support


Additional documentation, examples and code revisions are coming soon. For questions, issues or feature requests please reach out to Iain: iain@unc.edu. 

#### Documentation

The source code is located on github: [https://github.com/idc9/py_jive](https://github.com/idc9/r_jive). Currently the best math reference is the [AJIVE paper](https://arxiv.org/pdf/1704.02060.pdf).

#### Testing

Testing is done using [nose](http://nose.readthedocs.io/en/latest/).

#### Contributing

We welcome contributions to make this a stronger package: data examples, bug fixes, spelling errors, new features, etc. <!-- TODO: add a more CONTRIBUTING file with more detail -->

#### Citation

A [Journal of Statistical Software](https://www.jstatsoft.org/index) is hopefully coming soon...


