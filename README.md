# Angle-Based Joint and Individual Variation Explained

Python implementation of the [Angle-Based Joint and Individual Variation Explained](https://arxiv.org/pdf/1704.02060.pdf) (JIVE) algorithm. For an introduction to JIVE see [this page](/doc/jive_explaination.ipynb).

TODO: 

- add algorithm documentation
- add walk through with code + plots

# Installation instructions

```
pip install jive
```

# Minimal code example

TODO: this won't exactly work yet -- make it work
TODO: display plots

```
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

For some more example code demonstrting some of the basic functionality see [this notebook](doc/jive_demo.ipynb). TODO: make this link render.

TODO: add more example code including worked through data analyses.