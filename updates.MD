# Updates

## 10/19/17 (version 0.0.1)

- changes to Jive object API
	- Jive() no longer creates the scree plot on initialization
	- the results of SVD (i.e. U, D, V) are now called scores, sv, loadings

- added sparse functionality
	- Jive() now works for both sparse and dense matrices (user can input a mix)
	- Jive() will not do the full SVD for sparse matrices -- the user has to specify the rank of the initial SVD (via the init_svd_ranks argument which is a list). The initial SVD rank should be larger than the initial signal rank.
	- The block specific joint/initial space estimation present a problem for large, sparse matrices. If S is a sparse matrix then UU^TS will be a dense matrix the same size as S. To prevent creating and attempting to compute the SVD of very large sparse matrices we implemented some functions based on scipy.linalg.sparse.LinearOperator that will do lazy evaluation for UU^TS and (I - UU^T)S (see lazymatpy module).

## 1/23/18 (version 0.0.2)

- changes to Jive object API
	- block specific/common estimates now called via `jive.get_block_specific_estimates()` and `jive.get_common_joint_space_estimate()`
	- now only one wedin bound resampling procedure (e.g. removed basis resampling)
	- removed `jive.compute_wedin_bound`

- added random direction bound (e.g. `jive.sample_random_direction_bounds()`)

- jive block returns samples from random direction (before just returned point estimate).
	- samples can be accessed via `jive.wedin_sv_samples` and `jive.random_sv_samples`

- added diagnostic plot (e.g. `jive.plot_joint_diagnostic()`)

- added diagnostic demo

