from jive.AJIVE import AJIVE
from jive.PCA import PCA
from jive.ajive_fig2 import generate_data_ajive_fig2
from jive.viz.block_visualization import data_block_heatmaps, jive_full_estimate_heatmaps
import matplotlib.pyplot as plt

X, Y = generate_data_ajive_fig2()
plt.figure(figsize=[11,  5])
data_block_heatmaps([X, Y])
plt.savefig('figures/data_heatmaps.png', bbox_inches='tight')
plt.close()

# determine initial signal ranks by inspecting scree plots
plt.figure(figsize=[10, 5])
plt.subplot(1, 2, 1)
PCA().fit(X).plot_scree()
plt.subplot(1, 2, 2)
PCA().fit(Y).plot_scree()
plt.savefig('figures/scree_plots.png', bbox_inches='tight')
plt.close()

ajive = AJIVE(init_signal_ranks={'x': 2, 'y': 3})
ajive.fit(blocks={'x': X, 'y': Y})

plt.figure(figsize=[10, 20])
jive_full_estimate_heatmaps(ajive.get_full_block_estimates(),
                            blocks={'x': X, 'y': Y})
plt.savefig('figures/jive_estimate_heatmaps.png', bbox_inches='tight')
plt.close()

ajive.plot_joint_diagnostic()
plt.savefig('figures/jive_diagnostic.png', bbox_inches='tight')
plt.close()
