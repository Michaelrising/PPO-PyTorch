import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _utils import *
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

clustering = pd.read_csv('./kmeans_clustering_results.csv', header=0, names=['y', 'x_g', 'x_r', 'cluster'])
group1, group2, group3 = [np.array(clustering.loc[clustering['cluster'] == i].index) for i in [0,1,2]]
group2_1 = np.array(clustering.loc[clustering['cluster'] == 1].loc[clustering['x_r'] > 0].index)
group2_2 = np.array(clustering.loc[clustering['cluster'] == 1].loc[clustering['x_r'] <= 0].index)

comparision_list = ['85', '16', '01', '61']

states_file_list = ['../PPO_states/resistance_group/patient0' + str(i)+ "_converge_high_reward_states.csv" for i in [85, 16]] \
                   + ['../PPO_states/response_group/patient0' + i + "_converge_high_reward_states.csv" for i in ['01', '61']]
c_HI_list = []
for i, file in enumerate(states_file_list):
    patient = comparision_list[i]
    list_df = pd.read_csv('../Data/model_pars/Args_patient0' + patient + '.csv')
    K = np.array(list_df.loc[1, ~np.isnan(list_df.loc[1, :])])
    states = pd.read_csv(file, header=0, names=['HD', 'HI', 'PSA'])
    c_HI_list.append(np.array(states['HI'])/K[1])

cs = sns.color_palette('Paired')
plt.style.use('seaborn')
plt.style.use(['science', 'nature'])
colors = [cs[1], cs[3], cs[3], cs[5]]
lines = ['-','-', '--', '-']
fig, ax = plt.subplots(figsize=[10, 8])
for i in range(4):
    x = np.arange(c_HI_list[i].shape[0]) * 28
    ax.plot(x, c_HI_list[i], label='patient0' + comparision_list[i], c=colors[i], ls=lines[i])
    ax.set_xlim(-100, 4000)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlabel('Time (Days)', fontsize = 22)
plt.ylabel("Cell concentration", fontsize = 22)
plt.legend(fontsize = 20)

plt.style.use('default')
plt.style.use(['science', 'nature'])
axins = ax.inset_axes([1900, 0.05, 2000, 0.3], transform=ax.transData)
for i in range(2, 4):
    x = np.arange(c_HI_list[i].shape[0]) * 28
    axins.plot(x, c_HI_list[i], label='patient0' + comparision_list[i], c=colors[i], ls=lines[i])
    axins.text(x = -100, y = 5.5 * 1e-4, s='$10^{-4}$', fontsize = 9)
    axins.set_xticklabels([])
    axins.set_yticks([0, 1 * 1e-4, 2* 1e-4, 3* 1e-4, 4* 1e-4, 5* 1e-4], labels=[0, 1, 2, 3,4, 5], fontsize = 12)
plt.savefig('./4_Chose_clustered_patients_HI_analysis.png', dpi=300)
plt.show()
