import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import markers
import seaborn as sns
from sklearn.metrics import r2_score, roc_curve, auc
from LoadData import LoadData
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import random

cs = sns.color_palette("Paired")
parsdir = "../Data/model_pars"
parslist = os.listdir(parsdir)
A_list = []
K_list = []
states_list = []
f_pars_list = []
b_pars_list = []
# reading the ode parameters and the initial/terminal states
markPatient = [6, 11]
markIndex = []
patientList = []
finalDay_list = []
for args in parslist:
    pars_df = pd.read_csv(parsdir + '/' + args)
    A, K, states, final_pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
    A_list.append(A)
    K_list.append(K)
    states_list.append(states)
    f_pars_list.append(final_pars)
    b_pars_list.append(best_pars)
    patientNo = int(args[12:-4])
    if len(str(patientNo)) == 1:
        input = "patient00" + str(patientNo)
    elif len(str(patientNo)) == 2:
        input = "patient0" + str(patientNo)
    else:
        input = "patient" + str(patientNo)
    patientData = LoadData()._Patient_data(input)
    if patientNo == 2:
        patientData = patientData[:84]
    if patientNo == 46:
        patientData[43:46, 1] -= 10
    if patientNo == 56:
        patientData[46, 1] = (patientData[44, 1] + patientData[48, 1]) / 2
    if patientNo == 86:
        patientData[1, 1] = (patientData[1, 1] + patientData[8, 1]) / 2
    if patientNo == 104:
        patientData = patientData[:(-3)]
    finalDay = patientData[-1, 6] - patientData[0, 6]
    finalDay_list.append(finalDay)
    patientList.append(patientNo)
    if patientNo in markPatient:
        markIndex.append(best_pars[-3])

KMat = np.stack(K_list)
FinalParsMat = np.stack(f_pars_list)
BestParsMat = np.stack(b_pars_list)
FinalResIndex = FinalParsMat[:, -3]
BestResIndex = BestParsMat[:, -3]
markIndex = np.array(markIndex, dtype=np.float)
FinalDay = pd.Series(finalDay_list, index=patientList).sort_index()

Ai_growth_rate = FinalParsMat[:, 1]
Ai_growth_rate = pd.Series(Ai_growth_rate, index=patientList).sort_index()

dfFinalparsMat = pd.DataFrame(f_pars_list, index=patientList)
dfFinalResIndex = dfFinalparsMat.iloc[:, -3]
dfFinalResIndex = dfFinalResIndex.sort_index()
CompetitionIndex = 1 / (1 + np.exp(- dfFinalResIndex * FinalDay / 28 / 12))

patient_no = [1, 2, 3, 4, 6, 13, 14, 15, 16, 17,
              20, 22, 24, 26, 28, 29, 30, 31, 37, 39,
              40, 42, 44, 50, 51, 55, 56, 58, 60, 61,
              62, 63, 66, 71, 75, 77, 78, 79, 81, 84,
              86, 87, 91, 93, 94, 95, 96, 97, 100, 102,
              104, 105, 106, 108, 109, 32, 46, 64, 83, 92]
patient_no = list(set(patientList) & set(patient_no))
patient_yes = [11, 12, 19, 25, 36, 41, 52, 54, 85, 88, 99, 101]
patient_yes = list(set(patientList) & set(patient_yes))

ResIndexlabels = pd.DataFrame(np.concatenate((np.ones(len(patient_no)), -np.ones(len(patient_yes)))),
                              index=patient_no + patient_yes)
ResIndexlabels = ResIndexlabels.sort_index()
fpr, tpr, thresholds = roc_curve(ResIndexlabels, CompetitionIndex)
auc_score = auc(fpr, tpr)
plt.style.use('seaborn')
plt.style.use(['science', "nature"])
plt.figure(figsize=(8, 5))
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="AUC = %0.2f" % auc_score,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=14)
plt.savefig('../Experts_Analysis/ROC_competition_index.png', dpi=300)
plt.show()

### Yes
train_yes = random.sample(patient_yes, int(len(patient_yes) * 0.7))
test_yes = list(set(patient_yes) - set(train_yes))
### No
train_no = random.sample(patient_no, int(len(patient_no) * 0.7))
test_no = list(set(patient_no) - set(train_no))

train = train_yes + train_no
train.sort()
test = test_yes + test_no
test.sort()

#### K means clustering ####
data_for_kmeans = pd.DataFrame({'y': list(ResIndexlabels.loc[:, 0]),
                                'x_r': list(Ai_growth_rate),
                                'x_g': list(-dfFinalResIndex)}, index=ResIndexlabels.index)
normalized_x, norm = normalize(data_for_kmeans[['x_r', 'x_g']], axis = 0,return_norm=True, norm="l2")
normalized_data_for_kmeans = pd.DataFrame({'y': list(ResIndexlabels.loc[:, 0]),
                                           'x_r': normalized_x[:, 0] * np.sqrt(1),
                                           'x_g': normalized_x[:, 1]}, index=ResIndexlabels.index)
train_data = normalized_data_for_kmeans.loc[train]
test_data = normalized_data_for_kmeans.loc[test]

#### K means 3 clustering ####
kmeans = KMeans(n_clusters=3).fit(normalized_data_for_kmeans[['x_r', 'x_g']])
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids]
cen_y = [i[1] for i in centroids]
normalized_data_for_kmeans['cluster'] = kmeans.predict(normalized_data_for_kmeans[['x_r', 'x_g']])
normalized_data_for_kmeans['cen_x'] = normalized_data_for_kmeans.cluster.map({0: cen_x[0], 1: cen_x[1], 2: cen_x[2]})
normalized_data_for_kmeans['cen_y'] = normalized_data_for_kmeans.cluster.map({0: cen_y[0], 1: cen_y[1], 2: cen_y[2]})
colors = [cs[1], cs[3], cs[5]]
normalized_data_for_kmeans['c'] = normalized_data_for_kmeans.cluster.map({0: colors[0], 1: colors[1], 2: colors[2]})

original_cents = centroids * norm * np.array([1/np.sqrt(1), 1])
from scipy import interpolate
from scipy.spatial import ConvexHull

chosen_patients_list = [85, 16, 1, 61]
unconsen_patients_list = list(set(patientList) - set(chosen_patients_list))
unconsen_patients_list.sort()
fig = plt.figure(figsize=(10, 8))
plt.scatter(data_for_kmeans.loc[unconsen_patients_list].x_r, data_for_kmeans.loc[unconsen_patients_list].x_g,
            c = normalized_data_for_kmeans.loc[unconsen_patients_list].c, alpha=0.6, s=80)
plt.scatter(data_for_kmeans.loc[chosen_patients_list].x_r, data_for_kmeans.loc[chosen_patients_list].x_g,
            c = normalized_data_for_kmeans.loc[chosen_patients_list].c,marker='*', alpha=1, s=200)
plt.scatter(original_cents[:, 0], original_cents[:, 1], c = colors, marker='^', s = 160)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
loc = [[0.029, 0.52], [0.001, 0.4], [0.002, -0.3], [0.042, -0.13]]
for i in range(4):
    p = chosen_patients_list[i]
    l = loc[i]
    tx = data_for_kmeans.loc[p].x_r
    ty = data_for_kmeans.loc[p].x_g
    plt.text(l[0], l[1], s="({}, {})".format(np.round(tx, 3), np.round(ty, 2)), fontsize=16)
for i in np.sort(normalized_data_for_kmeans.cluster.unique()):
    # get the convex hull
    points = data_for_kmeans[normalized_data_for_kmeans.cluster == i][['x_r', 'x_g']].values
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices, 0],
                       points[hull.vertices, 0][0])
    y_hull = np.append(points[hull.vertices, 1],
                       points[hull.vertices, 1][0])

    # plot shape
    if i == 0:
        print(i)
        f = lambda x: 0*x
        x = np.array([0, 0.027])
        plt.plot(x, f(x), color=colors[i], lw=2,ls='--')
    # if i == 2:
    #     f=lambda x: (original_cents[2,1] + 1)/(original_cents[2,0] - 0.043) * (x-0.043) -1
    #     x = np.array([0.018, 0.043])
    #     plt.plot(x, f(x), c=colors[i], lw=2, ls='--')
        # fig.add_artist(Line2D([original_cents[0,0], 0.043], [original_cents[0,1], -1], color=colors[i], lw=2))
        # fig.add_artist(Line2D([original_cents[0,0], 0.019],[original_cents[0,1], (original_cents[0,1] + 1)/(original_cents[0,0] - 0.043) * (0.019-0.043) -1]
        #        , color=colors[i], lw=2))
    plt.fill(x_hull, y_hull, '--', c=colors[i], alpha=0.2)

plt.xlabel('Growth rate of HI ($r$)', fontsize = 22)
plt.ylabel('The resistance index ($\gamma$)', fontsize = 22)
legend_elements = [Line2D([0], [0], marker = 'o', color='w', alpha=0.6, label="Cluster-{}".format(i+1),
                          markerfacecolor=mcolor, markersize=8) for i, mcolor in enumerate(colors)]
legend_elements.extend([Line2D([0], [0], marker = '^', color='w',alpha=1, label="Centroid-{}".format(i+1),
                          markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(colors)])
plt.legend(handles = legend_elements, loc='upper right', ncol=2, fontsize = 18)
plt.savefig("./K-means_3_clustering.png")
plt.show()
group1, group2, group3 = normalized_data_for_kmeans.loc[normalized_data_for_kmeans['cluster']==0].index, \
                         normalized_data_for_kmeans.loc[normalized_data_for_kmeans['cluster']==1].index,\
                         normalized_data_for_kmeans.loc[normalized_data_for_kmeans['cluster']==2].index
data_for_kmeans['cluster'] = kmeans.predict(normalized_data_for_kmeans[['x_r', 'x_g']])
data_for_kmeans.to_csv("./kmeans_clustering_results.csv")

# Draw plots
sns.set(style="darkgrid")
plt.figure(figsize=(8, 5))
ax = sns.boxplot(-FinalResIndex, color='skyblue', orient='v')
ax = sns.swarmplot(-FinalResIndex, color='grey')
plt.scatter(x=-markIndex[0], y=0, color='red', label='patient006')
plt.scatter(x=-markIndex[1], y=0, color='yellow', label='patient011')
# Add jitter with the swarmplot function
# ax = sns.swarmplot(markIndex[0], color = 'red', label = 'patient006')
# ax = sns.swarmplot(markIndex[1], color = 'yellow', label = 'patient011')
plt.xlabel("Resistance Index $\gamma$", fontsize=16)
plt.legend(loc='upper left', fontsize=16)
# adding transparency to colors
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
plt.savefig('../Experts_Analysis/ResIndex.png', dpi=300)
plt.close()

GrAD, GrAI = FinalParsMat[:, 0], FinalParsMat[:, 1]
plt.style.use('seaborn')
plt.figure(figsize=(8, 5))
plt.scatter(GrAD, GrAI, s=12)
plt.plot([0, 0.1], [0, 0.1], color="red", lw=1)
plt.xlabel("$r_{HD}$", fontsize=16)
plt.ylabel("$r_{HI}$", fontsize=16)
plt.savefig('../Experts_Analysis/growthrate.png', dpi=300)
plt.close()

valdir = "../Data/model_validate"
vallist = os.listdir(valdir)
true_list = []
predict_list = []
for file in vallist:
    val_df = pd.read_csv(valdir + '/' + file)
    true_psa, pre_psa = [np.array(val_df.loc[i, :]) for i in range(2)]
    true_psa = true_psa[1:].astype(np.float)
    pre_psa = pre_psa[1:].astype(np.float)
    true_list.append(true_psa)
    predict_list.append(pre_psa)
TruePsa = np.concatenate(true_list)
PredictPsa = np.concatenate(predict_list)
r2 = r2_score(TruePsa, PredictPsa)
plt.style.use('seaborn')
plt.figure(figsize=(8, 5))
plt.scatter(TruePsa, PredictPsa, s=15)
plt.plot([0, 40], [0, 40], color="red", lw=1)
plt.xlabel("Measured PSA ($\mu g/ml$)", fontsize=16)
plt.ylabel("Simulated PSA ($\mu g/ml$)", fontsize=16)
plt.text(30, 10, f"$R^2 = $ {r2:<5.2f}", fontsize="large")
plt.savefig('../Experts_Analysis/Validation_PSA.png', dpi=300)
plt.close()

plt.style.use('seaborn')
plt.style.use(['science', 'nature'])
plt.figure(figsize=(45, 15))
plt.subplot(1, 3, 1)
plt.scatter(TruePsa, PredictPsa, s=80, c=cs[1])
plt.plot([0, 40], [0, 40], lw=3, c=cs[5])
plt.xlabel("Measured PSA ($\mu g/L$)", fontsize=45)
plt.ylabel("Simulated PSA ($\mu g/L$)", fontsize=45)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.text(25, 5, f"$R^2 = $ {r2:<5.2f}", fontsize=45)
plt.text(-8, 43, "a", fontdict={'size': 65, 'color': 'black', "family": 'Times New Roman'}, weight='bold')

plt.subplot(1, 3, 2)

ax = sns.boxplot(-FinalResIndex, color=cs[0], orient='v')
plt.scatter(x=-markIndex[0], y=0.005, color=cs[5], s=100, label='patient006')
plt.scatter(x=-markIndex[1], y=0.01, color=cs[1], s=100, label='patient011')
ax = sns.swarmplot(-FinalResIndex, color='grey', size=10)
plt.text(-1.4, -0.52, "b", fontdict={'size': 65, 'color': 'black', "family": 'Times New Roman'}, weight='bold')
# Add jitter with the swarmplot function
# ax = sns.swarmplot(markIndex[0], color = 'red', label = 'patient006')
# ax = sns.swarmplot(markIndex[1], color = 'yellow', label = 'patient011')
plt.xlabel("Resistance Index $\gamma$", fontsize=45)
plt.xticks(fontsize=35)
plt.legend(loc=(0.45, 0.07), fontsize=38)
# adding transparency to colors
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))

plt.subplot(1, 3, 3)
plt.plot(
    fpr,
    tpr,
    color=cs[7],
    lw=3,
    label="AUC = %0.2f" % auc_score,
)
plt.plot([0, 1], [0, 1], color=cs[1], lw=3, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlabel("False Positive Rate", fontsize=45)
plt.ylabel("True Positive Rate", fontsize=45)
plt.legend(loc="lower right", fontsize=45)
plt.text(-0.115, 1.08, "c", fontdict={'size': 65, 'color': 'black', "family": 'Times New Roman'}, weight='bold')
plt.savefig("../Experts_Analysis/model_analysis_three_plots.png")
plt.show()


#### K means 4 clustering ####
kmeans = KMeans(n_clusters=4).fit(normalized_data_for_kmeans[['x_r', 'x_g']])
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids]
cen_y = [i[1] for i in centroids]
normalized_data_for_kmeans['cluster'] = kmeans.predict(normalized_data_for_kmeans[['x_r', 'x_g']])
normalized_data_for_kmeans['cen_x'] = normalized_data_for_kmeans.cluster.map({0: cen_x[0], 1: cen_x[1], 2: cen_x[2], 3: cen_x[3]})
normalized_data_for_kmeans['cen_y'] = normalized_data_for_kmeans.cluster.map({0: cen_y[0], 1: cen_y[1], 2: cen_y[2], 3: cen_y[3]})
colors = [cs[1], cs[3], cs[5], cs[7]]
normalized_data_for_kmeans['c'] = normalized_data_for_kmeans.cluster.map({0: colors[0], 1: colors[1], 2: colors[2], 3: colors[3]})

original_cents = centroids * norm * np.array([1/np.sqrt(1), 1])
from scipy import interpolate
from scipy.spatial import ConvexHull

plt.figure(figsize=(10, 8))
plt.scatter(data_for_kmeans.x_r, data_for_kmeans.x_g, c = normalized_data_for_kmeans.c, alpha=0.6, s=80)
plt.scatter(original_cents[:, 0], original_cents[:, 1], c = colors, marker='^', s = 160)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
for i in normalized_data_for_kmeans.cluster.unique():
    # get the convex hull
    points = data_for_kmeans[normalized_data_for_kmeans.cluster == i][['x_r', 'x_g']].values
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices, 0],
                       points[hull.vertices, 0][0])
    y_hull = np.append(points[hull.vertices, 1],
                       points[hull.vertices, 1][0])

    # plot shape
    plt.fill(x_hull, y_hull, '--', c=colors[i], alpha=0.2)

plt.xlabel('Growth rate of HI ($r$)', fontsize = 22)
plt.ylabel('The resistance index ($\gamma$)', fontsize = 22)
legend_elements = [Line2D([0], [0], marker = 'o', color='w', alpha=0.6, label="Cluster-{}".format(i+1),
                          markerfacecolor=mcolor, markersize=8) for i, mcolor in enumerate(colors)]
legend_elements.extend([Line2D([0], [0], marker = '^', color='w',alpha=1, label="Centroid-{}".format(i+1),
                          markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(colors)])
plt.legend(handles = legend_elements, loc='upper right', ncol=2, fontsize = 18)
plt.savefig("./K-means_4_clustering.png")
plt.show()

#### Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
normalized_data_for_hiera = pd.DataFrame({'y': normalized_data_for_kmeans['y'],
                                          'x_r': normalized_data_for_kmeans['x_r'] * np.sqrt(1.),
                                          'x_g': normalized_data_for_kmeans['x_g'] * np.sqrt(5)})
clustering = AgglomerativeClustering(n_clusters=4).fit_predict(normalized_data_for_hiera[['x_r', 'x_g']])

normalized_data_for_hiera['cluster'] = clustering
colors = [cs[1], cs[3], cs[5], cs[7]]
normalized_data_for_hiera['c'] = normalized_data_for_hiera.cluster.map({0: colors[0], 1: colors[1], 2: colors[2], 3:colors[3]})
plt.figure(figsize=(10, 8))
plt.scatter(data_for_kmeans.x_r, data_for_kmeans.x_g, c = normalized_data_for_hiera.c, alpha=0.6, s=80)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
for i in normalized_data_for_hiera.cluster.unique():
    # get the convex hull
    points = data_for_kmeans[normalized_data_for_hiera.cluster == i][['x_r', 'x_g']].values
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices, 0],
                       points[hull.vertices, 0][0])
    y_hull = np.append(points[hull.vertices, 1],
                       points[hull.vertices, 1][0])

    # plot shape
    plt.fill(x_hull, y_hull, '--', c=colors[i], alpha=0.2)

plt.xlabel('Growth rate of HI ($r$)', fontsize = 22)
plt.ylabel('The resistance index ($\gamma$)', fontsize = 22)
legend_elements = [Line2D([0], [0], marker = 'o', color='w', alpha=0.6, label="Cluster-{}".format(i+1),
                          markerfacecolor=mcolor, markersize=8) for i, mcolor in enumerate(colors)]
plt.legend(handles = legend_elements, loc='upper right', fontsize = 18)
plt.savefig("./Hierarchical_clustering_clustering.png")
plt.show()


####