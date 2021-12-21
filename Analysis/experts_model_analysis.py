import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, roc_curve, auc
from LoadData import LoadData

cs = sns.color_palette("Paired")
parsdir = "../Data/model_pars"
parslist = os.listdir(parsdir)
A_list = []
K_list = []
states_list = []
f_pars_list =[]
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
markIndex = np.array(markIndex, dtype = np.float)
FinalDay = pd.Series(finalDay_list, index = patientList).sort_index()

dfFinalparsMat = pd.DataFrame(f_pars_list, index = patientList)
dfFinalResIndex = dfFinalparsMat.iloc[:,-3]
dfFinalResIndex = dfFinalResIndex.sort_index()
CompetitionIndex = 1/(1 + np.exp(- dfFinalResIndex* FinalDay /28/12))

patient_no = [1, 2, 3, 4, 6, 13, 14, 15, 16, 17,
	20, 22, 24, 26, 28, 29, 30, 31, 37, 39,
	40, 42, 44, 50, 51, 55, 56, 58, 60, 61,
	62, 63, 66, 71, 75, 77, 78, 79, 81, 84,
	86, 87, 91, 93, 94, 95, 96, 97, 100, 102,
	104, 105, 106, 108, 109, 32, 46, 64, 83, 92]
patient_yes = [11, 12, 19, 25, 36, 41, 52, 54, 85, 88, 99, 101]

ResIndexlabels = pd.DataFrame(np.concatenate((np.ones(len(patient_no)), -np.ones(len(patient_yes)))), index = patient_no + patient_yes)
ResIndexlabels = ResIndexlabels.loc[patientList].sort_index()
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
plt.legend(loc="lower right", fontsize = 14)
plt.savefig('../Experts_Analysis/ROC_competition_index.png', dpi = 300)
plt.show()

# Draw plots
sns.set(style="darkgrid")
plt.figure(figsize=(8, 5))
ax = sns.boxplot(-FinalResIndex,color='skyblue', orient='v')
ax = sns.swarmplot(-FinalResIndex, color='grey')
plt.scatter(x = -markIndex[0], y=0 , color = 'red', label = 'patient006')
plt.scatter(x = -markIndex[1], y=0 , color = 'yellow', label = 'patient011')
# Add jitter with the swarmplot function
# ax = sns.swarmplot(markIndex[0], color = 'red', label = 'patient006')
# ax = sns.swarmplot(markIndex[1], color = 'yellow', label = 'patient011')
plt.xlabel("Resistance Index $\gamma$",  fontsize=16)
plt.legend(loc='upper left',  fontsize=16)
# adding transparency to colors
for patch in ax.artists:
     r, g, b, a = patch.get_facecolor()
     patch.set_facecolor((r, g, b, .3))
plt.savefig('../Experts_Analysis/ResIndex.png', dpi = 300)
plt.close()


GrAD, GrAI = FinalParsMat[:, 0], FinalParsMat[:, 1]
plt.style.use('seaborn')
plt.figure(figsize=(8, 5))
plt.scatter(GrAD, GrAI, s = 12)
plt.plot([0, 0.1], [0, 0.1],color="red", lw = 1)
plt.xlabel("$r_{HD}$",  fontsize=16)
plt.ylabel("$r_{HI}$",  fontsize=16)
plt.savefig('../Experts_Analysis/growthrate.png', dpi = 300)
plt.close()

valdir = "../Data/model_validate"
vallist = os.listdir(valdir)
true_list = []
predict_list = []
for file in vallist:
    val_df = pd.read_csv(valdir + '/' + file)
    true_psa, pre_psa= [np.array(val_df.loc[i, :]) for i in range(2)]
    true_psa = true_psa[1:].astype(np.float)
    pre_psa = pre_psa[1:].astype(np.float)
    true_list.append(true_psa)
    predict_list.append(pre_psa)
TruePsa = np.concatenate(true_list)
PredictPsa = np.concatenate(predict_list)
r2 = r2_score(TruePsa, PredictPsa)
plt.style.use('seaborn')
plt.figure(figsize=(8, 5))
plt.scatter(TruePsa, PredictPsa, s = 15)
plt.plot([0, 40], [0, 40],color="red", lw = 1)
plt.xlabel("Measured PSA ($\mu g/ml$)",  fontsize=16)
plt.ylabel("Simulated PSA ($\mu g/ml$)",  fontsize=16)
plt.text(30,10, f"$R^2 = $ {r2:<5.2f}", fontsize = "large")
plt.savefig('../Experts_Analysis/Validation_PSA.png', dpi = 300)
plt.close()


# plt.figure(figsize=(24, 8))
# plt.subplot(3, 1, 1)
# plt.scatter(TruePsa, PredictPsa, s = 15)
# plt.plot([0, 40], [0, 40],color="red", lw = 1)
# plt.xlabel("Measured PSA ($\mu g/ml$)",  fontsize=14)
# plt.ylabel("Simulated PSA ($\mu g/ml$)",  fontsize=14)
# plt.text(30,10, f"$R^2 = $ {r2:<5.2f}", fontsize = "large")
#
# plt.subplot(3,1,2)
# ax = sns.boxplot(FinalResIndex,color='skyblue', orient='v')
# ax = sns.swarmplot(FinalResIndex, color='grey')
# plt.scatter(x = markIndex[0], y=0 , color = 'red', label = 'patient006')
# plt.scatter(x = markIndex[1], y=0 , color = 'yellow', label = 'patient011')
# # Add jitter with the swarmplot function
# # ax = sns.swarmplot(markIndex[0], color = 'red', label = 'patient006')
# # ax = sns.swarmplot(markIndex[1], color = 'yellow', label = 'patient011')
# plt.xlabel("Resistance Index $\gamma$",  fontsize=14)
# plt.legend(loc='upper left',  fontsize=14)
# # adding transparency to colors
# for patch in ax.artists:
#      r, g, b, a = patch.get_facecolor()
#      patch.set_facecolor((r, g, b, .3))
#
# plt.subplot(3,1,3)
# plt.plot(
#     fpr,
#     tpr,
#     color="darkorange",
#     lw=lw,
#     label="AUC = %0.2f" % auc_score,
# )
# plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate",fontsize=14)
# plt.ylabel("True Positive Rate",fontsize=14)
# plt.legend(loc="lower right", fontsize = 14)
#
# plt.savefig("../Experts_Analysis/model_analysis_three_plots.png")
# plt.show()

plt.style.use('seaborn')
plt.style.use(['science','nature'])
plt.figure(figsize=(45, 15))
plt.subplot(1, 3, 1)
plt.scatter(TruePsa, PredictPsa, s = 80, c = cs[1])
plt.plot([0, 40], [0, 40], lw = 3, c = cs[5])
plt.xlabel("Measured PSA ($\mu g/L$)",  fontsize=45)
plt.ylabel("Simulated PSA ($\mu g/L$)",  fontsize=45)
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.text(25,5, f"$R^2 = $ {r2:<5.2f}", fontsize = 45)
plt.text(-8, 43, "a", fontdict={'size': 65, 'color': 'black', "family": 'Times New Roman'},weight='bold')

plt.subplot(1,3,2)

ax = sns.boxplot(-FinalResIndex,color=cs[0], orient='v')
plt.scatter(x = -markIndex[0], y=0.005 , color = cs[5],s = 100, label = 'patient006')
plt.scatter(x = -markIndex[1], y=0.01 , color = cs[1], s = 100, label = 'patient011')
ax = sns.swarmplot(-FinalResIndex, color='grey', size = 10)
plt.text(-1.4, -0.52, "b", fontdict={'size': 65, 'color': 'black', "family": 'Times New Roman'},weight='bold')
# Add jitter with the swarmplot function
# ax = sns.swarmplot(markIndex[0], color = 'red', label = 'patient006')
# ax = sns.swarmplot(markIndex[1], color = 'yellow', label = 'patient011')
plt.xlabel("Resistance Index $\gamma$",  fontsize=45)
plt.xticks(fontsize = 35)
plt.legend(loc=(0.45,0.07),  fontsize=38)
# adding transparency to colors
for patch in ax.artists:
     r, g, b, a = patch.get_facecolor()
     patch.set_facecolor((r, g, b, .3))

plt.subplot(1,3,3)
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
plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)
plt.xlabel("False Positive Rate",fontsize=45)
plt.ylabel("True Positive Rate",fontsize=45)
plt.legend(loc="lower right", fontsize = 45)
plt.text(-0.115, 1.08, "c", fontdict={'size': 65, 'color': 'black', "family": 'Times New Roman'},weight='bold')
plt.savefig("../Experts_Analysis/model_analysis_three_plots.png")
plt.show()
