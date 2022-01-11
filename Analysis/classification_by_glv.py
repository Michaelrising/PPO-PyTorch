import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, roc_curve, auc
from LoadData import LoadData
import scipy.stats as st


patientlist = [1, 2, 3, 4, 6, 11, 12, 13, 15, 16, 17, 19, 20, 24, 25, 29, 30, 31, 32, 36, 37, 40, 42, 44, 46, 50, 51,
               52, 54, 56, 58, 61, 62, 63, 66, 71, 75, 77, 78, 79, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97,
               99, 100, 101, 102, 104, 105, 106, 108]
cs = sns.color_palette("Paired")
parsdir = "../GLV/analysis-sigmoid/model_pars/"

A_list = []
K_list = []
states_list = []
ALL_F_PARS_LIST = {}
ALL_B_PARS_LIST = {}
# reading the ode parameters and the initial/terminal states
markPatient = [6, 11]
markIndex = []
finalDay_list = []
for i in patientlist:
    if len(str(i)) == 1:
        patientNo = "patient00" + str(i)
    elif len(str(i)) == 2:
        patientNo = "patient0" + str(i)
    else:
        patientNo = "patient" + str(i)
    pars_list = os.listdir(parsdir + patientNo)
    patientData = np.array(pd.read_csv('../Data/dataTanaka/Bruchovsky_et_al/' + patientNo + '.txt'))
    finalDay_list.append(patientData[-1,-1] - patientData[0,-1])
    pars_list.sort()
    f_pars_list = []
    b_pars_list = []
    for pars in pars_list:
        pars_df = pd.read_csv(parsdir + patientNo + '/' + pars)
        A, K, states, final_pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
        f_pars_list.append(final_pars)
        b_pars_list.append(best_pars)
    ALL_F_PARS_LIST[patientNo] = np.stack(f_pars_list)
    ALL_B_PARS_LIST[patientNo] = np.stack(b_pars_list)
############## Clinician Classification ###############

patient_no = [1, 2, 3, 4, 6, 13, 14, 15, 16, 17,
              20, 22, 24, 26, 28, 29, 30, 31, 37, 39,
              40, 42, 44, 50, 51, 55, 56, 58, 60, 61,
              62, 63, 66, 71, 75, 77, 78, 79, 81, 84,
              86, 87, 91, 93, 94, 95, 96, 97, 100, 102,
              104, 105, 106, 108, 109, 32, 46, 64, 83, 92]
patient_no = list(set(patientlist) & set(patient_no))
patient_yes = [11, 12, 19, 25, 36, 41, 52, 54, 85, 88, 99, 101]
patient_yes = list(set(patient_yes) & set(patientlist))

#### Randomization #####
FinalDay = np.array(finalDay_list)
ResIndexlabels = pd.DataFrame(np.concatenate((np.ones(len(patient_no)), -np.ones(len(patient_yes)))),
                              index=patient_no + patient_yes)
ResIndexlabels = ResIndexlabels.sort_index()
fpr_list = []
tpr_list = []
thres_list = []
auc_list = []

plt.style.use('seaborn')
plt.style.use(['science', "nature"])
plt.figure(figsize=(8, 5))
for _ in range(100):
    rand_res_list = []
    for i in patientlist:
        if len(str(i)) == 1:
            patientNo = "patient00" + str(i)
        elif len(str(i)) == 2:
            patientNo = "patient0" + str(i)
        else:
            patientNo = "patient" + str(i)
        pars_arr = ALL_F_PARS_LIST[patientNo]
        rand = np.random.randint(1, pars_arr.shape[0], 1)
        res = pars_arr[rand.item(), -3]
        rand_res_list.append(res)
    rand_res_mat = np.stack(rand_res_list)
    resIndex = rand_res_mat.reshape(-1)
    CompetitionIndex = 1 / (1 + np.exp(- resIndex * FinalDay / 28 / 12))
    fpr, tpr, _ = roc_curve(ResIndexlabels, CompetitionIndex)
    plt.plot(
        fpr,
        tpr,
        color=cs[0],
        lw=1,
        alpha=0.6
    )
    fpr, tpr, thresholds = roc_curve(ResIndexlabels, CompetitionIndex, drop_intermediate=False)
    auc_score = auc(fpr, tpr)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    thres_list.append(thresholds)
    auc_list.append(auc_score)

# predicted expect and calculate confidence interval
mean_tpr = np.mean(np.stack(tpr_list), 0)
low_CI_bound, high_CI_bound = st.t.interval(0.9, mean_tpr.shape[0] - 1,
                                            loc=mean_tpr,
                                            scale=st.sem(np.stack(tpr_list)))
lw = 1
x=np.mean(np.stack(fpr_list), 0)
y=np.mean(np.stack(tpr_list), 0)
plt.plot(x,y , lw=2, c=cs[1])
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5,
                label='confidence interval')
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=14)
#plt.savefig('../Experts_Analysis/ROC_competition_index.png', dpi=300)
plt.show()