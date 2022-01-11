import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _utils import *
import seaborn as sns

a1, a2, a3, a4, a5, a6 = AnyObject(), AnyObject(), AnyObject(), AnyObject(), AnyObject(), AnyObject()

##############################################################
##################### Resistance Group #######################
##############################################################
cs = sns.color_palette()
doseFileLise = os.listdir("../PPO_policy/response_group")
for file in doseFileLise.copy():
    if "survival" in file  or "patient036" in file or "patient078" in file:
        doseFileLise.remove(file)
patientLables = []
patientCPA = []
patientLEU = []
patientSurvivalTime = []
for file in doseFileLise:
    doseSeq = pd.read_csv("../PPO_policy/response_group/" + file, names = ["Month", "CPA", "LEU"], header=0)
    patient = file[:10]
    patientLables.append(patient)
    patientSurvivalTime.append(np.array(doseSeq).shape[0] * 28)
    doseSeq["CPA"] = doseSeq["CPA"]/200
    doseSeq["LEU"] = doseSeq["LEU"]/7.5
    patientCPA.append(np.array(doseSeq["CPA"]))
    patientLEU.append(np.array(doseSeq["LEU"]))

df_ppo_CPA = pd.DataFrame(patientCPA, index = patientLables)
df_ppo_CPA = df_ppo_CPA.sort_index()
df_ppo_LEU = pd.DataFrame(patientLEU, index = patientLables)
df_ppo_LEU = df_ppo_LEU.sort_index()
df_ppo_Time = pd.DataFrame(patientSurvivalTime, index= patientLables, columns=["rl"])
df_ppo_Time = df_ppo_Time.sort_index()
onColor = "#FF0000"
onCpa = "#FF0000"
onLeu = "#87CEEB"
offColor = "#696969"
plt.style.use("seaborn")
plt.style.use(["science", 'nature'])
fig, ax = plt.subplots(figsize = (20, 30))
simu_stop = pd.read_csv('../Experts_policy/response_group_s_end_list.csv')

for l, patient in enumerate(df_ppo_CPA.index):
    cpa_ppo = df_ppo_CPA.loc[patient, ~np.isnan(df_ppo_CPA.loc[patient])]
    leu_ppo = df_ppo_LEU.loc[patient, ~np.isnan(df_ppo_LEU.loc[patient])]
    for month, cpaData in enumerate(cpa_ppo):
        leuData = leu_ppo[month]
        if cpaData != 0:
            if leuData != 0:
                barcontainer = ax.barh(patient +'-p', 28, left=month * 28, color=onColor, alpha=cpaData, hatch="///",
                        height=0.8, tick_label=None)
            else:
                barcontainer = ax.barh(patient+'-p', 28, left=month * 28, color=onCpa,
                        alpha=cpaData, height=0.8, tick_label=None)
            # ax.barh(patientLables[patient], 28, left = month * 28, hatch = "/", label = "LEU-ON",alpha = 0, height = 0.5, tick_label = None)
        if cpaData == 0 and leuData != 0:
            barcontainer = ax.barh(patient+'-p', 28, left=month * 28, color=onLeu, hatch="///",
                    alpha=0, height=0.8, tick_label=None)
        if ~np.isnan(df_ppo_CPA.loc[patient, month]) and cpaData == 0 and leuData == 0:
            barcontainer = ax.barh(patient+'-p', 28, left=month * 28, color=offColor, height=0.8, tick_label=None)
    s1 = plt.scatter(x=barcontainer.patches[0].get_x() + barcontainer.patches[0].get_width(),
                y=barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height() / 2, marker=4, color='black',
                s=200, label = 'S-End')

    clinical_data = pd.read_csv("../Data/dataTanaka/Bruchovsky_et_al/" + patient + ".txt", header=None)
    ONOFF = np.array(clinical_data.loc[:, 7])
    drugOnDays = 0
    drugOffDays = 0
    Days = np.array(clinical_data.loc[:, 9]) - np.array(clinical_data.loc[0, 9])
    CPA = np.array(clinical_data.loc[:, 2])
    LEU = np.array(clinical_data.loc[:, 3])
    cpa_left = 0
    leu_left = 0
    for ii in range(len(ONOFF) - 1):
        cpa = CPA[ii]
        leu = LEU[ii]
        if ~np.isnan(cpa):
            barcontainer = ax.barh(patient + '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=onColor, height=0.8, alpha = cpa/200,
                                   tick_label=None)
        if ~np.isnan(leu):
            barcontainer = ax.barh(patient + '-c', max(28 * int(leu/7.5), Days[ii + 1] - Days[ii]), left=Days[ii], hatch="///",color =onLeu,alpha = 0,
                                   height=0.8, tick_label=None)
        if np.isnan(leu) and np.isnan(cpa):
            barcontainer = ax.barh(patient + '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=offColor, height=0.8,
                                   tick_label=None)
        # if ONOFF[ii] == 1:
        #     drugOnDays += Days[ii + 1] - Days[ii]
        #     barcontainer= ax.barh(patient + '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=onColor, height=0.8, tick_label=None,
        #             alpha=0.5)
        # else:
        #     drugOffDays += Days[ii + 1] - Days[ii]
        #     barcontainer =ax.barh(patient+ '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=offColor, height=0.8, tick_label=None,
        #             alpha=0.5)
    if ~np.isnan(simu_stop.loc[l].item()):
        plt.scatter(x = simu_stop.loc[l].item(),  y = barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height()/2,
                marker=4, color = 'black', s = 200, label ='S-End')
    else:
        CPA = [0, 50, 100, 150, 200]
        LEU = [0, 7.5]
        extraDose = pd.read_csv("../Experts_policy/extrapolated/response_group/" +patient+"_extrapolated_doseSeq.csv")
        left = Days[-1]
        extraDose = np.array(extraDose)[:, -1]
        for ii in range(extraDose.shape[0]):
            extra_cpa = CPA[int(extraDose[ii]%5)]
            extra_leu = LEU[int(extraDose[ii]//5)]
            if left > 28 * 120:
                length = 28 * 121 - left
            else:
                length = 28
            if extra_cpa:
                ax.barh(patient+"-c", length, left=left, color=onCpa,alpha =extra_cpa/200, height=0.8, tick_label=None)
            if extra_leu:
                ax.barh(patient + '-c', 28 , left=left, hatch="///", color=onLeu, alpha=0,
                        height=0.8, tick_label=None)
            if not extra_leu and not extra_cpa:
                ax.barh(patient+'-c', length, left=left, color=offColor, height=0.8, alpha =1, tick_label=None)
            left += 28
            if left > 28 * 121:
                break
        plt.scatter(x=left, y=barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height() / 2,
                    marker=4, color='black', s=200, label='S-End')
    s2 = plt.scatter(x=barcontainer.patches[0].get_x() + barcontainer.patches[0].get_width(),
                y=barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height() / 2,
                marker="X", color='black', s=200, label='C-End')

locs, labels = plt.yticks()
labels = df_ppo_CPA.index
plt.yticks(np.arange(0.5, 59, 2), labels, fontsize = 22)
# plt.ylabel("1 $\longleftarrow$ Patient No. $\longrightarrow$ 108", fontsize = 24)
plt.xticks(fontsize = 22)
plt.xlabel("Time (Day)", fontsize = 24)
plt.xlim(-10, 3900)
plt.legend([ a1, a2, a3, a4, s1, s2 ], ['C$\&$L-On',"Cpa-On ","Leu-On" ,'Treat-Off', 'S-End', 'C-End'],
           handler_map={a1: AnyObjectHandler(color=onColor), a2:AnyObjectHandler(color=onCpa, _hatch=None),
                        a3: AnyObjectHandler(color=onLeu, alpha = 0), a4: AnyObjectHandler(color=offColor,alpha=1, _hatch=None)}
           , fontsize =16)

if not os.path.exists("../Analysis/"):
    os.mkdir("../Analysis/")
plt.savefig("../Analysis/Response_group_Strategy.png", dpi = 500)
plt.show()
plt.close()


resistance_group = os.listdir('../PPO_preTrained/response_group')
statesFileList = os.listdir('../PPO_states/response_group')
for file in statesFileList.copy():
    if "survival" in file  or "patient036" in file or "patient078" in file:
        statesFileList.remove(file)
patientLables = []
PSAThresholds = []
patientSurvivalTime = []
for file in statesFileList:
    statesSeq = pd.read_csv("../PPO_states/response_group/" + file, names = ["AD", "AI", "PSA"], header=0)
    patient = file[:10]
    patientLables.append(patient)
    psa = statesSeq['PSA']
    diff1_psa = psa.diff()
    ratio_psa = diff1_psa/psa[:-1]
    PSAThresholds.append(ratio_psa)

ppo_off= [ ]

df_ppo_drug = df_ppo_CPA + df_ppo_LEU
for patient_i in df_ppo_drug.index:
    patient_drug = np.array(df_ppo_drug.loc[patient_i, ~np.isnan(df_ppo_drug.loc[patient_i])])
    off_percentage = patient_drug[patient_drug == 0].shape[0]/patient_drug.shape[0]
    ppo_off.append(off_percentage)

off_clinical = []
list_clinical_time = []
for l, patient_i in enumerate(df_ppo_drug.index):
    clinical_data = pd.read_csv("../Data/dataTanaka/Bruchovsky_et_al/" + patient_i + ".txt", header=None)
    onoff = np.array(clinical_data.loc[:, 7])
    # if np.isnan(simu_stop.loc[l].item()):
    #     extraDose = pd.read_csv(
    #         "../Experts_policy/extrapolated/response_group/" + patient_i + "_extrapolated_doseSeq.csv")
    #     list_clinical_time.append(clinical_data.loc[clinical_data.shape[0]-1, 9] - clinical_data.loc[0, 9] + extraDose.shape[0]*28)
    # else:
    list_clinical_time.append(clinical_data.loc[clinical_data.shape[0] - 1, 9] - clinical_data.loc[0, 9])
    Days = np.array(clinical_data.loc[:, 9].diff()[1:])
    Days = np.append(Days, 28)
    offdays = sum(Days[~onoff.astype(bool)])
    off_percentage = offdays/sum(Days)
    off_clinical.append(off_percentage)
df_clinical_time = pd.DataFrame(list_clinical_time, columns =['clinical'], index = df_ppo_drug.index)

from scipy.stats import ttest_rel
print(ttest_rel(ppo_off, off_clinical))
print(np.array(ppo_off)-np.array(off_clinical))
np.mean(np.array(ppo_off)-np.array(off_clinical))

### Time to Progression ####
diff_time = (df_ppo_Time['rl'] - df_clinical_time['clinical'])/df_clinical_time['clinical']
print(ttest_rel(df_ppo_Time['rl'], df_clinical_time['clinical']))