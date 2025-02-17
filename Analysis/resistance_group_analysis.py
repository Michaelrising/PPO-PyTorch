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
doseFileLise = os.listdir("../PPO_policy/resistance_group")
for file in doseFileLise.copy():
    if "survival" in file  or "patient036" in file or "patient078" in file:
        doseFileLise.remove(file)
patientLables = []
patientCPA = []
patientLEU = []
patientSurvivalTime = []
for file in doseFileLise:
    doseSeq = pd.read_csv("../PPO_policy/resistance_group/" + file, names = ["Month", "CPA", "LEU"], header=0)
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
fig, ax = plt.subplots(figsize = (20, 10))
simu_stop = [1380, None, 1486, 1238, None, None, None, None, None, 1298]

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
                s=160, label = 'S-End')

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
    if simu_stop[l]:
        plt.scatter(x = simu_stop[l],  y = barcontainer.patches[0].get_y() + barcontainer.patches[0].get_height()/2,
                marker=4, color = 'black', s = 200, label ='S-End')
    else:
        CPA = [0, 50, 100, 150, 200]
        LEU = [0, 7.5]
        extraDose = pd.read_csv("../Experts_policy/extrapolated/" +patient+"_extrapolated_doseSeq.csv")
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
plt.yticks(np.arange(0.5, 19, 2), labels, fontsize = 22)
# plt.ylabel("1 $\longleftarrow$ Patient No. $\longrightarrow$ 108", fontsize = 24)
plt.xticks(fontsize = 22)
plt.xlabel("Time (Day)", fontsize = 24)
plt.xlim(-10, 3900)
plt.legend([ a1, a2, a3, a4, s1, s2 ], ['C$\&$L-On',"Cpa-On ","Leu-On" ,'Treat-Off', 'S-End', 'C-End'],
           handler_map={a1: AnyObjectHandler(color=onColor), a2:AnyObjectHandler(color=onCpa, _hatch=None),
                        a3: AnyObjectHandler(color=onLeu, alpha = 0), a4: AnyObjectHandler(color=offColor,alpha=1, _hatch=None)}
           , fontsize =18)
if not os.path.exists("../Analysis/"):
    os.mkdir("../Analysis/")
plt.savefig("../Analysis/Resistance_group_Strategy.png", dpi = 500)
plt.show()
plt.close()


resistance_group = os.listdir('../PPO_preTrained/resistance_group')
statesFileList = os.listdir('../PPO_states/resistance_group')
for file in statesFileList.copy():
    if "survival" in file  or "patient036" in file or "patient078" in file:
        statesFileList.remove(file)
patientLables = []
PSAThresholds = []
patientSurvivalTime = []
for file in statesFileList:
    statesSeq = pd.read_csv("../PPO_states/resistance_group/" + file, names = ["AD", "AI", "PSA"], header=0)
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
cpa_clinical_daily = []
leu_clinical_monthly = []
for patient_i in df_ppo_drug.index:
    clinical_data = pd.read_csv("../Data/dataTanaka/Bruchovsky_et_al/" + patient_i + ".txt", header=None)
    onoff = np.array(clinical_data.loc[:, 7])
    Days = np.array(clinical_data.loc[:, 9].diff()[1:])
    Days = np.append(Days, 28)
    offdays = sum(Days[~onoff.astype(bool)])
    off_percentage = offdays/sum(Days)
    off_clinical.append(off_percentage)
    patient_cpa = np.array(clinical_data.loc[:, 2])
    clinical_cpa_daily = np.sum(patient_cpa[~np.isnan(patient_cpa)] * Days[~np.isnan(patient_cpa)])/(clinical_data.loc[clinical_data.shape[0]-1, 9] - clinical_data.loc[0, 9])
    cpa_clinical_daily.append(clinical_cpa_daily)
    patient_leu = np.array(clinical_data.loc[:, 3])
    clinical_leu_monthly = patient_leu[~np.isnan(patient_leu)].sum()/(clinical_data.loc[clinical_data.shape[0]-1, 9] - clinical_data.loc[0, 9]) * 28
    leu_clinical_monthly.append(clinical_leu_monthly)

from scipy.stats import ttest_rel
print(ttest_rel(ppo_off, off_clinical))
print(np.array(ppo_off)-np.array(off_clinical))

## daily drug administration ##
ppo_cpa_daily = []
ppo_leu_monthly = []
for patient_i in df_ppo_drug.index:
    patient_cpa = np.array(df_ppo_CPA.loc[patient_i, ~np.isnan(df_ppo_CPA.loc[patient_i])])
    cpa_daily = sum(patient_cpa)/(patient_cpa.shape[0]) * 200
    patient_leu = np.array(df_ppo_LEU.loc[patient_i, ~np.isnan(df_ppo_LEU.loc[patient_i])])
    leu_monthly = sum(patient_leu)/(patient_leu.shape[0]) * 7.5
    ppo_cpa_daily.append(cpa_daily)
    ppo_leu_monthly.append(leu_monthly)

print(ttest_rel(ppo_cpa_daily, cpa_clinical_daily))
print(ttest_rel(ppo_leu_monthly, leu_clinical_monthly))