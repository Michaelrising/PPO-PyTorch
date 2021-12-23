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
        # if ONOFF[ii] == 1:
        #     drugOnDays += Days[ii + 1] - Days[ii]
        #     barcontainer= ax.barh(patient + '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=onColor, height=0.8, tick_label=None,
        #             alpha=0.5)
        # else:
        #     drugOffDays += Days[ii + 1] - Days[ii]
        #     barcontainer =ax.barh(patient+ '-c', Days[ii + 1] - Days[ii], left=Days[ii], color=offColor, height=0.8, tick_label=None,
        #             alpha=0.5)
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

#############################################################################
################################# All patients ##############################
#############################################################################
cs = sns.color_palette()
doseFileLise = os.listdir("../PPO_policy/converge")
for file in doseFileLise.copy():
    if "survival" in file  or "patient036" in file or "patient078" in file:
        doseFileLise.remove(file)
patientLables = []
patientCPA = []
patientLEU = []
patientSurvivalTime = []
for file in doseFileLise:
    doseSeq = pd.read_csv("../PPO_policy/converge/" + file, names = ["Month", "CPA", "LEU"], header=0)
    patient = file[:10]
    patientLables.append(patient)
    patientSurvivalTime.append(np.array(doseSeq).shape[0] * 28)
    doseSeq["CPA"] = doseSeq["CPA"]/200
    doseSeq["LEU"] = doseSeq["LEU"]/7.5
    patientCPA.append(np.array(doseSeq["CPA"]))
    patientLEU.append(np.array(doseSeq["LEU"]))

dfpatientCPA = pd.DataFrame(patientCPA, index = patientLables)
dfpatientCPA = dfpatientCPA.sort_index()
dfpatientLEU = pd.DataFrame(patientLEU, index = patientLables)
dfpatientLEU = dfpatientLEU.sort_index()
dfpatientTime = pd.DataFrame(patientSurvivalTime, index= patientLables, columns=["rl"])
dfpatientTime = dfpatientTime.sort_index()

plt.style.use("seaborn")
plt.style.use(["science", 'nature'])
fig, ax = plt.subplots(figsize = (20, 10))

for month in range(int(max(patientSurvivalTime)/28)):
    # subPatientSurvivalTime = np.ones(len(doseFileLise)) * 28
    for patient in dfpatientCPA.index: # range(len(doseFileLise)):
        cpaData = dfpatientCPA.loc[patient, month] if ~np.isnan(dfpatientCPA.loc[patient, month]) else 0
        leuData = dfpatientLEU.loc[patient, month] if ~np.isnan(dfpatientLEU.loc[patient, month]) else 0
        if cpaData != 0:
            if leuData != 0:
                ax.barh(patient, 28, left = month * 28 , color = onColor, label = "Cpa&Leu-On", alpha = cpaData, hatch = "///", height = 0.8, tick_label = None)
            else:
                ax.barh(patient, 28, left=month * 28, color=onCpa, label="Cpa-On",
                        alpha=cpaData, height=0.8, tick_label=None)
            #ax.barh(patientLables[patient], 28, left = month * 28, hatch = "/", label = "LEU-ON",alpha = 0, height = 0.5, tick_label = None)
        if cpaData == 0 and leuData != 0:
            ax.barh(patient, 28, left=month * 28, color=onLeu, label="Leu-On",hatch = "///",
                    alpha=0, height=0.8, tick_label=None)
        if ~np.isnan(dfpatientCPA.loc[patient, month]) and cpaData==0 and leuData == 0:
            ax.barh(patient, 28, left = month * 28, color=offColor, label='Treat-Off', height=0.8, tick_label = None)

locs, labels = plt.yticks()
plt.yticks(locs, labels, rotation = 20)
plt.ylabel("1 $\longleftarrow$ Patient No. $\longrightarrow$ 108", fontsize = 24)
plt.xticks(fontsize = 22)
plt.xlabel("Time (Day)", fontsize = 24)
plt.xlim(-10, 3900)
plt.legend([ a1, a2, a3, a4], ['C$\&$L-On',"Cpa-On ","Leu-On" ,'Treat-Off'],
           handler_map={a1: AnyObjectHandler(), a2:AnyObjectHandler1(), a3:AnyObjectHandler2(), a4: AnyObjectHandler3()}
           , fontsize =18)
if not os.path.exists("../PPO_Analysis/"):
    os.mkdir("../PPO_Analysis/")
plt.savefig("../PPO_Analysis/RL_Dose_Strategy.png", dpi = 500)
plt.show()
plt.close()


# plt.legend([AnyObject()], ['CPA-LEU treatment On'],
#            handler_map={AnyObject: AnyObjectHandler1()})
# plt.legend([AnyObject()], ['LEU treatment On'],
#            handler_map={AnyObject: AnyObjectHandler2()})
# plt.legend([AnyObject()], ['Treatment Off'],
#            handler_map={AnyObject: AnyObjectHandler3()})

extrapolatedlist = os.listdir("../Experts_policy/extrapolated")
print(len(extrapolatedlist))
doseFileLise.sort()
fig, ax = plt.subplots(figsize = (20, 10))
CliniDosageDict = {}
for file in doseFileLise:
    patient = file[:10]
    patientData = pd.read_csv("../Data/dataTanaka/Bruchovsky_et_al/" + patient + ".txt", header=None)
    ONOFF = np.array(patientData.loc[:, 7])
    cpa = np.array(patientData.loc[~np.isnan(patientData.loc[:,2]), 2]).sum().item()
    leu = np.array(patientData.loc[~np.isnan(patientData.loc[:,3]), 3]).sum().item()
    drugOnDays = 0
    drugOffDays = 0
    dosage = [cpa, leu ]

    Days = np.array(patientData.loc[:, 9]) - np.array(patientData.loc[0, 9])
    for ii in range(len(ONOFF) - 1):
        if ONOFF[ii] == 1:
            drugOnDays+=Days[ii+1]-Days[ii]
            ax.barh(patient, Days[ii+1]-Days[ii], left = Days[ii], color = onColor, height = 0.8, tick_label = None, alpha = 0.4)
        else:
            drugOffDays+=Days[ii + 1] - Days[ii]
            ax.barh(patient, Days[ii + 1] - Days[ii], left=Days[ii], color=offColor, height=0.8, tick_label=None, alpha = 0.4)
    CliniDosageDict[patient] = dosage + [drugOnDays, drugOffDays]

for file in extrapolatedlist:
    patient = file[:10]
    if patient in patientLables:
        extraDose = pd.read_csv("../Experts_policy/extrapolated/" + file)
        patientData = pd.read_csv("../Data/dataTanaka/Bruchovsky_et_al/" + patient + ".txt", header=None)
        Days = np.array(patientData.loc[:, 9]) - np.array(patientData.loc[0, 9])
        left = Days[-1]
        extraDose = np.array(extraDose)[:,-1]
        for ii in range(extraDose.shape[0]):
            if left > 28*120:
                length = 28*121-left
            else:
                length = 28
            if extraDose[ii]:
                ax.barh(patient, length, left=left, color=onColor, height=0.8, tick_label=None)
            else:
                ax.barh(patient, length, left=left, color=offColor, height=0.8, tick_label=None)
            left += 28
            if left > 28*121:
                break



locs, labels = plt.yticks()
plt.yticks(locs, labels, rotation = 20)
plt.ylabel("1 $\longleftarrow$ Patient No. $\longrightarrow$ 108", fontsize = 24)
plt.xticks(fontsize = 22)
plt.xlabel("Time (Day)", fontsize = 24)
plt.xlim(-10, 3900)
plt.legend([ a1, a2, a3, a4], ['Treat-On', 'Treat-Off', "Extra-On", "Extra-Off" ],
           handler_map={a1: AnyObjectHandler12(), a2:AnyObjectHandler32(), a3: AnyObjectHandler1(), a4:  AnyObjectHandler3()}
           , fontsize = 18, loc= 'right')

if not os.path.exists("../Experts_Analysis/"):
    os.mkdir("../Experts_Analysis/")
plt.savefig("../Experts_Analysis/Clinician_Dose_Strategy.png", dpi = 500)
plt.show()
plt.close()

CliniDosageDf = pd.DataFrame.from_dict(CliniDosageDict, orient = "index")
RlTotalCPA = (dfpatientCPA * 200).sum(axis = 1)
RLTotalLEU = (dfpatientLEU * 7.5 ).sum(axis = 1)
RlDaysCPA_Free = pd.DataFrame([sum(dfpatientCPA.loc[index] == 0) * 28 for index in dfpatientCPA.index], index = dfpatientCPA.index)
RlDaysLEU_Free = pd.DataFrame([sum(dfpatientLEU.loc[index] == 0) * 28 for index in dfpatientLEU.index], index = dfpatientLEU.index)
RlDays_Free = pd.concat((RlDaysCPA_Free, RlDaysLEU_Free), axis =1)
RLDosageDf = pd.DataFrame()

# Drug administration days percentage
RLDoseDayPercentage =  RlDays_Free.min(axis = 1)/dfpatientTime['rl']
CliniDoseDayPercentage  = CliniDosageDf.loc[:,2]/CliniDosageDf.loc[:,[2,3]].sum(axis = 1)
RLDoseCPAAva = RlTotalCPA / dfpatientTime['rl']
RLDoseLEUAva = RLTotalLEU / dfpatientTime['rl']
CliniDoseCPAAva = CliniDosageDf.loc[:,0]/(CliniDosageDf.loc[:,2]+ CliniDosageDf.loc[:,3])
CliniDoseLEUAva = CliniDosageDf.loc[:,1]/(CliniDosageDf.loc[:,3] + CliniDosageDf.loc[:,2])
ClinipatientTime = (CliniDosageDf.loc[:,3] + CliniDosageDf.loc[:,2])
CPAReducePercentage =(CliniDoseCPAAva - RLDoseCPAAva)/CliniDoseCPAAva
LEUReducepercentage = (CliniDoseLEUAva - RLDoseLEUAva)/CliniDoseLEUAva

from scipy.stats import ttest_ind
Diff = RLDoseDayPercentage - CliniDoseDayPercentage
ttest_ind(RLDoseDayPercentage, CliniDoseDayPercentage)
ttest_ind(RLDoseCPAAva, CliniDoseCPAAva)
ttest_ind(RLDoseLEUAva, CliniDoseLEUAva)
ttest_ind(dfpatientTime, ClinipatientTime)
time_diff = dfpatientTime['rl'] - ClinipatientTime

plt.figure(figsize=(15, 10))
dfpatientTime["Experts' Policy"] = ClinipatientTime
dfpatientTime.rename(columns={"rl": "RL agent's Policy"}, inplace=True)
# ax = sns.boxplot(data = dfpatientTime, palette = "Paired", orient='v', width = 0.3)
ax = sns.swarmplot(data = dfpatientTime,color = "grey", size = 10)
mean_survival_time = dfpatientTime.mean()
plt.axhline(y = mean_survival_time)
# Add jitter with the swarmplot function
# ax = sns.swarmplot(markIndex[0], color = 'red', label = 'patient006')
# ax = sns.swarmplot(markIndex[1], color = 'yellow', label = 'patient011')
# plt.xlabel("Resistance Index $\gamma$",  fontsize=35)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylabel("Survival time / (Day)", fontsize = 25)
# adding transparency to colors
for patch in ax.artists:
     r, g, b, a = patch.get_facecolor()
     patch.set_facecolor((r, g, b, .3))
plt.show()


