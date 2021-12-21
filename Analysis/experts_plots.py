import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import bernoulli
import xitorch
import random
from Analysis.tMGLV import CancerODEGlv_CPU, solve_ivp
from Analysis.LoadData import LoadData
import seaborn as sns

if not os.path.exists("../Experts_policy/extrapolated/"):
    os.mkdir("../Experts_policy/extrapolated")


parsdir = "../Data/model_pars"
parslist = os.listdir(parsdir)
mean_v = 5
mean_psa = 22.1
alldata = LoadData().Double_Drug()
for args in parslist:
    args = "Args_patient011.csv"
    pars_df = pd.read_csv(parsdir + '/' + args)
    patientNo = args[5:(-4)]
    #patientNo = "patient011"
    print(patientNo)
    A, K, states, final_pars, best_pars = [torch.from_numpy(np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])])).float() for
                                           i in range(5)]
    actions_seq = np.array(pd.read_csv("../Experts_policy/" + patientNo + "_actions_seqs.csv"))[:, 0]
    original_steps = actions_seq.shape[0]
    actions_seqs_prediction = deque(actions_seq[np.where(actions_seq != 0)[0]])
    Init = states[:3]
    data = alldata[patientNo]
    if patientNo == "patient002":
        data = data[:84]
    if patientNo == "patient046":
        data[43:46, 1] -= 10
    if patientNo == "patient056":
        data[46, 1] = (data[44, 1] + data[48, 1]) / 2
    if patientNo == "patient086":
        data[1, 1] = (data[1, 1] + data[8, 1]) / 2
    if patientNo == "patient104":
        data = data[:(-3)]
    Days = data[:, 6] - data[0, 6]
    PSA = data[:, 1]
    OnOff = data[:, 5]
    index = np.where(np.isnan(PSA))[0]
    PSA = torch.from_numpy(np.delete(PSA, index)).float()
    DAYS = np.delete(Days, index)
    inputs = torch.linspace(start=Days[0], end=Days[-1], steps=int(Days[-1] - Days[0]) + 1, dtype=torch.float)
    cancerode = CancerODEGlv_CPU(patientNo, A=A, K=K, pars=best_pars)
    OriginalOut = solve_ivp(cancerode.forward, ts=inputs, y0=Init, params=(), atol=1e-08, rtol=1e-05)
    OriginalOut = OriginalOut.detach().numpy()
    ad = OriginalOut[:, 0]
    ai = OriginalOut[:, 1]
    psa = OriginalOut[:, -1]
    x = inputs.numpy()

    # prospective eval of done
    original_metastasis_ai_deque = deque(maxlen=121)
    original_metastasis_ad_deque = deque(maxlen=121)
    slicing = np.linspace(start = 0, stop = Days[-1], endpoint = False, num = int(Days[-1] // 28), dtype = np.int) # DAYS.astype(np.int)
    # adMeasured = ad[slicing]
    # aiMeasured = ai[slicing]
    Truncated_flag = False
    for ss in slicing:
        adMeasured = ad[ss]
        aiMeasured = ai[ss]
        try:
            metastasis_ad = bernoulli.rvs((adMeasured / K[0]) ** 1.5, size=1).item() if adMeasured / K[0] > 0.5 else 0
            metastasis_ai = bernoulli.rvs((aiMeasured / K[1]) ** 1.5, size=1).item() if aiMeasured / K[1] > 0.5 else 0
        except ValueError:
            print(adMeasured / K[0])
            print(aiMeasured / K[1])
        original_metastasis_ai_deque.append(metastasis_ai)
        original_metastasis_ad_deque.append(metastasis_ad)
        done = bool(original_metastasis_ad_deque.count(1) >= 12) or bool(original_metastasis_ai_deque.count(1) >= 12) or adMeasured >= K[0] or aiMeasured >= K[1]
        if done:
            Truncated_flag = True
            flagss = ss
            TruncatedAD = ad[:ss]
            TruncatedAD1 = ad[ss:]
            TruncatedAI = ai[:ss]
            TruncatedAI1 = ai[ss:]
            TruncatedPSA = psa[:ss]
            TruncatedPSA1 = psa[ss:]
            TruncatedX = x[:ss]
            TruncatedX1 = x[ss:]
            break


    def Done(x, y, metastasis_ad_deque=original_metastasis_ad_deque, metastasis_ai_deque=original_metastasis_ai_deque,
             s1=original_steps, s2=0):  # x: ad y: ai
        metastasis_ad = bernoulli.rvs((x / K[0]) ** 1.5, size=1).item() if x / K[0] > 0.5 else 0
        metastasis_ai = bernoulli.rvs((y / K[1]) ** 1.5, size=1).item() if y / K[1] > 0.5 else 0
        metastasis_ad_deque.append(metastasis_ad)
        metastasis_ai_deque.append(metastasis_ai)
        mask = bool(
            x > K[0]
            or y >= K[1]
            or bool(s1 + s2 > 121)
            or bool(metastasis_ad_deque.count(1) >= 12)
            or bool(metastasis_ai_deque.count(1) >= 12)
        )
        return mask


    ending = inputs[-1]
    new_inits = torch.from_numpy(np.array([ad[-1], ai[-1], psa[-1]])).float()
    new_ai = ai[-1];
    new_ad = ad[-1];
    new_psa = psa[-1]
    # in the doctor's policy they can only consider the psa level
    # predict
    PredictOut = []
    extraActSeq = []
    savepath = "../Experts_figs/" + patientNo
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    if not Truncated_flag and not done:
        temp = OnOff[::-1]
        __action = temp[0]
        dose_times = 0
        # define the last action is drug administrated or not
        if temp[0]:
            for ii in range(temp.shape[0]):
                dose_times += 1
                if temp[ii] == 1 and temp[ii + 1] == 0:
                    break
        max_dosage_times = 8
        new_steps = 0
        while not done:
            if bool(__action) and dose_times < max_dosage_times:
                action = actions_seqs_prediction.popleft()
                actions_seqs_prediction.append(action)
                dose_times += 1
            else:
                if new_psa > 10:
                    action = actions_seqs_prediction.popleft()
                    actions_seqs_prediction.append(action)
                    dose_times += 1
                else:
                    action = 0
                    dose_times = 0
            __action = action
            extraActSeq.append(action)
            new_steps += 1
            t_stamp = torch.linspace(start=ending, end=ending + 27, steps=28, dtype=torch.float)
            new_out = solve_ivp(cancerode.forward, ts=t_stamp, y0=new_inits, params=(int(action), ending,), atol=1e-08, rtol=1e-05)
            new_ad = new_out[:, 0].detach().numpy()
            new_ai = new_out[:, 1].detach().numpy()
            new_psa = new_out[:, -1].detach().numpy()
            ending = t_stamp[-1]
            new_psa = new_psa[-1]
            new_ad = new_ad[-1]
            new_ai = new_ai[-1]
            new_inits = torch.from_numpy(np.array([new_ad, new_ai, new_psa])).float()

            done = Done(new_ad, new_ai, s2=new_steps)
            PredictOut.append(new_out.detach().numpy())
            # OUT = np.concatenate((OUT, new_out.detach().numpy()))
        extraActDf = pd.DataFrame(extraActSeq)
        extraActDf.to_csv("../Experts_policy/extrapolated/" + patientNo + "_extrapolated_doseSeq.csv")
        PredictOut = np.vstack(PredictOut)
        AllOut = np.concatenate((OriginalOut, PredictOut))
        x = np.arange(AllOut.shape[0])
        ad = AllOut[:, 0]
        ai = AllOut[:, 1]
        psa = AllOut[:, 2]
        OriginalX = x[:int(Days[-1])]
        PredictX = x[int(Days[-1]):]
        OriginalAI = ai[:int(Days[-1])]
        PredictAI = ai[int(Days[-1]):]
        OriginalAD = ad[:int(Days[-1])]
        PredictAD = ad[int(Days[-1]):]
        OriginalPSA = psa[:int(Days[-1])]
        PredictPSA = psa[int(Days[-1]):]
        cs = sns.color_palette("Paired")
        plt.style.use('seaborn')
        plt.style.use(["science", "nature"])
        plt.figure(figsize=(21, 7))
        plt.subplot(1, 2, 1)
        plt.scatter(DAYS, PSA, marker="*", alpha=0.6, s = 80)
        plt.xlim(-20, x[-1]+20)
        plt.text(-300, 31, "a", fontdict={'size': '32', 'color': 'black', "family": 'Times New Roman'},weight='bold')
        plt.plot(OriginalX, OriginalPSA, c = cs[1], lw = 2)
        plt.plot(PredictX, PredictPSA, c='red', ls='--', lw =2 )
        plt.axvspan(xmin=PredictX[0], xmax=PredictX[-1], ymin=0, facecolor="green", alpha=0.2, label="Predict")
        plt.xlabel("Time (Days)",  fontsize=22)
        plt.ylabel("PSA level (ug/L)",  fontsize=20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.legend(loc="upper left",  fontsize=20)
        plt.subplot(1, 2, 2)
        plt.xlim(-20, x[-1] + 20)
        plt.axvspan(xmin=PredictX[0], xmax=PredictX[-1], ymin=0, facecolor="green", alpha=0.2, label="Predict")
        plt.plot(OriginalX, OriginalAD, label="HD", c = cs[1], lw=2)
        plt.plot(OriginalX, OriginalAI, label="HI", c=cs[3], lw=2)
        plt.plot(PredictX, PredictAD, c=cs[5], ls='--', lw=2)
        plt.plot(PredictX, PredictAI, c=cs[7], ls='--', lw=2)
        plt.xlabel("Time (Days)",  fontsize=20)
        plt.ylabel("Cell counts",  fontsize=20)
        plt.legend(loc = "upper left",  fontsize=20, ncol =3)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.savefig(savepath+ "/Evolution_" + patientNo + ".png", dpi=300)
        # plt.show()
        plt.close()
    else:
        # Initialize the figure style
        plt.style.use('seaborn')
        plt.style.use(["science", "nature"])
        plt.figure(figsize=(21, 7))
        plt.subplot(1, 2, 1)
        plt.scatter(DAYS, PSA, marker="*", alpha=0.6, s = 80)
        plt.xlim(-20, x[-1] + 20)
        plt.text(-200, 32, "b", fontdict={'size': 32, 'color': 'black', "family": 'Times New Roman'},weight='bold')
        plt.plot(TruncatedX, TruncatedPSA, lw=2)
        plt.plot(TruncatedX1, TruncatedPSA1, c='grey',lw=2)
        plt.axvspan(xmin = flagss, xmax = x[-1], ymin = 0, facecolor = 'grey',alpha=0.3, label = "Done")
        plt.xlabel("Time (Days)",  fontsize=20)
        plt.ylabel("PSA level (ug/L)",  fontsize=20)
        plt.legend(loc='upper left',  fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplot(1, 2, 2)
        plt.xlim(-20, x[-1] + 20)
        plt.axvspan(xmin=flagss, xmax=x[-1], ymin=0, facecolor='grey', alpha=0.3, label="Done")
        plt.plot(TruncatedX, TruncatedAD, label="HD",lw=2)
        plt.plot(TruncatedX, TruncatedAI, label="HI",lw=2)
        plt.plot(TruncatedX1, TruncatedAD1, c='grey',lw=2)
        plt.plot(TruncatedX1, TruncatedAI1, c='grey',lw=2)

        # plt.annotate("", xy=(flagss + 200, plt.ylim()[1] * 0.9), xytext=(flagss, plt.ylim()[1] * 0.9), arrowprops=dict(arrowstyle="->"))
        # plt.text((flagss+x[-1])/2 - 100, plt.ylim()[1] * 0.8, "Done" , c='grey')
        # plt.vlines(x=flagss, ymin = 0, ymax = max(TruncatedAD) , colors='red', linestyles ='dashed', linewidths = 1)
        plt.xlabel("Time (Days)",  fontsize=20)
        plt.ylabel("Cell counts",  fontsize=20)
        plt.legend(loc='upper left',  fontsize=20, ncol = 3)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(savepath + "/Evolution_" + patientNo + ".png", dpi=300)
        # plt.show()
        plt.close()