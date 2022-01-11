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
import scipy.stats as st
#
# extra_path = "../Experts_policy/analysis/extrapolated"
# if not os.path.exists(extra_path):
#     os.mkdir(extra_path)
savepath = "../Experts_figs/analysis/"
if not os.path.exists(savepath):
    os.mkdir(savepath)
patientlist =  [1, 2, 3, 4, 6, 11, 12, 13, 15, 16, 17, 19, 20, 24, 25, 29, 30, 31, 32, 36, 37, 40, 42, 44, 46, 50, 51,
               52, 54, 56, 58, 61, 62, 63, 66, 71, 75, 77, 78, 79, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97,
               99, 100, 101, 102, 104, 105, 106, 108]
cell_size = 5.236e-10
mean_v = 5
Mean_psa = 22.1
alldata = LoadData().Double_Drug()
s_endlist = []
for no in patientlist:
    if len(str(no)) == 1:
        patientNo = "patient00" + str(no)
    elif len(str(no)) == 2:
        patientNo = "patient0" + str(no)
    else:
        patientNo = "patient" + str(no)
    # patientNo ='patient108'
    argslist = os.listdir('../GLV/analysis-sigmoid/model_pars/'+patientNo)
    argslist.sort()
    AD_List = []
    AI_List = []
    PSA_List = []
    X_List = []
    Truncated_flag_list = []
    for patient_args in argslist:
        pars_df = pd.read_csv('../GLV/analysis-sigmoid/model_pars/'+patientNo +'/' + patient_args)
        #patientNo = "patient011"
        print(patientNo)
        A, K, states, final_pars, best_pars = [torch.from_numpy(np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])])).float() for
                                               i in range(5)]
        actions_seq = np.array(pd.read_csv("../Experts_policy/" + patientNo + "_actions_seqs.csv"))[:, 0]
        original_steps = actions_seq.shape[0]
        actions_seqs_prediction = deque(actions_seq[np.where(actions_seq != 0)[0]])

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
        Init = torch.tensor([mean_v / Mean_psa * PSA[0] / cell_size, 1e-4 * K[1], PSA[0]],
                            dtype=torch.float)  # states[:3]
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

        flagss = None
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
                Truncated_flag_list.append(Truncated_flag)
                flagss = ss
                print(ss)
                TruncatedAD = ad[:ss]
                TruncatedAD1 = ad[ss:]
                TruncatedAI = ai[:ss]
                TruncatedAI1 = ai[ss:]
                TruncatedPSA = psa[:ss]
                TruncatedPSA1 = psa[ss:]
                TruncatedX = x[:ss]
                TruncatedX1 = x[ss:]
                AD = ad
                AI = ai
                Psa = psa
                X = np.array([ss, x[-1]])
                AD_List.append(AD)
                AI_List.append(AI)
                PSA_List.append(Psa)
                X_List.append(X)
                break

        s_endlist.append(flagss)
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
            # extraActDf.to_csv(extra_path + patient_args[5:-4] + "_extrapolated_doseSeq.csv")
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
            AD = ad
            AI = ai
            Psa = psa
            X = np.array([int(Days[-1]), x[-1]])
            AD_List.append(AD)
            AI_List.append(AI)
            PSA_List.append(Psa)
            X_List.append(X)
            Truncated_flag_list.append(Truncated_flag)
    Flag = sum(Truncated_flag_list)
    ad_list = []
    ai_list = []
    psa_list = []
    x_list = []
    X_array = np.stack(X_List)
    add_0_length = max(X_array[:, 1]) - X_array[:, 1]
    for i in range(len(AD_List)):
        AD_List[i] = np.concatenate((AD_List[i], -np.ones(int(add_0_length[i]))))
        AI_List[i] = np.concatenate((AI_List[i], -np.ones(int(add_0_length[i]))))
        PSA_List[i] = np.concatenate((PSA_List[i], -np.ones(int(add_0_length[i]))))
    AD_arr = np.stack(AD_List)
    AI_arr = np.stack(AI_List)
    PSA_arr = np.stack(PSA_List)
    mean_ad = np.array([np.mean(AD_arr[AD_arr[:, i] != -1, i], axis=0) for i in range(AD_arr.shape[1])])
    sd_ad = np.array([st.sem(AD_arr[AD_arr[:, i] != -1, i]) for i in range(AD_arr.shape[1])])
    low_ad_bound, high_ad_bound = st.t.interval(0.95, mean_ad.shape[0] - 1, loc=mean_ad, scale=sd_ad)

    mean_ai = np.array([np.mean(AI_arr[AI_arr[:, i] != -1, i], axis=0) for i in range(AI_arr.shape[1])])
    sd_ai = np.array([st.sem(AI_arr[AI_arr[:, i] != -1, i]) for i in range(AI_arr.shape[1])])
    low_ai_bound, high_ai_bound = st.t.interval(0.95, mean_ai.shape[0] - 1, loc=mean_ai, scale=sd_ai)

    mean_psa = np.array([np.mean(PSA_arr[PSA_arr[:, i] != -1, i], axis=0) for i in range(PSA_arr.shape[1])])
    sd_psa = np.array([st.sem(PSA_arr[PSA_arr[:, i] != -1, i]) for i in range(PSA_arr.shape[1])])
    low_psa_bound, high_psa_bound = st.t.interval(0.95, mean_psa.shape[0] - 1, loc=mean_psa, scale=sd_psa)
    cs = sns.color_palette("Paired")
    plt.style.use('seaborn')
    plt.style.use(["science", "nature"])
    plt.figure(figsize=(21, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(DAYS, PSA, marker="*", alpha=0.6, s=80)
    if Flag < 5:
        plt.plot(np.arange(int(Days[-1])), mean_psa[:int(Days[-1])], c=cs[1], lw=2)
        plt.fill_between(np.arange(int(np.mean(X_array[:, 1] + 1))), low_psa_bound[:int(np.mean(X_array[:, 1] + 1))], high_psa_bound[:int(np.mean(X_array[:, 1] + 1))], color=cs[0], alpha=0.5)
        predict_x = np.arange(int(Days[-1]), int(np.mean(X_array[:, 1] + 1)))
        plt.plot(predict_x, mean_psa[int(Days[-1]):int(np.mean(X_array[:, 1] + 1))], color=cs[1], lw=2, ls='--')
        plt.axvspan(xmin=int(Days[-1]), xmax= np.mean(X_array[:, 1] + 1), ymin=0, facecolor=cs[3], alpha=0.2, label="Predict")
    else:
        truncated_x = int(np.mean(X_array[Truncated_flag_list, 0]).item())
        plt.plot(np.arange(truncated_x), mean_psa[:truncated_x], c=cs[1], lw=2)
        plt.fill_between(np.arange(truncated_x), low_psa_bound[:truncated_x], high_psa_bound[:truncated_x], color=cs[0], alpha=0.5)
        done_x = np.arange(truncated_x, int(Days[-1])+1)
        plt.plot(done_x, mean_psa[truncated_x:int(Days[-1])+1], c='grey', lw=2)
        plt.fill_between(done_x, low_psa_bound[truncated_x:int(Days[-1])+1], high_psa_bound[truncated_x:int(Days[-1])+1], color='grey',
                         alpha=0.5)
        plt.axvspan(xmin=truncated_x, xmax=int(Days[-1]), ymin=0, facecolor='grey', alpha=0.2, label="Done")
    plt.xlabel("Time (Days)", fontsize=22)
    plt.ylabel("PSA level (ug/L)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper left", fontsize=20)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(int(Days[-1])), mean_ai[:int(Days[-1])], c=cs[3], lw=2, label='HI')
    plt.fill_between(np.arange(int(Days[-1])), low_ai_bound[:int(Days[-1])], high_ai_bound[:int(Days[-1])], color=cs[2], alpha=0.5)
    plt.plot(np.arange(int(Days[-1])), mean_ad[:int(Days[-1])], c=cs[1], lw=2, label='HD')
    plt.fill_between(np.arange(int(Days[-1])), low_ad_bound[:int(Days[-1])], high_ad_bound[:int(Days[-1])], color=cs[0], alpha=0.5)
    if Flag < 5:
        predict_x = np.arange(int(Days[-1]), int(np.mean(X_array[:, 1] + 1)))
        plt.plot(predict_x, mean_ad[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], ls='--', c=cs[1], lw=2)
        plt.fill_between(predict_x, low_ad_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], high_ad_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], color=cs[0],
                         alpha=0.5)
        plt.plot(predict_x, mean_ai[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], ls='--', lw=2, c=cs[3])
        plt.fill_between(predict_x, low_ai_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], high_ai_bound[int(Days[-1]): int(np.mean(X_array[:, 1] + 1))], color=cs[2],
                         alpha=0.5)
        plt.axvspan(xmin=int(Days[-1]), xmax=np.mean(X_array[:, 1] + 1), ymin=0, facecolor=cs[3], alpha=0.2, label="Predict")
    else:
        truncated_x = int(np.mean(X_array[Truncated_flag_list, 0]).item())
        done_x = np.arange(truncated_x, int(Days[-1]) + 1)
        plt.plot(done_x, mean_ad[truncated_x:int(Days[-1])+1], c='grey', lw=2)
        plt.fill_between(done_x, low_ad_bound[truncated_x:int(Days[-1])+1], high_ad_bound[truncated_x:int(Days[-1])+1], color='grey',
                         alpha=0.5)
        plt.plot(done_x, mean_ai[truncated_x:int(Days[-1])+1], c='grey', lw=2)
        plt.fill_between(done_x, low_ai_bound[truncated_x:int(Days[-1])+1], high_ai_bound[truncated_x:int(Days[-1])+1], color='grey',
                         alpha=0.5)
        plt.axvspan(xmin=truncated_x, xmax=int(Days[-1]), ymin=0, facecolor='grey', alpha=0.2, label="Done")
    plt.xlabel("Time (Days)", fontsize=20)
    plt.ylabel("Cell counts", fontsize=20)
    plt.legend(loc='upper left', fontsize=20, ncol=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig(savepath + patientNo + "_evolution.png", dpi=300)
    plt.show()


    # if Flag < 5:
    #     PredictPSA = []
    #     for i in range(len(AD_List)):
    #         if not Truncated_flag_list[i]:
    #             x_list.append(X_List[i])
    #             PredictPSA.append(PSA_List[i][X[0]:] for i in range(len(PSA_List)))
    #     X = [x_list[0][0]]
    #     X.append(np.mean([x[1] for x in x_list]))
    #     original_psa = mean_psa[:X[0]]
    #
    #
    #     cs = sns.color_palette("Paired")
    #     plt.style.use('seaborn')
    #     plt.style.use(["science", "nature"])
    #     plt.figure(figsize=(21, 7))
    #     plt.subplot(1, 2, 1)
    #     plt.scatter(DAYS, PSA, marker="*", alpha=0.6, s = 80)
    #     plt.xlim(-20, x[-1]+20)
    #     # plt.text(-x[-1] * 0.1, max(PSA) * 1.1, "c", fontdict={'size': '32', 'color': 'black', "family": 'Times New Roman'},weight='bold')
    #     plt.plot(OriginalX, OriginalPSA, c = cs[1], lw = 2)
    #     plt.plot(PredictX, PredictPSA, c=cs[1], ls='--', lw =2 )
    #     plt.axvspan(xmin=PredictX[0], xmax=PredictX[-1], ymin=0, facecolor="green", alpha=0.2, label="Predict")
    #     plt.xlabel("Time (Days)",  fontsize=22)
    #     plt.ylabel("PSA level (ug/L)",  fontsize=20)
    #     plt.xticks(np.arange(0, x[-1]+20, 500), np.arange(0, x[-1]+20, 500), fontsize=20)
    #     plt.yticks(fontsize = 20)
    #     plt.legend(loc="upper left",  fontsize=20)
    #     plt.subplot(1, 2, 2)
    #     plt.xlim(-20, x[-1]+20)
    #     plt.axvspan(xmin=PredictX[0], xmax=PredictX[-1], ymin=0, facecolor="green", alpha=0.2, label="Predict")
    #     plt.plot(OriginalX, OriginalAD, label="HD", c = cs[1], lw=2)
    #     plt.plot(OriginalX, OriginalAI, label="HI", c=cs[3], lw=2)
    #     plt.plot(PredictX, PredictAD, c=cs[1], ls='--', lw=2)
    #     plt.plot(PredictX, PredictAI, c=cs[3], ls='--', lw=2)
    #     plt.xlabel("Time (Days)",  fontsize=20)
    #     plt.ylabel("Cell counts",  fontsize=20)
    #     plt.legend(loc = "upper left",  fontsize=20, ncol =3)
    #     plt.xticks(np.arange(0, x[-1]+20, 500), np.arange(0, x[-1]+20, 500), fontsize=20)
    #     plt.yticks(fontsize=20)
    #
    #     plt.savefig(savepath + patientNo + "_evolution.png", dpi=300)
    #     # plt.show()
    #     plt.close()
    # else:
    #     # Initialize the figure style
    #     plt.style.use('seaborn')
    #     plt.style.use(["science", "nature"])
    #     plt.figure(figsize=(21, 7))
    #     plt.subplot(1, 2, 1)
    #     plt.scatter(DAYS, PSA, marker="*", alpha=0.6, s = 80)
    #     plt.xlim(-20, x[-1] + 20)
    #     # plt.text(-x[-1] * 0.1, max(PSA) * 1.1, "c", fontdict={'size': 32, 'color': 'black', "family": 'Times New Roman'},weight='bold')
    #     plt.plot(TruncatedX, TruncatedPSA, lw=2)
    #     plt.plot(TruncatedX1, TruncatedPSA1, c='grey',lw=2)
    #     plt.axvspan(xmin = flagss, xmax = x[-1], ymin = 0, facecolor = 'grey',alpha=0.3, label = "Done")
    #     plt.xlabel("Time (Days)",  fontsize=20)
    #     plt.ylabel("PSA level (ug/L)",  fontsize=20)
    #     plt.legend(loc='upper left',  fontsize=20)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.subplot(1, 2, 2)
    #     plt.xlim(-20, x[-1] + 20)
    #     plt.axvspan(xmin=flagss, xmax=x[-1], ymin=0, facecolor='grey', alpha=0.3, label="Done")
    #     plt.plot(TruncatedX, TruncatedAD, label="HD",lw=2)
    #     plt.plot(TruncatedX, TruncatedAI, label="HI",lw=2)
    #     plt.plot(TruncatedX1, TruncatedAD1, c='grey',lw=2)
    #     plt.plot(TruncatedX1, TruncatedAI1, c='grey',lw=2)
    #
    #     # plt.annotate("", xy=(flagss + 200, plt.ylim()[1] * 0.9), xytext=(flagss, plt.ylim()[1] * 0.9), arrowprops=dict(arrowstyle="->"))
    #     # plt.text((flagss+x[-1])/2 - 100, plt.ylim()[1] * 0.8, "Done" , c='grey')
    #     # plt.vlines(x=flagss, ymin = 0, ymax = max(TruncatedAD) , colors='red', linestyles ='dashed', linewidths = 1)
    #     plt.xlabel("Time (Days)",  fontsize=20)
    #     plt.ylabel("Cell counts",  fontsize=20)
    #     plt.legend(loc='upper left',  fontsize=20, ncol = 3)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.savefig(savepath + patientNo + "_evolution.png", dpi=300)
    #     # plt.show()
    #     plt.close()
# pd.DataFrame(s_endlist).to_csv('../Experts_policy/response_group_s_end_list.csv', index = None)