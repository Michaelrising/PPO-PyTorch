import os
import glob
import time
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import argparse
import gym
from env.gym_cancer.envs.cancercontrol import CancerControl
import matplotlib.pyplot as plt
# import pybullet_envs
import seaborn as sns
from PPO import PPO


def set_device(cuda=None):
    print("============================================================================================")

    # set device to cpu or cuda
    device = torch.device('cpu')

    if torch.cuda.is_available() and cuda is not None:
        device = torch.device('cuda:' + str(cuda))
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")
    return device


#################################### Testing ###################################


def plot_figure(data, save_path, par = 0):
    if par:
        save_name = "_best_survival_time.png"
    else:
        save_name = '_best_reward.png'
    states = data["states"]
    doses = data["doses"]
    cs = sns.color_palette('Paired')
    x = np.arange(states.shape[0]) * 28
    ad = states[:, 0]
    ai = states[:, 1]
    psa = states[:, 2]
    fig = plt.figure(figsize=(15, 5))
    plt.style.use('seaborn')
    plt.style.use(['science', "nature"])
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, psa, linestyle="-", linewidth=2)
    # plt.scatter(x, psa, color=colors)
    ax1.set_xlabel("Time (Days)",  fontsize=20)
    ax1.set_ylabel("PSA level (ug/L)",  fontsize=20)
    ax1.tick_params(labelsize = 20)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, ad, label="HD", c = cs[1], lw=2)
    ax2.plot(x, ai, label="HI", c = cs[0], lw=2)
    ax2.set_xlabel("Time (Days)",  fontsize=20)
    ax2.set_ylabel("Cell counts",  fontsize=20)
    ax2.tick_params(labelsize=20)
    ax2.legend(loc='upper right',  fontsize=20)
    plt.savefig(save_path + "_evolution" + save_name, dpi=300)
    plt.close()

    COLOR_CPA = cs[0]
    COLOR_LEU = cs[1]
    x_dose = np.arange(0, doses.shape[0]) * 28
    fig = plt.figure(figsize=(7.5, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(x_dose, doses[:,0], color=COLOR_CPA, label='CPA', lw=2)
    ax1.set_xlabel("Days",  fontsize=20)
    ax1.set_ylabel("CPA (mg/ Day)", fontsize=20)
    # ax1.set_ylim(-5, 205)
    ax1.set_yticks(np.arange(0, 250, 50),  fontsize=20)
    ax1.tick_params(axis="y", labelcolor=COLOR_CPA,  labelsize=20)
    ax1.legend(loc=(0.03, 0.9), fontsize=20, facecolor=COLOR_CPA)
    plt.grid(False)
    ax2.plot(x_dose, doses[:, 1], color=COLOR_LEU, label='LEU', lw=2, ls='--')
    ax2.set_ylabel("LEU (ml/ Month)", fontsize=20)
    ax2.tick_params(axis="y", labelcolor=COLOR_LEU, labelsize=20)
    ax2.set_yticks(np.arange(0, 22.5, 7.5),  fontsize=20)
    plt.ylim(-.5, 15)
    ax2.legend(loc=(0.03, 0.8), fontsize=20, facecolor=COLOR_LEU)
    plt.savefig(save_path+'_dosages' + save_name, dpi=300)
    plt.close()
    return

def test(args, file, group = 'resistance'):
    ####### initialize environment hyperparameters ######

    env_name = args.env_id  # "RoboschoolWalker2d-v1"
    num_env = args.num_env
    max_updates = args.max_updates
    eval_interval = args.eval_interval
    model_save_start_updating_steps = args.model_save_start_updating_steps
    eval_times = args.eval_times

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 120  # max timesteps in one episode

    print_freq = 2  # print avg reward in the interval (in num updating steps)
    log_freq = 2  # log avg reward in the interval (in num updating steps)
    save_model_freq = eval_interval * 10  # save model frequency (in num updating steps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num updating steps)

    ####################################################
    ################ PPO hyperparameters ################

    decay_step_size = 500
    decay_ratio = 0.5
    update_timestep = 1  # update policy every n timesteps
    K_epochs = 4  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.00003  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    random_seed = args.seed  # set random seed if required (0 = no random seed)

    ########################### Env Parameters ##########################

    if len(str(args.number)) == 1:
        patientNo = "patient00" + str(args.number)
    elif len(str(args.number)) == 2:
        patientNo = "patient0" + str(args.number)
    else:
        patientNo = "patient" + str(args.number)
    # patientNo ="patient006"
    list_df = args.patients_pars[patientNo]
    A, K, states, pars, best_pars = [np.array(list_df.loc[i, ~np.isnan(list_df.loc[i, :])]) for i in range(5)]
    A = A.reshape(2, 2)
    # A = np.array(list_df.loc[0, ~np.isnan(list_df.loc[0, :])]).reshape(2, 2)
    # K = np.array(list_df.loc[1, ~np.isnan(list_df.loc[1, :])])
    # states = np.array(list_df.loc[2, ~np.isnan(list_df.loc[2, :])])
    # pars = np.array(list_df.loc[3, ~np.isnan(list_df.loc[3, :])])
    init_state = states[:3]
    terminate_state = states[3:]

    # default_acts = pd.read_csv("../Model_creation/test-sigmoid/model_actions/" + patientNo + "_actions_seqs.csv")
    # default_acts = np.array(default_acts)
    #
    # default_action = np.array(default_acts[:, 0], dtype=np.int)
    weight = np.ones(2) / 2
    base = 1.125
    m1 = args.m1
    m2 = args.m2
    drug_decay = args.drug_decay
    drug_length = 8

    patient = (A, K, best_pars, init_state, terminate_state, weight, base, m1, m2, drug_decay, drug_length)

    test_env = CancerControl(patient=patient)

    # state space dimension
    state_dim = test_env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = test_env.action_space.shape[0]
    else:
        action_dim = test_env.action_space.n


    # initialize a PPO agent
    ppo_agent = PPO(
            state_dim,
            action_dim,
            lr_actor,
            lr_critic,
            gamma,
            K_epochs,
            eps_clip,
            has_continuous_action_space,
            num_env,
            device,
            decay_step_size,
            decay_ratio,
            action_std)


    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    best_directory = "PPO_preTrained" + '/' + env_name + '/' + patientNo + "/" + "final/"
    best_name = os.listdir(best_directory)
    checkpoint_path = best_directory + max(best_name) # "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    checkpoint_path = "PPO_preTrained/" + group + "_group/" + file
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    record_states_high_reward = 0
    record_dose_high_reward = 0
    record_states_high_survival_time = 0
    record_dose_survival_month = 0
    test_running_reward = 0
    total_test_episodes = 20
    states = []
    doses = []
    record_reward = -1000
    record_survival_month = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        ep_survival_month = 0
        fea, state = test_env.reset()
        states.append(state)
        while True:
            _, action, _ = ppo_agent.greedy_select_action(fea)
            fea, state, reward, done, infos = test_env.step(action)
            states.append(state)
            doses.append(infos["dose"])
            ep_reward += reward
            ep_survival_month += 1
            if done:
                break
        if record_reward < ep_reward:
            record_reward = ep_reward
            record_states_high_reward = np.vstack(states.copy())
            record_dose_high_reward = np.vstack(doses)
        if record_survival_month < ep_survival_month:
            record_survival_month = ep_survival_month
            record_states_high_survival_time = np.vstack(states.copy())
            record_dose_survival_month = np.vstack(doses)
        states.clear()
        doses.clear()
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    test_env.close()
    # maximum rewards
    High_reward = {"states": record_states_high_reward, "doses": record_dose_high_reward}
    High_survival = {"states": record_states_high_survival_time, "doses": record_dose_survival_month}
    if not os.path.exists("./PPO_figs/" + group + "_group/"):
        os.makedirs("./PPO_figs/"+ group + "_group/")
    if not os.path.exists("./PPO_policy/" + group + "_group/"):
        os.makedirs("./PPO_policy/" + group + "_group/")
    if not os.path.exists("./PPO_states/" + group + "_group/"):
        os.makedirs("./PPO_states/" + group + "_group/")
    savepath = "./PPO_figs/" + group + "_group/" +patientNo
    plot_figure(High_reward, savepath , 0)
    plot_figure(High_survival, savepath, 1)
    pd.DataFrame(record_states_high_reward).to_csv(
        "./PPO_states/" + group + "_group/" + patientNo + "_converge_high_reward_states.csv")
    pd.DataFrame(record_states_high_survival_time).to_csv(
        "./PPO_states/" + group + "_group/" + patientNo + "_converge_high_survival_states.csv")
    pd.DataFrame(record_dose_high_reward).to_csv("./PPO_policy/" + group + "_group/" + patientNo + "_converge_high_reward_dosages.csv")
    pd.DataFrame(record_dose_survival_month).to_csv("./PPO_policy/" + group + "_group/" + patientNo + "_converge_high_survival_dosages.csv")
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    print("============================================================================================")

    parsdir = "./Data/model_pars"
    parslist = os.listdir(parsdir)
    patient_pars = {}
    patient_test = []
    patient_train = []
    # reading the ode parameters and the initial/terminal states
    for args in parslist:
        pars_df = pd.read_csv(parsdir + '/' + args)
        patient = args[5:(-4)]
        patient_train.append(patient)
        if patient not in patient_test:
            patient_pars[patient] = pars_df

    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'CancerControl-v0' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--env_id', type=str, default='gym_cancer:CancerControl-v0')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument("--cuda_cpu", type=str, default="cuda", help="Set device as cuda or cpu")
    parser.add_argument('--m1', type=float, default=0.5)
    parser.add_argument('--m2', type=int, default=12)
    parser.add_argument('--drug_decay', type=float, default=0.75, help="The decay rate for drug penalty")
    parser.add_argument('--seed', type=int, default=0)  # random.randint(0,100000))
    parser.add_argument('--patients_pars', type=dict, default=patient_pars)
    parser.add_argument('--patients_train', type=list, default=patient_train)
    parser.add_argument('--number', '-n', type=int, help='Patient No., int type, requested',
                        default=6)  # the only one argument needed to be inputted
    parser.add_argument('--num_env', type=int, help='number of environments',
                        default=2)
    parser.add_argument('--max_updates', type=int, help='max number of updating times',
                        default=int(1e5))
    parser.add_argument("--eval_interval", type=int, help="interval to evaluate the policy and plot figures",
                        default=50)
    parser.add_argument('--decayflag', type=bool, default=True, help='lr decay flag')
    parser.add_argument('--model_save_start_updating_steps', type=int, default=500,
                        help="The start steps of saving best model")
    parser.add_argument("--eval_times", type=int, default=10, help='The evaluation time of current policy')
    args = parser.parse_args()

    print("============================================================================================")

    device = set_device() if args.cuda_cpu == "cpu" else set_device(args.cuda)
    resistance_group = os.listdir('./PPO_preTrained/resistance_group')
    response_group = os.listdir('./PPO_preTrained/response_group')
    long_response_group = os.listdir('./PPO_preTrained/long_response_group')
    for file in response_group:
        Number = int(file[7:10])
        args.number = Number
        print(Number)
        test(args, file, "response")

    for file in resistance_group:
        Number = int(file[7:10])
        args.number = Number
        print(Number)
        test(args, file, 'resistance')
    for file in long_response_group:
        Number = int(file[7:10])
        args.number = Number
        print(Number)
        test(args, file, 'long_response')