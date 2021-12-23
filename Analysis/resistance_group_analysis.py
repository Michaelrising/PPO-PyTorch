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

