#!/bin/bash

cd /home/yitao/CancerNN/Model_creation

# 62 patients
for ii in {1..108}; do nohup python -u  xitorch_glv_cpu_10.py -n $ii > mutations/model_infos/glvmodel_mutations_$ii.log 2>&1 & done
# retrain the not good patient ODE models
for ii in  6 25 36 60 62 63 75 85 86 87 95 99 101 105; do nohup python -u  xitorch_glv_cpu_10.py -n $ii > new_inits2/model_infos/retrained_glvmodel_new_inits3_$ii.log 2>&1 & done
# for under estimation, we increase the capacity of resistance cell
for ii in 13 19 54 60 83 87 95 96 100 101 105; do  python -u  xitorch_glv_cpu_10.py -n $ii > adptive_inits/model_infos/under_retrained_glvmodel_new_inits2_$ii.log 2>&1 & done
for ii in 24 25 36 40 44 50 56 62 63 75 86 97; do  python -u  xitorch_glv_cpu_10.py -n $ii > adptive_inits/model_infos/under_retrained_glvmodel_new_inits2_$ii.log 2>&1 & done
for ii in {1..108}; do nohup python -u  ode_model_test.py -n $ii > test-sigmoid/model_infos/sigmoid_glv_$ii.log 2>&1 & done
nohup python -u  xitorch_glv_cpu_10.py -n 11 > test/model_infos/test_glv_11.log 2>&1 &

nohup python -u  ode_model_test.py -n 11 > test/model_infos/test_glv_11.log 2>&1 &

nohup python -u  train.py --seed 29999 > ./logs/gym_cancer:CancerControl-v0/patient011/train29999.log 2>&1 &

for ii in 4 36 46 50 62 63 83 87 101 ; do nohup python -u  ode_model_test.py -n $ii > ./retrain-sigmoid/model_infos/retrained_sigmoid_glv$ii.log 2>&1 & done
for ii in 13 16 17 32 56 71 78 79 83 91 92 104 105; do nohup python -u  ode_model_test.py -n $ii > ./retrain-sigmoid/model_infos/retrained_sigmoid_glv$ii.log 2>&1 & done
56 63 75 83 87 91 99 101 102 105 106 108
11 1 62 46 3 4 13
for ii in {1..108}; do nohup python -u  ode_model_test.py -n $ii > retrain-sigmoid/model_infos/glvmodel_sigmoid_$ii.log 2>&1 & done

nohup python -u main.py  > ./test.log 2>&1 &
nohup python -u  ode_model_test.py -n 13 > ./retrain-sigmoid/model_infos/retrained_sigmoid_glv4_13.log 2>&1 &
for ii in 24 46 56; do nohup python -u  train.py -n $ii --cuda 0 --m1 0.8 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train$ii.log 2>&1 & done
for ii in 19 25 37 ; do nohup python -u  train.py -n $ii --cuda 1 --m1 0.8 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train$ii.log 2>&1 & done
for ii in 36 63 75 105 ; do nohup python -u  train.py -n $ii --cuda 1 --m1 0.8 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train$ii.log 2>&1 & done

find . -name '._*'  -type d -print -exec rm -rf {} \;
find . -type d -name 'sacd-seed0-20211021-0849*' -exec rm -rf {} +
rm -rf 'sacd-seed0-20211019*'
nohup python -u  train.py -n 24 --cuda 1 > ./logs/gym_cancer:CancerControl-v0/train24.log 2>&1 &

# for cuda 0 run 0-10 for cuda1 run 11-20
for ii in {1..10}; do nohup python -u  train.py -n $ii > ./new_logs/gym_cancer:CancerControl-v0/train$ii.log 2>&1 & done
# now finish 0-25

nohup python -u  train.py -n 6 --cuda 0 --m1 0.8 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_4.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 0 --m1 0.6 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_5.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 0 --m1 0.4 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_6.log 2>&1 &

nohup python -u  train.py -n 6 --cuda 0 --m1 0.9 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_09.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 0 --m1 0.75 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_075.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 0 --m1 0.7 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_07.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 0 --m1 0.55 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_055.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 0 --m1 0.5 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_05.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 1 --m1 0.35 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_035.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 1 --m1 0.3 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_03.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 1 --m1 0.25 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_025.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 1 --m1 0.2 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_02.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 1 --m1 0.15 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_015.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 1 --m1 0.1 --m2 12 > ./logs/gym_cancer:CancerControl-v0/train6_m1_01.log 2>&1 &

nohup python -u  train.py -n 6 --cuda 1 --m1 0.8 --m2 10 > ./logs/gym_cancer:CancerControl-v0/train6_1.log 2>&1 &
nohup python -u  train.py -n 6 --cuda 1 --m1 0.8 --m2 8 > ./logs/gym_cancer:CancerControl-v0/train6_2.log 2>&1 &
nohup python -u  train.py -n 15 --cuda 1 --m1 0.5 --m2 12 --seed 1998 > ./logs/gym_cancer:CancerControl-v0/train15_test1.log 2>&1 &

nohup python -u  train.py -n 6 --cuda 1 --m1 0.5 --m2 12 --seed 0 > ./logs/gym_cancer:CancerControl-v0/train6_seed_0.log 2>&1 &
for ii in {1..4}; do nohup python -u  train.py -n 6 --cuda 1 --m1 0.5 --m2 12 --seed $ii > ./logs/gym_cancer:CancerControl-v0/train6_seed_$ii.log 2>&1 & done
for ii in {5..8}; do nohup python -u  train.py -n 6 --cuda 0 --m1 0.5 --m2 12 --seed $ii > ./logs/gym_cancer:CancerControl-v0/train6_seed_$ii.log 2>&1 & done

1 2 3 4 6 11/ 12 13 15 16 17 19 20 24 25 29 30 31 /32 36 37 40 42 44 46 50 51 52 54 56 58/ 61 62 63 66
for ii in  106 108 ; do for kk in {1..10}; do nohup python -u ode_model_test.py -n $ii --t $kk > ./analysis-sigmoid/analysis_sigmoid__$ii_$kk.log 2>&1 & done; done


// 71 75 77 78 79 83 84 85 86 87 88// 91 92 93 94 95/ 96 97 99/ 100 101 102 104 105/ 106 108
for kk in {1..10}; do nohup python -u ode_model_test.py -n 61 --t $kk  > ./analysis-sigmoid/analysis_sigmoid_61_$kk.log 2>&1 & done
for kk in {1..10}; do nohup python -u ode_model_test.py -n 62 --t $kk  > ./analysis-sigmoid/analysis_sigmoid_62_$kk.log 2>&1 & done
for kk in {1..10}; do nohup python -u ode_model_test.py -n 63 --t $kk  > ./analysis-sigmoid/analysis_sigmoid_63_$kk.log 2>&1 & done
for kk in {1..10}; do nohup python -u ode_model_test.py -n 66 --t $kk  > ./analysis-sigmoid/analysis_sigmoid_66_$kk.log 2>&1 & done
for kk in {1..10}; do nohup python -u ode_model_test.py -n 17 --t $kk  > ./analysis-sigmoid/analysis_sigmoid_17_$kk.log 2>&1 & done
for kk in {1..10}; do nohup python -u ode_model_test.py -n 19 --t $kk  > ./analysis-sigmoid/analysis_sigmoid_19_$kk.log 2>&1 & done
for ii in {28295..28304}; do kill $ii; done
/12
28184 python
28185 python
28186 python
28187 python
28188 python
28189 python
28190 python
28191 python
28192 python
28193 python
/13
28194 python
28195 python
28196 python
28197 python
28198 python
28199 python
28200 python
28201 python
28202 python
28203 python
/15
28204 python
28205 python
28206 python
28207 python
28208 python
28209 python
28210 python
28211 python
28212 python
28213 python
/16
28214 python
28215 python
28216 python
28217 python
28218 python
28219 python
28220 python
28221 python
28222 python
28223 python
/17
28224 python
28225 python
28226 python
28227 python
28229 python
28230 python
28231 python
28232 python
28233 python
28234 python
/19
28235 python
28236 python
28237 python
28238 python
28239 python
28240 python
28241 python
28242 python
28243 python
28244 python
/20
28245 python
28246 python
28247 python
28248 python
28249 python
28250 python
28251 python
28252 python
28253 python
28254 python
/24
28255 python
28256 python
28257 python
28258 python
28259 python
28260 python
28261 python
28262 python
28263 python
28264 python
/25
28265 python
28266 python
28267 python
28268 python
28269 python
28270 python
28271 python
28272 python
28273 python
28274 python
/29
28275 python
28276 python
28277 python
28278 python
28279 python
28280 python
28281 python
28282 python
28283 python
28284 python
/30
28285 python
28286 python
28287 python
28288 python
28289 python
28290 python
28291 python
28292 python
28293 python
28294 python
/
28295 python
28296 python
28297 python
28298 python
28299 python
28300 python
28301 python
28302 python
28303 python
28304 python