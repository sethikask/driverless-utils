import pandas as pd
import numpy as np
import os

"""
Convert centreline data into sim-friendly bounds data
"""

#How often to sample input data
N_DIVISOR = 20
#Safety factor (metres)
N_SAFETY = 0.5

dir_path = os.path.abspath("data/")
name = "handlingTrack"

input_dir = os.path.join(dir_path, "inputs", name)
output_dir = os.path.join(dir_path, "outputs", name)

trackData = os.path.join(input_dir, "traj_ltpl_cl.csv")
#raceData = os.path.join(input_dir, "traj_race_cl.csv")
output_name = "output_fsds.csv"
output_path = os.path.join(output_dir, output_name)

trackDataFrame = pd.read_csv(trackData)

x = trackDataFrame['# x_ref_m'].to_numpy()
y = trackDataFrame[' y_ref_m'].to_numpy()
w_r = trackDataFrame[' width_right_m'].to_numpy() + N_SAFETY
w_l = trackDataFrame[' width_left_m'].to_numpy() + N_SAFETY
x_normvec = trackDataFrame[' x_normvec_m'].to_numpy()
y_normvec = trackDataFrame[' y_normvec_m'].to_numpy()

start_width = w_l[0] + w_r[0]

output_data = pd.DataFrame(columns=('colour', 'x', 'y', 'unk1', 'unk2', 'unk3', 'unk4'))

output_data.loc[0] = ['big_orange', 0, w_l[0], 0, 0, 0, 0]
output_data.loc[1] = ['big_orange', 0, -w_r[0], 0, 0, 0, 0]

for i in range(len(x)):
    if i%N_DIVISOR != 0:
        continue

    x_left = x[i] + x_normvec[i] * w_l[i]
    y_left = y[i] + y_normvec[i] * w_l[i]

    x_right = x[i] - x_normvec[i] * w_r[i]
    y_right = y[i] - y_normvec[i] * w_r[i]

    output_data.loc[len(output_data)] = ['blue', x_left, y_left, 0, 0, 0, 0]
    output_data.loc[len(output_data)] = ['yellow', x_right, y_right, 0, 0, 0, 0]

output_data.to_csv(output_path, index=False, header=False)