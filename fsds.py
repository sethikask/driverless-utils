import sys, os

#Insert FSDS library path
fsds_lib_path = os.path.abspath("python-client")
sys.path.insert(0, fsds_lib_path)

import time
import math
import fsds #type: ignore
import numpy as np
import pandas as pd

#Plotting imports
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from collections import deque
from PyQt5 import QtWidgets
from matplotlib.patches import FancyArrowPatch

params = {
    'track_name': 'handlingTrack', #Track folder name
    'raceline_name': 'traj_race_cl.csv', #Rename racing line file to this

    #Preprocessing
    'n_period': 5, #How often should the data be sampled

    #Plotting
    'V_window_size': 50, #XLength of velocity window

    #Car data
    'wheelbase': 0.76, #metres
    'steering_mult': 2.5,
    'steering_coeff': 0.2422, #Input [-1,1] * coeff = steering angle (rad)

    #Simulator
    'velocity_mult': 1.0, #Factor of ideal velocity to target, 1=normal, 0.6=testing
    'target_dist': 10, #Distance to steering target point

    #Main loop
    'tick_rate': 50 #ms

}
dir_path = os.path.abspath("data")
raceline_path = os.path.join(dir_path, "inputs", params['track_name'], params['raceline_name'])

def initialise_fsds():
    #Connect to the AirSim simulator 
    client = fsds.FSDSClient(ip="172.20.144.1", port=41451)
    #Check network connection
    client.confirmConnection()
    #True = Disable manual control
    client.enableApiControl(True)
    return client

def preprocess_track_data(raceline_file_path):
    trajectory_data_frame = pd.read_csv(raceline_file_path, sep=',')
    trajectory_data_frame = trajectory_data_frame.iloc[::params['n_period']]

    s_opt = trajectory_data_frame['s_opt'].to_numpy()
    v_opt = trajectory_data_frame['v_opt'].to_numpy()
    x = trajectory_data_frame['x'].to_numpy()
    y = trajectory_data_frame['y'].to_numpy()
    z = trajectory_data_frame['z'].to_numpy()

    track_data = {
        's': np.array(s_opt),
        'x': np.array(x),
        'y': np.array(y),
        'z': np.array(z),
        'V': np.array(v_opt),
    }

    return track_data

def get_closest_point_index(current_pos, x_list, y_list, z_list, current_index, lookahead=20):
    n = len(x_list)
    indices = [(current_index + i) % n for i in range(lookahead)]

    coords = np.column_stack((
        np.take(x_list, indices),
        np.take(y_list, indices),
        np.take(z_list, indices)
    ))

    distances = np.linalg.norm(coords - current_pos, axis=1)
    return indices[np.argmin(distances)]

def calculate_target_vector(current_position, target_position):
    target_vector = target_position - current_position
    target_vector_norm = np.linalg.norm(target_vector)
    if target_vector_norm > 0:
        target_vector = 20 * target_vector / target_vector_norm
    return target_vector

def get_angle_between(vector1, vector2):
    vector1 = vector1[:2]
    vector2 = vector2[:2]

    dot = np.dot(vector1, vector2)
    cross = vector1[0]*vector2[1] - vector1[1]*vector2[0]

    return np.arctan2(cross, dot)

class Plotter:
    def __init__(self, app, max_V=90):
        plt.ion()

        self.fig_main, self.ax_main = plt.subplots(2, 1)
        self.app = app

        self.ax_main[0].set_xlim(-95,100)
        self.ax_main[0].set_ylim(-95, 10)
        self.sc_main = self.ax_main[0].scatter([], [], c=[], cmap="plasma")
        self.cbar_main = self.fig_main.colorbar(self.sc_main, ax=self.ax_main[0])

        # --- Data buffers ---
        self.xdata_s = deque(maxlen=params['V_window_size'])
        self.ydata_true = deque(maxlen=params['V_window_size'])
        self.ydata_exp = deque(maxlen=params['V_window_size'])

        self.line_true, = self.ax_main[1].plot([], [], '-o', markersize=4, label='True')
        self.line_exp, = self.ax_main[1].plot([], [], '-o', markersize=4, label='Expected')
        self.ax_main[1].set_xlim(0, params['V_window_size'] - 1)
        self.ax_main[1].set_ylim(0, max_V + 10)  

        self.velocity_arrow = FancyArrowPatch(
            (0,0 ),
            (30, 20),
            color="red",
            arrowstyle="->",
            mutation_scale=15
        )
        self.target_arrow = FancyArrowPatch(
            (0,0 ),
            (20, 30),
            color="green",
            arrowstyle="->",
            mutation_scale=15
        )
        self.ax_main[0].add_patch(self.velocity_arrow)
        self.ax_main[0].add_patch(self.target_arrow)
        self.current_sc = self.ax_main[0].scatter(0, 0, c='red', s=100, label='Current Position')
        self.target_sc = self.ax_main[0].scatter(0, 0, c='green', s=100, label='Target')

    def refresh_racing_line(self, x, y):
        order = np.linspace(0, 1, len(x))
        self.sc_main.set_offsets(np.column_stack((x, y)))
        self.sc_main.set_array(order)
        self.cbar_main.update_normal(self.sc_main)

    def refresh_current_position(self, x, y):
        self.current_sc.set_offsets(np.column_stack((x, y)))

    def refresh_target_position(self, x, y):
        self.target_sc.set_offsets(np.column_stack((x, y)))
    
    def refresh_vectors(self, current_position, velocity_vector, target_vector):
        current_position = current_position[:2]
        velocity_vector = velocity_vector[:2]
        target_vector = target_vector[:2]

        self.velocity_arrow.set_positions(current_position, current_position + velocity_vector*2)
        self.target_arrow.set_positions(current_position, target_vector + current_position)

    def refresh_velocities(self, current_total_s, current_velocity, expected_velocity):
        self.xdata_s.append(current_total_s)
        self.ydata_true.append(current_velocity)
        self.ydata_exp.append(expected_velocity)

        xvals = range(len(self.xdata_s))
        self.line_true.set_data(xvals, self.ydata_true)
        self.line_exp.set_data(xvals, self.ydata_exp)

    def refresh(self):
        self.fig_main.canvas.draw_idle()
        self.app.processEvents()

class Controller:
    def __init__(self):
        pass

    def calculate_throttle(self, car_controls, v_current_magnitude, v_desired):
        throttle_coeff = 0.5
        if v_current_magnitude < v_desired:
            car_controls.throttle = np.clip(throttle_coeff * (v_desired - v_current_magnitude), 0, 1)
            car_controls.brake = 0.0
        else:
            car_controls.throttle = 0.0

        return car_controls

    def calculate_braking(self, car_controls, v_current_magnitude, v_desired):
        brake_coeff = 0.2
        if (v_desired - v_current_magnitude) < -0.75:
            car_controls.brake = abs(v_desired-v_current_magnitude) * brake_coeff
        return car_controls
    
    def calculate_steering(self, car_controls, current_velocity, target_vector):
        max_steering = 1
        target_distance = np.linalg.norm(target_vector)
        target_angle = get_angle_between(target_vector, current_velocity)

        required_angle = np.arctan((params['steering_mult'] * 2 * params['wheelbase'] * np.sin(target_angle))/target_distance)

        required_steering = np.clip(required_angle/params['steering_coeff'], -max_steering, max_steering)
        car_controls.steering = required_steering
        
        return car_controls

class Simulator:
    def __init__(self, track_data):
        self.controller = Controller()
        self.client = initialise_fsds()
        self.client.reset()

        #Allow manual control after initialisation
        #self.client.enableApiControl(False)
        #time.sleep(10)
        self.client.enableApiControl(True)
        self.VELOCITY_MULT = params['velocity_mult']

        avg_diff = np.mean(np.diff(track_data['s']))
        self.TARGET_OFFSET = int(math.floor(params['target_dist']/avg_diff))

        self.track_data = track_data
        
        self.progress_index = 0
        self.target_index = self.TARGET_OFFSET
        self.n_points = len(self.track_data['s'])
        self.track_length = max(self.track_data['s'])
        self.current_s_mod = self.track_data['s'][0]
        self.total_distanced_travelled = 0.0

        self.update_car_state()

    def update_car_state(self):
        self.carState = self.client.getCarState()
        self.current_speed = self.carState.kinematics_estimated.linear_velocity
        self.current_speed = np.array([self.current_speed.x_val, self.current_speed.y_val, self.current_speed.z_val])
        self.current_speed_magnitude = np.linalg.norm(self.current_speed)

        self.current_position = self.carState.kinematics_estimated.position
        self.current_position = np.array([self.current_position.x_val, self.current_position.y_val, self.current_position.z_val])

        self.current_acceleration = self.carState.kinematics_estimated.linear_acceleration
        self.current_acceleration = np.array([self.current_acceleration.x_val, self.current_acceleration.y_val, self.current_acceleration.z_val])
        self.current_acceleration_magnitude = np.linalg.norm(self.current_acceleration)

        target_position=np.array([self.track_data['x'][self.target_index], self.track_data['y'][self.target_index], self.track_data['z'][self.target_index]])
        self.target_vector = calculate_target_vector(self.current_position, target_position)

    def refresh_index(self):
        #Find closest point ahead on the racing line
        self.progress_index = get_closest_point_index(self.current_position, self.track_data['x'], self.track_data['y'], self.track_data['z'], self.progress_index, lookahead=25)

        current_s = self.track_data['s'][self.progress_index]
        ds = current_s - self.current_s_mod

        if ds < -0.5 * self.track_length:
            ds += self.track_length
        self.total_distanced_travelled += max(ds, 0.0)

        self.current_s_mod = current_s
        self.target_index = (self.progress_index + self.TARGET_OFFSET) % self.n_points

    def control_vehicle(self):
        car_controls = fsds.CarControls()
        self.refresh_index()
        v_desired = self.track_data['V'][self.progress_index] * self.VELOCITY_MULT

        #Use current position for desired velocity, target position for desired steering
        car_controls = self.controller.calculate_throttle(car_controls, self.current_speed_magnitude, v_desired)
        car_controls = self.controller.calculate_braking(car_controls, self.current_speed_magnitude, v_desired)
        car_controls = self.controller.calculate_steering(car_controls, self.current_speed, self.target_vector)

        #print(car_controls.throttle, car_controls.brake, car_controls.steering)

        self.client.setCarControls(car_controls)

if __name__ == '__main__':

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    track_data = preprocess_track_data(raceline_path)

    #Initialise plots
    plot = Plotter(app=app, max_V=max(track_data['V']))
    plot.refresh_racing_line(track_data['x'], track_data['y'])
    plot.refresh()
    
    #Initialise simulator
    sim = Simulator(track_data=track_data)
    client = sim.client

    while True:
        prev_index = sim.progress_index
        sim.update_car_state()
        sim.control_vehicle()

        plot.refresh_current_position(sim.current_position[0], sim.current_position[1])
        plot.refresh_target_position(track_data['x'][sim.target_index], track_data['y'][sim.target_index])
        plot.refresh_vectors(sim.current_position, sim.current_speed, sim.target_vector)
        if prev_index != sim.progress_index:
            plot.refresh_velocities(sim.total_distanced_travelled, sim.current_speed_magnitude, track_data['V'][sim.progress_index]*sim.VELOCITY_MULT)
        plot.refresh()

        time.sleep(params['tick_rate']/1000)