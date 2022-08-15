import random
import time
import numpy as np
import pandas as pd
from enum import Enum
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QLineF, QPointF

from perlin_numpy.perlin_numpy import generate_perlin_noise_3d  # From local directory
#from perlin_numpy import generate_perlin_noise_3d  # From PIP


class PerlinFlowFieldAnimation:
    class DrawType(Enum):
        POINT = 0
        ELLIPSE = 1
        LINE = 2

    class DrawColorStyle(Enum):
        WHITE = 0                 # Pure white
        GRAYSCALE = 1             # Random varying gray intensities
        HSV_ANGLE = 2             # Hue based on the force vector angle, random S and V
        HUE_CHANGING = 3          # Updates the hue each draw per particle, fixed S at max and random V
        HUE_POS = 4               # Fixed hue based on location, random S and fixed V at max
        HUE_Y_POS_SAT_LENGTH = 5  # Changes hue and saturation based on line length and y position, fixed V at max
        HUE_Y_POS = 6             # Changes hue y position, random S and fixed V at max
        HUE_LENGTH = 7            # Changes hue based on line length, random S and fixed V at max
        HUE_SAT_LENGTH = 8        # Changes hue and saturation based on line length and fixed V at max
        HUE_FIXED = 9             # Fixed hue, random S and fixed V at max

    preconfigs = {
        'LINES_HUE_Y_POS': {
            'enable_acc': True, 'force_mag': 0.4, 'friction_mag': 0.1, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.HUE_Y_POS, 'particle_draw_size': 2,
            'stop_draw_after_max_length': False, 'alpha_value': 5, 'noise_divider': 16},
        'LINES_HUE_SAT_LENGTH': {
            'enable_acc': True, 'force_mag': 0.4, 'friction_mag': 0.1, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.HUE_SAT_LENGTH, 'particle_draw_size': 2,
            'stop_draw_after_max_length': False, 'alpha_value': 5, 'noise_divider': 16},
        'LINES_HUE_LENGTH': {
            'enable_acc': True, 'force_mag': 0.4, 'friction_mag': 0.1, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.HUE_LENGTH, 'particle_draw_size': 2,
            'stop_draw_after_max_length': False, 'alpha_value': 5, 'noise_divider': 16},
        'LINES_HUE_Y_POS_SAT_LENGTH_LIMITED': {
            'enable_acc': True, 'force_mag': 0.4, 'friction_mag': 0.1, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.HUE_Y_POS_SAT_LENGTH, 'particle_draw_size': 2,
            'stop_draw_after_max_length': True, 'alpha_value': 5, 'noise_divider': 16},
        'RGB_ELLIPSES_FOREVER': {
            'enable_acc': True, 'force_mag': 0.01, 'friction_mag': 0.5, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': True, 'particle_draw_type': DrawType.ELLIPSE,
            'particle_draw_color_style': DrawColorStyle.HUE_POS, 'particle_draw_size': 16,
            'stop_draw_after_max_length': False, 'alpha_value': 20, 'noise_divider': 16},
        'NO_ACC_LINES': {
            'enable_acc': False, 'angle_multiplier': 1.5,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.HUE_Y_POS_SAT_LENGTH, 'particle_draw_size': 2,
            'stop_draw_after_max_length': False, 'alpha_value': 10, 'noise_divider': 16},
        'ELLIPSES_HUE_CHANGING': {
            'enable_acc': True, 'force_mag': 0.2, 'friction_mag': 0.3, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.ELLIPSE,
            'particle_draw_color_style': DrawColorStyle.HUE_CHANGING, 'particle_draw_size': 4,
            'stop_draw_after_max_length': False, 'alpha_value': 10, 'noise_divider': 16},
        'ELLIPSES_HSV_ANGLE_SLOW': {
            'enable_acc': True, 'force_mag': 0.1, 'friction_mag': 0.4, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.ELLIPSE,
            'particle_draw_color_style': DrawColorStyle.HSV_ANGLE, 'particle_draw_size': 4,
            'stop_draw_after_max_length': False, 'alpha_value': 10, 'noise_divider': 16},
        'WHITE_WEB': {
            'enable_acc': True, 'force_mag': 1.0, 'friction_mag': 0.4, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.WHITE, 'particle_draw_size': 1,
            'stop_draw_after_max_length': False, 'alpha_value': 5, 'noise_divider': 16},
        'WHITE_SCRIBBLES': {
            'enable_acc': True, 'force_mag': 0.4, 'friction_mag': 0.0, 'max_vel': 3.0, 'angle_multiplier': 1.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.WHITE, 'particle_draw_size': 1,
            'stop_draw_after_max_length': False, 'alpha_value': 5, 'noise_divider': 16},
        'DUST': {
            'enable_acc': True, 'force_mag': 0.1, 'friction_mag': 0.5, 'max_vel': 0, 'angle_multiplier': 2.0,
            'draw_vectors': False, 'clear_each_update': True, 'particle_draw_type': DrawType.ELLIPSE,
            'particle_draw_color_style': DrawColorStyle.WHITE, 'particle_draw_size': 4,
            'stop_draw_after_max_length': False, 'alpha_value': 20, 'noise_divider': 16},
        'SHIMMER': {
            'enable_acc': True, 'force_mag': 0.1, 'friction_mag': 0.5, 'max_vel': 0, 'angle_multiplier': 2.0,
            'draw_vectors': False, 'clear_each_update': True, 'particle_draw_type': DrawType.ELLIPSE,
            'particle_draw_color_style': DrawColorStyle.GRAYSCALE, 'particle_draw_size': 4,
            'stop_draw_after_max_length': False, 'alpha_value': 30, 'noise_divider': 16},
        'WHITE_PIPES': {
            'enable_acc': True, 'force_mag': 0.5, 'friction_mag': 0.0, 'max_vel': 0.5, 'angle_multiplier': 3.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.WHITE, 'particle_draw_size': 4,
            'stop_draw_after_max_length': False, 'alpha_value': 5, 'noise_divider': 16},
        'COLORED_PIPES': {
            'enable_acc': True, 'force_mag': 0.5, 'friction_mag': 0.0, 'max_vel': 0.5, 'angle_multiplier': 3.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.HUE_CHANGING, 'particle_draw_size': 4,
            'stop_draw_after_max_length': False, 'alpha_value': 10, 'noise_divider': 16},
        'HUE_LENGTH_PIPES': {
            'enable_acc': True, 'force_mag': 0.5, 'friction_mag': 0.0, 'max_vel': 0.5, 'angle_multiplier': 3.0,
            'draw_vectors': False, 'clear_each_update': False, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.HUE_LENGTH, 'particle_draw_size': 4,
            'stop_draw_after_max_length': False, 'alpha_value': 10, 'noise_divider': 16},
        'DEBUG_NOISE': {
            'enable_acc': True, 'force_mag': 0.4, 'friction_mag': 0.1, 'max_vel': 0, 'angle_multiplier': 1.0,
            'draw_vectors': True, 'clear_each_update': True, 'particle_draw_type': DrawType.LINE,
            'particle_draw_color_style': DrawColorStyle.WHITE, 'particle_draw_size': 2,
            'stop_draw_after_max_length': False, 'alpha_value': 255, 'noise_divider': 16},
    }

    def __init__(self, image_width, image_height, noise_divider=16):
        self.image_width = image_width
        self.image_height = image_height
        self.image = QtGui.QImage(self.image_width, self.image_height, QtGui.QImage.Format_ARGB32_Premultiplied)
        self.image.fill(QtGui.QColor('black'))

        self.noise_divider = noise_divider
        self.noise_time_slices = 128

        self.angle_multiplier = 1
        self.num_of_particles = 500
        self.enable_acc = True
        self.force_mag = 0.4
        self.friction_mag = 0.1
        self.max_vel = 0
        self.step_size = 0.001 * max(self.image_width, self.image_height)

        self.draw_vectors = False
        self.clear_each_update = False
        self.current_iteration = 0

        self.base_hue = random.randint(0, 360)
        self.alpha_value = 20
        self.particle_draw_type = self.DrawType.LINE
        self.particle_draw_color_style = self.DrawColorStyle.HUE_Y_POS_SAT_LENGTH
        self.particle_draw_size = 2

        self.stop_draw_after_max_length = False
        self.max_line_length = self.image_width

        start = time.time()
        self.noise_width = int(self.image_width / self.noise_divider)
        self.noise_height = int(self.image_height / self.noise_divider)
        # Res width and height must be integers of the noise width and height
        if self.noise_width > 10 and self.noise_height > 10:
            self.res_width = int(self.noise_width / 10)
            self.res_height = int(self.noise_height / 10)
        else:
            self.res_width = 1
            self.res_height = 1

        print(f'{self.noise_width=}, {self.noise_height=}, {self.res_width=}, {self.res_height=}')
        self.perlin_3d_noise = generate_perlin_noise_3d(
            (self.noise_time_slices, self.noise_width, self.noise_height),
            (1, self.res_width, self.res_height),
            tileable=(True, False, False))
        # print(f'{self.perlin_3d_noise[0].shape=}')
        elapsed_time = time.time() - start
        print(f'--> elapsed_time for generate_perlin_noise_3d: {elapsed_time*1e3:.6f} ms')

        start = time.time()
        self.precomputed_vectors = []
        self.precomputed_vectors_dx = []
        self.precomputed_vectors_dy = []
        self.precomputed_vectors_angles = []
        self.precompute_vectors()
        elapsed_time = time.time() - start
        print(f'--> elapsed_time for pre-computing vectors: {elapsed_time*1e3:.6f} ms')

        self.particles_df = None

    def init_particles(self):
        pos_x = np.random.randint(0, self.image_width, size=self.num_of_particles)
        pos_y = np.random.randint(0, self.image_height, size=self.num_of_particles)
        self.particles_df = pd.DataFrame({
            'pos_x': pos_x, 'pos_y': pos_y, 'previous_pos_x': pos_x, 'previous_pos_y': pos_y,
            'vel_x': np.zeros(self.num_of_particles), 'vel_y': np.zeros(self.num_of_particles),
            'acc_x': np.zeros(self.num_of_particles), 'acc_y': np.zeros(self.num_of_particles),
            'x_index': np.zeros(self.num_of_particles), 'y_index': np.zeros(self.num_of_particles),
            'force_x': np.zeros(self.num_of_particles), 'force_y': np.zeros(self.num_of_particles),
            'angle': np.zeros(self.num_of_particles),
            'friction_x': np.zeros(self.num_of_particles), 'friction_y': np.zeros(self.num_of_particles),
            'line_length': np.zeros(self.num_of_particles), 'hue_value': np.random.randint(0, 360, size=self.num_of_particles)})

    def update_particles(self):
        time_iteration_index = self.current_iteration % len(self.perlin_3d_noise)
        # follow()
        self.particles_df['x_index'] = ((self.particles_df['pos_x'] / self.noise_divider) - 1).astype(int)
        self.particles_df['y_index'] = ((self.particles_df['pos_y'] / self.noise_divider) - 1).astype(int)
        self.particles_df['force_x'] = self.precomputed_vectors_dx[time_iteration_index][self.particles_df['x_index'], self.particles_df['y_index']] * self.force_mag
        self.particles_df['force_y'] = self.precomputed_vectors_dy[time_iteration_index][self.particles_df['x_index'], self.particles_df['y_index']] * self.force_mag
        self.particles_df['angle'] = self.precomputed_vectors_angles[time_iteration_index][self.particles_df['x_index'], self.particles_df['y_index']]
        # apply_force()
        self.particles_df['acc_x'] += self.particles_df['force_x']
        self.particles_df['acc_y'] += self.particles_df['force_y']

        # update()
        if self.enable_acc:
            # --> Calculate friction
            self.particles_df['friction_x'] = self.friction_mag * self.particles_df['vel_x']
            self.particles_df['friction_y'] = self.friction_mag * self.particles_df['vel_y']
            # --> Set velocity
            self.particles_df['vel_x'] += self.particles_df['acc_x'] - self.particles_df['friction_x']
            self.particles_df['vel_y'] += self.particles_df['acc_y'] - self.particles_df['friction_y']
            # --> Clip to max velocity
            if self.max_vel != 0:
                self.particles_df['vel_x'] = self.particles_df['vel_x'].clip(upper=self.max_vel, lower=-self.max_vel)
                self.particles_df['vel_y'] = self.particles_df['vel_y'].clip(upper=self.max_vel, lower=-self.max_vel)
            # --> Clear acceleration
            self.particles_df['acc_x'] = 0
            self.particles_df['acc_y'] = 0
            # --> Move in x and y based on velocity
            self.particles_df['pos_x'] += self.particles_df['vel_x']
            self.particles_df['pos_y'] += self.particles_df['vel_y']
        else:
            # --> Constant step size
            self.particles_df['pos_x'] += self.step_size * np.cos(np.radians(self.particles_df['angle']))
            self.particles_df['pos_y'] += self.step_size * np.sin(np.radians(self.particles_df['angle']))
        # --> Check boundaries
        boundary_cond_x_pos = self.particles_df['pos_x'] > self.image_width
        self.particles_df.loc[boundary_cond_x_pos, 'pos_x'] = 0
        self.particles_df.loc[boundary_cond_x_pos, 'previous_pos_x'] = self.particles_df['pos_x']
        self.particles_df.loc[boundary_cond_x_pos, 'previous_pos_y'] = self.particles_df['pos_y']
        boundary_cond_x_neg = self.particles_df['pos_x'] < 0
        self.particles_df.loc[boundary_cond_x_neg, 'pos_x'] = self.image_width
        self.particles_df.loc[boundary_cond_x_neg, 'previous_pos_x'] = self.particles_df['pos_x']
        self.particles_df.loc[boundary_cond_x_neg, 'previous_pos_y'] = self.particles_df['pos_y']
        boundary_cond_y_pos = self.particles_df['pos_y'] > self.image_height
        self.particles_df.loc[boundary_cond_y_pos, 'pos_y'] = 0
        self.particles_df.loc[boundary_cond_y_pos, 'previous_pos_x'] = self.particles_df['pos_x']
        self.particles_df.loc[boundary_cond_y_pos, 'previous_pos_y'] = self.particles_df['pos_y']
        boundary_cond_y_neg = self.particles_df['pos_y'] < 0
        self.particles_df.loc[boundary_cond_y_neg, 'pos_y'] = self.image_height
        self.particles_df.loc[boundary_cond_y_neg, 'previous_pos_x'] = self.particles_df['pos_x']
        self.particles_df.loc[boundary_cond_y_neg, 'previous_pos_y'] = self.particles_df['pos_y']
        # --> Update line length
        self.particles_df['line_length'] += np.sqrt((self.particles_df['previous_pos_x'] - self.particles_df['pos_x']) ** 2 + (self.particles_df['previous_pos_y'] - self.particles_df['pos_y']) ** 2)
        # print(self.particles_df)

    def draw_particles(self, painter):
        # NOTE: Do not use 'df.iterrows()', it is too slow
        pos_x = self.particles_df['pos_x'].to_numpy()
        pos_y = self.particles_df['pos_y'].to_numpy()
        previous_pos_x = self.particles_df['previous_pos_x'].to_numpy()
        previous_pos_y = self.particles_df['previous_pos_y'].to_numpy()
        line_length = self.particles_df['line_length'].to_numpy()
        angle = self.particles_df['angle'].to_numpy()
        hue_value = self.particles_df['hue_value'].to_numpy()

        for n in range(self.num_of_particles):
            if self.stop_draw_after_max_length and line_length[n] > self.max_line_length:
                continue
            # Select color
            color_to_use = QtGui.QColor('white')
            if self.particle_draw_color_style == self.DrawColorStyle.WHITE:
                color_to_use = QtGui.QColor(255, 255, 255, self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.GRAYSCALE:
                color_to_use.setHsv(0, 0, random.randint(127, 255), self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.HSV_ANGLE:
                color_to_use.setHsv(int(angle[n]), random.randint(127, 255), random.randint(127, 255), self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.HUE_CHANGING:
                color_to_use.setHsv(int(hue_value[n]), 255, random.randint(127, 255), self.alpha_value)
                hue_value[n] += 1
            elif self.particle_draw_color_style == self.DrawColorStyle.HUE_POS:
                hue = int(self.base_hue + pos_x[n] + pos_y[n]) % 360
                color_to_use.setHsv(hue, random.randint(127, 255), 255, self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.HUE_Y_POS_SAT_LENGTH:
                sat = int(127 * (self.max_line_length - line_length[n]) / self.max_line_length) % 255
                hue = int((self.base_hue + 360 * (self.image_height - pos_y[n]) / self.image_height) % 360)
                color_to_use.setHsv(hue, sat, 255, self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.HUE_Y_POS:
                hue = int((self.base_hue + 360 * (self.image_height - pos_y[n]) / self.image_height) % 360)
                color_to_use.setHsv(hue, random.randint(127, 255), 255, self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.HUE_LENGTH:
                hue = int((self.base_hue + 360 * line_length[n] * 0.00001) % 360)
                color_to_use.setHsv(hue, random.randint(127, 255), 255, self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.HUE_SAT_LENGTH:
                sat = int(127 + line_length[n] * 0.00001) % 255
                hue = int((self.base_hue + 360 * line_length[n] * 0.00001) % 360)
                color_to_use.setHsv(hue, sat, 255, self.alpha_value)
            elif self.particle_draw_color_style == self.DrawColorStyle.HUE_FIXED:
                hue = int(self.base_hue) % 360
                color_to_use.setHsv(hue, random.randint(127, 255), 255, self.alpha_value)
            # Select draw type
            if self.particle_draw_type == self.DrawType.POINT:
                painter.setPen(QtGui.QPen(color_to_use, self.particle_draw_size))
                painter.drawPoint(int(pos_x[n]), int(pos_y[n]))
            elif self.particle_draw_type == self.DrawType.ELLIPSE:
                painter.setPen(QtGui.QPen(color_to_use, self.particle_draw_size))
                painter.drawEllipse(int(pos_x[n]), int(pos_y[n]), self.particle_draw_size, self.particle_draw_size)
            elif self.particle_draw_type == self.DrawType.LINE:
                painter.setPen(QtGui.QPen(color_to_use, self.particle_draw_size))
                painter.drawLine(int(pos_x[n]), int(pos_y[n]), int(previous_pos_x[n]), int(previous_pos_y[n]))
        # --> Store previous
        self.particles_df['previous_pos_x'] = self.particles_df['pos_x']
        self.particles_df['previous_pos_y'] = self.particles_df['pos_y']

    def precompute_vectors(self):
        for n in range(len(self.perlin_3d_noise)):
            vectors, dx, dy, angles = self.get_vector_for_time_index(n)
            self.precomputed_vectors.append(vectors)
            self.precomputed_vectors_dx.append(dx)
            self.precomputed_vectors_dy.append(dy)
            self.precomputed_vectors_angles.append(angles)

    def get_vector_for_time_index(self, n):
        # Get vectors from the Perlin noise
        noise = self.perlin_3d_noise[n]
        noise_normalized = (noise - noise.min()) / (noise.max() - noise.min())
        vectors = []
        dx = []
        dy = []
        angles = []
        angled_line = QLineF()
        for x in range(self.noise_width):
            for y in range(self.noise_height):
                angle = 360 * noise_normalized[x, y] * self.angle_multiplier
                pos_x = x * self.noise_divider + int(self.noise_divider / 2)
                pos_y = y * self.noise_divider + int(self.noise_divider / 2)
                angled_line.setP1(QPointF(pos_x, pos_y))
                angled_line.setAngle(angle)
                angled_line.setLength(self.noise_divider / 2)
                vectors.append(angled_line.unitVector())
                dx.append(angled_line.dx())
                dy.append(angled_line.dy())
                angles.append(angle)
        vectors = np.array(vectors).reshape((self.noise_width, self.noise_height))
        dx = np.array(dx).reshape((self.noise_width, self.noise_height))
        dy = np.array(dy).reshape((self.noise_width, self.noise_height))
        angles = np.array(angles).reshape((self.noise_width, self.noise_height))
        return vectors, dx, dy, angles

    def draw_update(self):
        painter = QtGui.QPainter(self.image)
        painter.setRenderHints(painter.Antialiasing)
        if self.clear_each_update:
            painter.fillRect(0, 0, self.image_width, self.image_height, QtGui.QBrush(Qt.black, Qt.SolidPattern))
        # Get vectors
        current_index = self.current_iteration % len(self.perlin_3d_noise)
        vectors = self.precomputed_vectors[current_index]
        # Draw vectors
        if self.draw_vectors:
            for y_vectors in vectors:
                for vector in y_vectors:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 127), 2))
                    original_length = vector.length()
                    vector.setLength(6)
                    painter.drawLine(vector)
                    vector.setLength(original_length)
        # Update particles
        # start = time.time()
        self.update_particles()
        # print(f'----> elapsed_time for update_particles(): {(time.time() - start)*1e3:.6f} ms')
        # start = time.time()
        self.draw_particles(painter)
        # print(f'----> elapsed_time for draw_particles(): {(time.time() - start)*1e3:.6f} ms')
        # End painter
        painter.end()
        # Update iteration counter
        self.current_iteration += 1
        return self.image

