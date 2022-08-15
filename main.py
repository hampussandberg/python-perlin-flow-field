import sys
import time
from enum import Enum
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from PIL import ImageQt

from perlin_flow_field_animation import PerlinFlowFieldAnimation

width = 1280
height = 480
update_delay_ms = 100


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # Load the UI file
        uic.loadUi('main.ui', self)
        self.setWindowTitle('python-flow-field')
        # Set size of pixmap label
        self.animation_label.setFixedWidth(width)
        self.animation_label.setFixedHeight(height)

        # Set label pixmap initially to black
        canvas = QtGui.QPixmap(self.animation_label.width(), self.animation_label.height())
        canvas.fill(QtGui.QColor('black'))
        self.animation_label.setPixmap(canvas)
        print(f'{self.animation_label.width()=}, {self.animation_label.height()=}')

        # Init
        self.animation = None
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(False)
        self.timer.setInterval(update_delay_ms)
        self.timer.timeout.connect(self.update_draw)

        # Add content to UI elements
        for item in PerlinFlowFieldAnimation.DrawType:
            self.particle_draw_type.addItem(item.name)
        self.particle_draw_type.setCurrentIndex(PerlinFlowFieldAnimation.DrawType.LINE.value)
        for item in PerlinFlowFieldAnimation.DrawColorStyle:
            self.particle_draw_color_style.addItem(item.name)
        self.particle_draw_color_style.setCurrentIndex(PerlinFlowFieldAnimation.DrawColorStyle.HUE_Y_POS_SAT_LENGTH.value)
        for key in PerlinFlowFieldAnimation.preconfigs:
            self.preconfig_comboBox.addItem(key)

        # Connect UI elements
        self.start_pushButton.clicked.connect(self.start)
        self.stop_pushButton.clicked.connect(self.stop)
        self.clear_pushButton.clicked.connect(self.clear)

        self.force_mag.valueChanged.connect(self.update_particle_properties)
        self.friction_mag.valueChanged.connect(self.update_particle_properties)
        self.max_vel.valueChanged.connect(self.update_particle_properties)
        self.particle_draw_size.valueChanged.connect(self.update_particle_properties)
        self.alpha_value.valueChanged.connect(self.update_particle_properties)
        self.preconfig_comboBox.currentIndexChanged.connect(self.change_preconfig)
        self.preconfig_comboBox.setCurrentIndex(0)
        self.change_preconfig(0)

        self.add = QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self)
        self.add.activated.connect(self.save_image)
        self.save_pushButton.clicked.connect(self.save_image)

        self.ui_elements_to_toggle = [
            self.num_of_particles,
            self.enable_acc,
            self.particle_draw_type,
            self.particle_draw_color_style,
            self.stop_draw_after_max_length,
            self.max_line_length,
            self.preconfig_comboBox,
            self.noise_divider,
            self.angle_multiplier
        ]

    def save_image(self):
        save_path = f'image_{int(time.time())}.png'
        image = ImageQt.fromqpixmap(self.animation_label.pixmap())
        image.save(save_path)
        print(f'--> Saved image to {save_path}')

    def change_preconfig(self, index):
        config_key = self.preconfig_comboBox.currentText()
        if config_key in PerlinFlowFieldAnimation.preconfigs:
            # print(f'{PerlinFlowFieldAnimation.preconfigs[config_key]=}')
            for key, value in PerlinFlowFieldAnimation.preconfigs[config_key].items():
                # print(key, '->', value)
                try:
                    ui_element = getattr(self, key)
                    # print(type(ui_element))
                    if isinstance(ui_element, QtWidgets.QCheckBox):
                        ui_element.setChecked(value)
                    elif isinstance(ui_element, QtWidgets.QDoubleSpinBox) or isinstance(ui_element, QtWidgets.QSpinBox):
                        ui_element.setValue(value)
                    elif isinstance(ui_element, QtWidgets.QComboBox):
                        if isinstance(value, Enum):
                            ui_element.setCurrentIndex(value.value)
                        else:
                            ui_element.setCurrentIndex(value)
                except AttributeError as error:
                    print(error)

    def start(self):
        # First clear
        self.clear()
        # Create animation
        self.animation = PerlinFlowFieldAnimation(
            self.animation_label.width(), self.animation_label.height(),
            noise_divider=self.noise_divider.value())

        # Modify setting
        self.animation.num_of_particles = self.num_of_particles.value()
        self.animation.enable_acc = self.enable_acc.isChecked()
        self.animation.force_mag = self.force_mag.value()
        self.animation.friction_mag = self.friction_mag.value()
        self.animation.max_vel = self.max_vel.value()
        self.animation.particle_draw_type = PerlinFlowFieldAnimation.DrawType(self.particle_draw_type.currentIndex())
        self.animation.particle_draw_color_style = PerlinFlowFieldAnimation.DrawColorStyle(self.particle_draw_color_style.currentIndex())
        self.animation.particle_draw_size = self.particle_draw_size.value()
        self.animation.stop_draw_after_max_length = self.stop_draw_after_max_length.isChecked()
        self.animation.max_line_length = self.max_line_length.value()
        self.animation.alpha_value = self.alpha_value.value()
        self.animation.angle_multiplier = self.angle_multiplier.value()
        # Init particles and start time
        self.animation.init_particles()
        self.timer.start()
        # Disable UI elements
        for ui_element in self.ui_elements_to_toggle:
            ui_element.setEnabled(False)

    def stop(self):
        self.timer.stop()
        self.animation = None
        # Enable UI elements
        for ui_element in self.ui_elements_to_toggle:
            ui_element.setEnabled(True)

    def clear(self):
        self.animation_label.pixmap().fill(QtGui.QColor('black'))
        self.update()

    def update_particle_properties(self, _):
        if self.animation:
            self.animation.force_mag = self.force_mag.value()
            self.animation.friction_mag = self.friction_mag.value()
            self.animation.max_vel = self.max_vel.value()
            self.animation.particle_draw_size = self.particle_draw_size.value()
            self.animation.alpha_value = self.alpha_value.value()

    def update_draw(self):
        # Settings
        self.animation.draw_vectors = self.draw_vectors.isChecked()
        self.animation.clear_each_update = self.clear_each_update.isChecked()
        # Update
        # start = time.time()
        image = self.animation.draw_update()
        self.animation_label.setPixmap(QtGui.QPixmap.fromImage(image))
        self.current_iteration_label.setText(f'Current Iteration: {self.animation.current_iteration}')
        self.update()
        # print(f'--> elapsed_time for update_draw(): {(time.time() - start)*1e3:.6f} ms')


def main():
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    # print(f'Using AA_EnableHighDpiScaling > {QtWidgets.QApplication.testAttribute(Qt.AA_EnableHighDpiScaling)}')
    # print(f'Using AA_UseHighDpiPixmaps    > {QtWidgets.QApplication.testAttribute(Qt.AA_UseHighDpiPixmaps)}')

    window = MainWindow()
    window.show()
    app.exec_()
    sys.exit(app.exec_())
    return 0

if __name__ == "__main__":
    main()
