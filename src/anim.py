import matplotlib
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Qt5Agg")


class Frequency:
    def __init__(self) -> None:
        # http://college-physics.com/book/oscillations/harmonic-oscillator/
        self._amplitude = None
        self._frequency = None
        self._period = None
        self._angular_frequency = None
        self._phase_angle = None  # radial system
        self._phase_offset = None

        self.max_speed = None
        self.max_acceleration = None

        self.set_phase_offset(np.float(0))

    def set_amplitude(self, amplitude: np.float, traceback=None):
        if traceback is None:
            traceback = []
        traceback.append("amplitude")

        self._amplitude = amplitude

    def set_max_values(self, _: np.float, traceback=None):
        if traceback is None:
            traceback = []
        traceback.append("max values")

        if self.amplitude is not None and self.angular_frequency is not None:
            self.max_speed = np.multiply(self.amplitude, self.angular_frequency)
            self.max_acceleration = np.multiply(self.amplitude, np.square(self.angular_frequency))

    def set_period(self, period: np.float, traceback=None):
        if traceback is None:
            traceback = []
        traceback.append("period")

        self._period = period
        if "frequency" not in traceback:
            self.set_frequency(np.divide(1, period), traceback)
        if "angular frequency" not in traceback:
            self.set_angular_frequency(np.divide(2 * np.pi, period), traceback)

    def set_frequency(self, frequency: np.float, traceback=None):
        if traceback is None:
            traceback = []
        traceback.append("frequency")

        self._frequency = frequency
        if "period" not in traceback:
            self.set_period(np.divide(1, frequency), traceback)
        if "angular frequency" not in traceback:
            self.set_angular_frequency(np.float(2 * np.pi * frequency), traceback)

    def set_angular_frequency(self, angular_frequency: np.float, traceback=None):
        if traceback is None:
            traceback = []
        traceback.append("angular frequency")

        self._angular_frequency = angular_frequency
        if "frequency" not in traceback:
            self.set_frequency(np.divide(angular_frequency, 2 * np.pi), traceback)
        if "period" not in traceback:
            self.set_period(np.divide(2 * np.pi, angular_frequency))

    def set_phase_angle(self, phase_angle: np.float, traceback=None):
        if traceback is None:
            traceback = []
        traceback.append("phase angle")
        print(f"angle:  {phase_angle}")

        self._phase_angle = phase_angle
        if "phase offset" not in traceback:
            if self.period is not None:
                self.set_phase_offset(np.interp(phase_angle, (0, 2 * np.pi), (0, self.period)), traceback)
            elif self.frequency is not None:
                self.set_phase_offset(np.interp(phase_angle, (0, 2 * np.pi), (0, np.divide(1, self.frequency))), traceback)

    def set_phase_offset(self, phase_offset: np.float, traceback=None):
        if traceback is None:
            traceback = []
        traceback.append("phase offset")

        self._phase_offset = phase_offset
        if "phase angle" not in traceback and self.period is not None:
            if self.period is not None:
                self.set_phase_angle(np.interp(phase_offset, (0, self.period), (0, 2 * np.pi)), traceback)
            elif self.frequency is not None:
                self.set_phase_angle(np.interp(phase_offset, (0, np.divide(1, self.frequency)), (0, 2 * np.pi)), traceback)

    def get_s_from_t(self, t: np.float16):
        t = t - self._phase_offset
        return np.multiply(self.amplitude, np.sin(np.multiply(self.angular_frequency, t)))

    def get_v_from_t(self, t: np.float16):
        t = t - self._phase_offset
        return np.multiply(self.max_speed, np.cos(np.multiply(self.angular_frequency, t)))

    def get_a_from_t(self, t: np.float16):
        t = t - self._phase_offset
        return np.multiply(-1, self.max_acceleration, np.sin(np.multiply(self.angular_frequency, t)))

    def get_s_during(self, iterable):
        return [self.get_s_from_t(i) for i in iterable]

    amplitude = property(fset=(lambda self, amplitude: self.set_amplitude(amplitude)),
                         fget=lambda self: self._amplitude)
    frequency = property(fset=(lambda self, frequency: self.set_frequency(frequency)),
                         fget=lambda self: self._frequency)
    period = property(fset=(lambda self, period: self.set_period(period)), fget=lambda self: self._period)
    wavelength = property(fset=(lambda self, period: self.set_period(period)), fget=lambda self: self._period)
    angular_frequency = property(fset=(lambda self, frequency: self.set_angular_frequency(frequency)),
                                 fget=lambda self: self._angular_frequency)

    def __str__(self):
        return f"angular frequency: {self.angular_frequency}\namplitude:{self.amplitude}\nphase angle: {self._phase_angle}"


class Oszillator(Frequency):
    def __init__(self) -> None:
        super().__init__()


class Welle(Frequency):
    def __init__(self, phases: int, offsets: np.float, amplitude: np.float, wavelength: np.float, oscillator_frequency: np.float) -> None:
        super().__init__()

        # wave specific stuff
        self.oscillator_frequency = oscillator_frequency
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.ausbreitungsgeschwindigkeit = np.multiply(self.wavelength, self.oscillator_frequency)

        self.oscillators = np.array([], Oszillator)
        self.oscillatory_positions = np.array([], np.float16)
        self.phases = phases
        self.offsets = offsets

        angles_reached = np.float(0)
        while int(angles_reached / 360) < phases:
            angles_reached += self.offsets

            new_oscillator = Oszillator()
            new_oscillator.set_amplitude(self.amplitude)
            new_oscillator.set_frequency(self.oscillator_frequency)
            new_oscillator.set_phase_angle(np.deg2rad(angles_reached))

            self.oscillatory_positions = np.append(self.oscillatory_positions, new_oscillator._phase_offset)
            self.oscillators = np.append(self.oscillators, [new_oscillator])


    def plot_at_time(self, t_from: np.float, t_to: np.float, steps):
        x_values = self.oscillatory_positions
        z_values = np.arange(t_from, t_to, steps)

        y_values = np.array([x.get_s_during(z_values) for x in self.oscillators], np.float16)

        z_values, x_values = np.meshgrid(z_values, x_values)

        #print(x_values, x_values.shape)
        #print(y_values, y_values.shape)
        #print(z_values, z_values.shape)

        # Creating a wireframe plot with the points
        # x1,y1,z1 along with the plot line as red
        fig = plt.figure()

        # Make a horizontal slider to control the frequency.
        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        freq_slider = Slider(
            ax=axfreq,
            label='Frequency [Hz]',
            valmin=0.1,
            valmax=30,
            valinit=3,
        )

        freq_slider.on_changed(self.update)

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(x_values, z_values, y_values)
        plt.show()

    def update(self, val):
        print(val)


if __name__ == "__main__":
    print("Hello World")
    welle = Welle(1, 20, 1, 1, 1)
    welle.plot_at_time(0, 2, 0.05)
