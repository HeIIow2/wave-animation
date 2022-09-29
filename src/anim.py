from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np

class Frequency:
    def __init__(self, amplitude: np.float16, frequenz: np.float16, phasenverschiebung_rad: np.float16) -> None:
        self.amplitude = amplitude
        self.frequenz = frequenz
        self.schwingungsdauer = np.divide(1, frequenz)
        self.phasenverschiebung_rad = phasenverschiebung_rad
        # (rad / 2pi) * schwingungsdauer
        self.t_difference = np.multiply(self.schwingungsdauer, np.divide(self.phasenverschiebung_rad, 2*np.pi))

        self.omega = 2 * np.pi * self.frequenz
        self.max_geschwindigkeit = np.multiply(self.amplitude, self.omega)
        self.max_beschleunigung = np.multiply(self.amplitude, np.square(self.omega))

    def get_s_from_t(self, t: np.float16):
        t = t - self.t_difference
        return np.multiply(self.amplitude, np.sin(np.multiply(self.omega, t)))
    
    def get_v_from_t(self, t: np.float16):
        t = t - self.t_difference
        return np.multiply(self.max_geschwindigkeit, np.cos(np.multiply(self.omega, t)))

    def get_a_from_t(self, t: np.float16):
        t = t - self.t_difference
        return np.multiply(-1, self.max_beschleunigung, np.sin(np.multiply(self.omega, t)))

    def get_s_during(self, iterable):
        return [self.get_s_from_t(i) for i in iterable]


class Oszillator(Frequency):
    def __init__(self, frequenz: np.float16, amplitude: np.float16, phasenverschiebung_rad: np.float16) -> None:
        super().__init__(amplitude, frequenz, phasenverschiebung_rad)


class Welle(Frequency):
    def __init__(self, amplitude: np.float16, wellenlaenge: np.float16, oszyllator_frequenz: np.float16, phases: int, dots_per_phase: int) -> None:
        self.wellenlaenge = wellenlaenge
        super().__init__(amplitude, np.divide(1, self.wellenlaenge), 0)

        self.oszyllator_frequenz = oszyllator_frequenz
        self.ausbreitungsgeschwindigkeit = np.multiply(self.wellenlaenge, self.oszyllator_frequenz)

        self.oszyllators = np.array([], Oszillator)
        self.oszyllator_positions = np.array([], np.float16)
        self.phases = phases
        self.dots_per_phase = dots_per_phase
        for phase in range(self.phases):
            current_phase_difference = phase * self.wellenlaenge
            for i in range(self.dots_per_phase):
                phase_difference = np.multiply(i, np.divide(2*np.pi, self.dots_per_phase))
                distance_from_start = current_phase_difference + np.multiply(i, np.divide(self.wellenlaenge, self.dots_per_phase))
                self.oszyllator_positions = np.append(self.oszyllator_positions, distance_from_start)
                self.oszyllators = np.append(self.oszyllators, Oszillator(self.oszyllator_frequenz, self.amplitude, phase_difference))


        # lamda ist T für wellen
        """
        c = s / t = lamda / T mit f = 1 / T gilt c: c = lamda * f
        """
    
    def plot_at_time(self, t_from: np.float16, t_to: np.float16, steps):
        self.x_values = self.oszyllator_positions
        self.z_values = np.arange(t_from, t_to, steps)
        self.y_values = np.array([x.get_s_during(self.z_values) for x in self.oszyllators], np.float16)

        self.z_values, self.x_values = np.meshgrid(self.z_values, self.x_values)

        print(self.x_values, self.x_values.shape)
        print(self.y_values, self.y_values.shape)
        print(self.z_values, self.z_values.shape)
        



        # Creating a wireframe plot with the points
        # x1,y1,z1 along with the plot line as red
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(self.x_values,self.z_values,  self.y_values, color="red")
        ax.set_yscale(0.5, 'linear')
        plt.show()



if __name__ == "__main__":
    print("Hello World")
    welle = Welle(0.5, 4, 8, 4, 10)
    welle.plot_at_time(0, 10, 0.1)
