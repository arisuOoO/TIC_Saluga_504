import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
import os


def generate_signal(n, f_max, fs):
    random_signal = np.random.normal(0, 10, n)
    time_values = np.arange(n) / fs

    w = f_max / (fs / 2)
    sos = signal.butter(3, w, 'low', output='sos')
    filtered_signal = signal.sosfiltfilt(sos, random_signal)

    return time_values, filtered_signal


def plot_graph(x, y, title, xlabel, ylabel, filename):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.plot(x, y, linewidth=1)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(filename, dpi=600)
    plt.show()


if __name__ == "__main__":
    # Вариант 9
    n = 500
    Fs = 1000
    F_max = 19

    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Генерация и фильтрация сигнала
    time_values, filtered_signal = generate_signal(n, F_max, Fs)

    # График сигнала
    plot_graph(
        time_values,
        filtered_signal,
        f"Сигнал з максимальною частотою F_max = {F_max} Гц",
        "Час (секунди)",
        "Амплітуда сигналу",
        f"./figures/Сигнал_Fmax_{F_max}Гц.png"
    )

    # Расчет спектра
    spectrum = fft.fft(filtered_signal)
    spectrum_abs = np.abs(fft.fftshift(spectrum))

    # Частотная ось
    freq = fft.fftfreq(n, d=1 / Fs)
    freq_shifted = fft.fftshift(freq)

    # График спектра
    plot_graph(
        freq_shifted,
        spectrum_abs,
        f"Спектр сигналу з максимальною частотою F_max = {F_max} Гц",
        "Частота (Гц)",
        "Амплітуда спектру",
        f"./figures/Спектр_Fmax_{F_max}Гц.png"
    )