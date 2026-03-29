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


def plot_subplots(x, y_list, title, xlabel, ylabel, filenamesave, dt_values):
    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    s = 0

    for i in range(2):
        for j in range(2):
            ax[i][j].plot(x, y_list[s], linewidth=1)
            ax[i][j].set_title(f"Dt = {dt_values[s]}", fontsize=14)
            ax[i][j].grid(True)
            s += 1

    fig.supxlabel(xlabel, fontsize=14)
    fig.supylabel(ylabel, fontsize=14)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(filenamesave, dpi=600)
    plt.show()


if __name__ == "__main__":
    # Варіант 9
    n = 500
    Fs = 1000
    F_max = 19
    F_filter = 26
    Dt_values = [2, 4, 8, 16]

    if not os.path.exists("figures"):
        os.makedirs("figures")

    # ---------------- ПР2: генерація сигналу ----------------
    time_values, filtered_signal = generate_signal(n, F_max, Fs)

    plot_graph(
        time_values,
        filtered_signal,
        f"Сигнал з максимальною частотою F_max = {F_max} Гц",
        "Час (секунди)",
        "Амплітуда сигналу",
        f"./figures/Сигнал_Fmax_{F_max}Гц.png"
    )

    spectrum = fft.fft(filtered_signal)
    spectrum_abs = np.abs(fft.fftshift(spectrum))
    freq = fft.fftfreq(n, d=1 / Fs)
    freq_shifted = fft.fftshift(freq)

    plot_graph(
        freq_shifted,
        spectrum_abs,
        f"Спектр сигналу з максимальною частотою F_max = {F_max} Гц",
        "Частота (Гц)",
        "Амплітуда спектру",
        f"./figures/Спектр_Fmax_{F_max}Гц.png"
    )

    # ---------------- ПР3 ----------------
    discrete_signals = []
    discrete_spectrums = []
    restored_signals = []
    variances = []
    snr_ratios = []

    for Dt in Dt_values:
        # Дискретизація сигналу
        discrete_signal = np.zeros(n)

        for i in range(0, round(n / Dt)):
            index = i * Dt
            if index < n:
                discrete_signal[index] = filtered_signal[index]

        discrete_signals.append(discrete_signal.tolist())

        # Спектр дискретизованого сигналу
        discrete_spectrum = np.abs(fft.fftshift(fft.fft(discrete_signal)))
        discrete_spectrums.append(discrete_spectrum.tolist())

        # Відновлення сигналу через ФНЧ
        w_restore = F_filter / (Fs / 2)
        restore_filter = signal.butter(3, w_restore, 'low', output='sos')
        restored_signal = signal.sosfiltfilt(restore_filter, discrete_signal)
        restored_signals.append(restored_signal.tolist())

        # Дисперсія та співвідношення сигнал-шум
        E1 = restored_signal - filtered_signal
        variance_error = np.var(E1)
        snr = np.var(filtered_signal) / variance_error

        variances.append(variance_error)
        snr_ratios.append(snr)

    # Графік дискретизованих сигналів
    plot_subplots(
        time_values,
        discrete_signals,
        "Сигнали з кроком дискретизації Dt = (2, 4, 8, 16)",
        "Час (секунди)",
        "Амплітуда сигналу",
        "./figures/Дискретизовані сигнали.png",
        Dt_values
    )

    # Графік спектрів дискретизованих сигналів
    plot_subplots(
        freq_shifted,
        discrete_spectrums,
        "Спектри сигналів з кроком дискретизації Dt = (2, 4, 8, 16)",
        "Частота (Гц)",
        "Амплітуда спектру",
        "./figures/Спектри дискретизованих сигналів.png",
        Dt_values
    )

    # Графік відновлених сигналів
    plot_subplots(
        time_values,
        restored_signals,
        "Відновлені аналогові сигнали з кроком дискретизації Dt = (2, 4, 8, 16)",
        "Час (секунди)",
        "Амплітуда сигналу",
        "./figures/Відновлені сигнали.png",
        Dt_values
    )

    # Графік дисперсії
    plot_graph(
        Dt_values,
        variances,
        "Залежність дисперсії від кроку дискретизації",
        "Крок дискретизації",
        "Дисперсія",
        "./figures/Дисперсія від кроку дискретизації.png"
    )

    # Графік співвідношення сигнал-шум
    plot_graph(
        Dt_values,
        snr_ratios,
        "Залежність співвідношення сигнал-шум від кроку дискретизації",
        "Крок дискретизації",
        "ССШ",
        "./figures/ССШ від кроку дискретизації.png"
    )