import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft


def generate_signal(n, f_max, fs):
    random_signal = np.random.normal(0, 10, n)
    time_values = np.arange(n) / fs

    w = f_max / (fs / 2)
    sos = signal.butter(3, w, 'low', output='sos')
    filtered_signal = signal.sosfiltfilt(sos, random_signal)

    return time_values, filtered_signal


def calculate_spectrum(sig, fs):
    spectrum = fft.fft(sig)
    spectrum_abs = np.abs(fft.fftshift(spectrum))
    freq = fft.fftfreq(len(sig), d=1 / fs)
    freq_shifted = fft.fftshift(freq)
    return freq_shifted, spectrum_abs


def plot_graph(x, y, title, xlabel, ylabel, filename, linewidth=1, step=False):
    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))

    if step:
        ax.step(x, y, where='post', linewidth=linewidth)
    else:
        ax.plot(x, y, linewidth=linewidth)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(filename, dpi=600)
    plt.show()


def plot_subplots(x, y_list, subplot_titles, title, xlabel, ylabel, filename, step=False):
    fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
    s = 0

    for i in range(2):
        for j in range(2):
            if step:
                ax[i][j].step(x, y_list[s], where='post', linewidth=1)
            else:
                ax[i][j].plot(x, y_list[s], linewidth=1)

            ax[i][j].set_title(subplot_titles[s], fontsize=12)
            ax[i][j].grid(True)
            s += 1

    fig.supxlabel(xlabel, fontsize=14)
    fig.supylabel(ylabel, fontsize=14)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(filename, dpi=600)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # ---------------- Загальні параметри ----------------
    n = 500
    Fs = 1000
    F_max = 19          # Варіант 9 для ПР2/ПР4
    F_filter = 26       # Варіант 9 для ПР3
    Dt_values = [2, 4, 8, 16]
    M_values = [4, 16, 64, 256]

    if not os.path.exists("figures"):
        os.makedirs("figures")

    # ====================================================
    #                   ПРАКТИЧНА РОБОТА 2
    # ====================================================
    time_values, filtered_signal = generate_signal(n, F_max, Fs)

    plot_graph(
        time_values,
        filtered_signal,
        f"Сигнал з максимальною частотою F_max = {F_max} Гц",
        "Час (секунди)",
        "Амплітуда сигналу",
        f"./figures/Сигнал_Fmax_{F_max}Гц.png"
    )

    freq_shifted, spectrum_abs = calculate_spectrum(filtered_signal, Fs)

    plot_graph(
        freq_shifted,
        spectrum_abs,
        f"Спектр сигналу з максимальною частотою F_max = {F_max} Гц",
        "Частота (Гц)",
        "Амплітуда спектру",
        f"./figures/Спектр_Fmax_{F_max}Гц.png"
    )

    # ====================================================
    #                   ПРАКТИЧНА РОБОТА 3
    # ====================================================
    discrete_signals = []
    discrete_spectrums = []
    restored_signals = []
    variances_pr3 = []
    snr_ratios_pr3 = []

    w_restore = F_filter / (Fs / 2)
    restore_filter = signal.butter(3, w_restore, 'low', output='sos')

    for Dt in Dt_values:
        discrete_signal = np.zeros(n)

        for i in range(0, round(n / Dt)):
            index = i * Dt
            if index < n:
                discrete_signal[index] = filtered_signal[index]

        discrete_signals.append(discrete_signal.copy())

        _, discrete_spectrum = calculate_spectrum(discrete_signal, Fs)
        discrete_spectrums.append(discrete_spectrum.copy())

        restored_signal = signal.sosfiltfilt(restore_filter, discrete_signal)
        restored_signals.append(restored_signal.copy())

        E1 = restored_signal - filtered_signal
        variance_error = np.var(E1)
        snr_value = np.var(filtered_signal) / variance_error if variance_error != 0 else np.inf

        variances_pr3.append(variance_error)
        snr_ratios_pr3.append(snr_value)

    plot_subplots(
        time_values,
        discrete_signals,
        [f"Dt = {dt}" for dt in Dt_values],
        "Сигнали з кроком дискретизації Dt = (2, 4, 8, 16)",
        "Час (секунди)",
        "Амплітуда сигналу",
        "./figures/Дискретизовані сигнали.png"
    )

    plot_subplots(
        freq_shifted,
        discrete_spectrums,
        [f"Dt = {dt}" for dt in Dt_values],
        "Спектри сигналів з кроком дискретизації Dt = (2, 4, 8, 16)",
        "Частота (Гц)",
        "Амплітуда спектру",
        "./figures/Спектри дискретизованих сигналів.png"
    )

    plot_subplots(
        time_values,
        restored_signals,
        [f"Dt = {dt}" for dt in Dt_values],
        "Відновлені аналогові сигнали з кроком дискретизації Dt = (2, 4, 8, 16)",
        "Час (секунди)",
        "Амплітуда сигналу",
        "./figures/Відновлені сигнали.png"
    )

    plot_graph(
        Dt_values,
        variances_pr3,
        "Залежність дисперсії від кроку дискретизації",
        "Крок дискретизації",
        "Дисперсія",
        "./figures/Дисперсія від кроку дискретизації.png"
    )

    plot_graph(
        Dt_values,
        snr_ratios_pr3,
        "Залежність співвідношення сигнал-шум від кроку дискретизації",
        "Крок дискретизації",
        "ССШ",
        "./figures/ССШ від кроку дискретизації.png"
    )

    # ====================================================
    #                   ПРАКТИЧНА РОБОТА 4
    # ====================================================
    quantized_signals = []
    variances_pr4 = []
    snr_ratios_pr4 = []

    for M in M_values:
        bits = []

        delta = (np.max(filtered_signal) - np.min(filtered_signal)) / (M - 1)
        quantize_signal = delta * np.round(filtered_signal / delta)
        quantized_signals.append(quantize_signal.copy())

        quantize_levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + delta, delta)
        if len(quantize_levels) < M:
            quantize_levels = np.linspace(np.min(quantize_signal), np.max(quantize_signal), M)

        quantize_levels = np.round(quantize_levels[:M], 10)

        bit_count = int(np.log2(M))
        quantize_bit = [format(i, '0' + str(bit_count) + 'b') for i in range(M)]

        quantize_table = np.c_[quantize_levels, quantize_bit]

        # Таблиця квантування
        fig_table, ax_table = plt.subplots(figsize=(14 / 2.54, M / 2.54))
        table = ax_table.table(
            cellText=quantize_table,
            colLabels=['Значення сигналу', 'Кодова послідовність'],
            loc='center'
        )
        table.set_fontsize(14)
        table.scale(1, 2)
        ax_table.axis('off')
        fig_table.tight_layout()
        fig_table.savefig(f"./figures/Таблиця квантування для {M} рівнів.png", dpi=600)
        plt.show()

        # Кодування сигналу у біти
        level_to_bit = {level: bit for level, bit in zip(quantize_levels, quantize_bit)}

        for signal_value in quantize_signal:
            rounded_value = np.round(signal_value, 10)

            if rounded_value in level_to_bit:
                bits.append(level_to_bit[rounded_value])
            else:
                nearest_index = np.argmin(np.abs(quantize_levels - signal_value))
                bits.append(quantize_bit[nearest_index])

        bits = [int(item) for item in ''.join(bits)]

        # Графік бітової послідовності
        fig_bits, ax_bits = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        ax_bits.step(np.arange(0, len(bits)), bits, where='post', linewidth=0.1)
        ax_bits.set_xlabel('Біти', fontsize=14)
        ax_bits.set_ylabel('Амплітуда сигналу', fontsize=14)
        ax_bits.set_title(
            f'Кодова послідовність сигналу при кількості рівнів квантування {M}',
            fontsize=14
        )
        ax_bits.grid(True)
        fig_bits.tight_layout()
        fig_bits.savefig(f"./figures/Кодова послідовність для {M} рівнів.png", dpi=600)
        plt.show()

        # Дисперсія та ССШ
        error_signal = quantize_signal - filtered_signal
        variance_error = np.var(error_signal)
        snr_value = np.var(filtered_signal) / variance_error if variance_error != 0 else np.inf

        variances_pr4.append(variance_error)
        snr_ratios_pr4.append(snr_value)

    plot_subplots(
        time_values,
        quantized_signals,
        [f"M = {m}" for m in M_values],
        "Цифрові сигнали з рівнями квантування (4, 16, 64, 256)",
        "Час (секунди)",
        "Амплітуда сигналу",
        "./figures/Цифрові сигнали з різними рівнями квантування.png",
        step=True
    )

    plot_graph(
        M_values,
        variances_pr4,
        "Залежність дисперсії від кількості рівнів квантування",
        "Кількість рівнів квантування",
        "Дисперсія",
        "./figures/Дисперсія від кількості рівнів квантування.png"
    )

    plot_graph(
        M_values,
        snr_ratios_pr4,
        "Залежність співвідношення сигнал-шум від кількості рівнів квантування",
        "Кількість рівнів квантування",
        "ССШ",
        "./figures/ССШ від кількості рівнів квантування.png"
    )