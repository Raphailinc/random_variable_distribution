import numpy as np
import math

def make_array(N):
    y0 = 0.84616823
    k = 8
    sequence = [y0]  # массив псевдослучайных чисел
    for i in range(N - 1):  # метод середины квадратов
        val = sequence[-1] ** 2
        fractional, _ = math.modf(val * 10 ** (k // 2))
        next_val = int(fractional * 10 ** k) * 10 ** (-k)
        sequence.append(next_val)
    return sequence

# по новой формуле заполняем массив
def apply_inverse_function(sequence):
    return [math.asin(math.sqrt(y / 2)) for y in sequence]

def calc_bin_counts(new_sequence, num_bins):
    print("Подсчет количества попаданий в каждом интервале.")
    print("Количество интервалов:", num_bins)
    bin_counts, _ = np.histogram(new_sequence, bins=np.linspace(0, math.pi / 4, num_bins + 1))
    print("Распределение по интервалам:", bin_counts)
    return bin_counts

# Теоритическая вероятность попадания случайной величины в i-ый интервал.
def calc_expected_probabilities(num_bins):
    delta = (math.pi / 4) / num_bins
    probabilities = []
    for i in range(1, num_bins + 1):
        x_i = i * delta
        x_i_minus_1 = (i - 1) * delta
        p_i = 2 * (math.sin(x_i) ** 2) - 2 * (math.sin(x_i_minus_1) ** 2)
        probabilities.append(p_i)
    return probabilities

def calc_X2(new_sequence, bin_counts, probabilities):
    n = len(new_sequence)
    expected_count = [n * probabilities[i] for i in range(len(bin_counts))]
    print("Вычисление X^2.")
    chi_square_stat = np.sum((bin_counts - expected_count) ** 2 / expected_count)
    return chi_square_stat

N = 5000  # число элементов
num_bins = 14  # кол-во интервалов
sequence = make_array(N)
new_sequence = apply_inverse_function(sequence)
bin_counts = calc_bin_counts(new_sequence, num_bins)
expected_probabilities = calc_expected_probabilities(num_bins)
chi_square_stat = calc_X2(new_sequence, bin_counts, expected_probabilities)  # хи-квадрат

print("X2:", chi_square_stat)
print(new_sequence)