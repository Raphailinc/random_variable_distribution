# Random Variable Distribution Toolkit

Генерация и анализ распределения случайной величины вида `X = asin(sqrt(U/2))`, где `U ~ U(0, 1)`. Есть два генератора равномерных чисел: стандартный NumPy и детерминированный middle-square. Поддерживается расчёт хи-квадрат статистики и (при наличии SciPy) p-value.

## Быстрый старт
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m random_variable_distribution -n 5000 -b 14 -g numpy --seed 42
```

Пример вывода:
```
Generator: numpy
Samples: 5000, bins: 14
Mean: 0.741262, Std: 0.402934
Chi-square: 10.8275, p-value: 0.6250
First 8 samples: 0.886074, 0.837400, 0.467790, 0.606330, 0.174409, 0.099871, 0.126145, 0.194592
Bin counts (observed): 104, 191, ...
Bin counts (expected): 39.4, 116.8, ...
```

## Основные функции
- `generate_uniform(size, method="numpy", seed=None, k=8)`: равномерные числа (`method` — `numpy` или `middle-square`).
- `transform_sample(uniforms)`: применяет обратную функцию `asin(sqrt(u/2))`.
- `expected_probabilities(num_bins)`: теоретические вероятности по интервалам на `[0, π/4]`.
- `analyze_distribution(...)`: полный цикл генерации, биннинга и расчёта хи-квадрат; возвращает `SampleSummary` с chi-square, p-value, средним и σ.

Запуск как модуля: `python -m random_variable_distribution --help`.

## Тесты
```bash
pytest
```

## Идеи развития
- Добавить больше генераторов (LCG, криптографический).
- Вывести результаты в CSV/JSON.
- CLI-опция для сохранения гистограмм. 
