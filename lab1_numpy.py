# -*- coding: utf-8 -*-
"""
Лабораторная работа "Продвинутый NumPy"
Ростелеком ИТ Школа, тема 1.
Все задачи (1.1 - 5.3) выполнены в одном файле, каждая помечена соответствующим комментарием.
"""

import numpy as np
import time
import os

np.random.seed(42)          # воспроизводимость результатов
OUT = "."  # папка с результатами


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# ЗАДАЧА 1. Числа с плавающей точкой
# =====================================================================
line("ЗАДАЧА 1.1 — массив float32 [0.1, 0.2, 0.3] и сумма")

a32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
s32 = a32.sum()
print(f"float32: sum = {s32!r}   (точное 0.6)")
print(f"разница с 0.6: {abs(s32 - 0.6):.3e}")

a64 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
s64 = a64.sum()
print(f"float64: sum = {s64!r}   (точное 0.6)")
print(f"разница с 0.6: {abs(s64 - 0.6):.3e}")
# Вывод: ни float32, ни float64 не представляют 0.1/0.2/0.3 точно,
# сумма оказывается чуть больше 0.6 — это ошибка округления.

line("ЗАДАЧА 1.2 — машинное эпсилон через np.finfo")

for dt in (np.float32, np.float64):
    fi = np.finfo(dt)
    print(f"{np.dtype(dt).name:>8}: eps = {fi.eps:.3e},  "
          f"min = {fi.min:.3e},  max = {fi.max:.3e},  precision = {fi.precision}")

line("ЗАДАЧА 1.3 — катастрофическая отмена (cancellation)")

# Точное квадратное уравнение: x^2 - 200000x + 1 = 0
# Малый корень x2 = 1/x1 ~ 5e-6. Формула x2 = (c/x1) устойчива,
# а "школьная" x2 = (-b - sqrt(D)) / (2a) при вычитании близких чисел
# теряет значащие разряды.
a, b, c = 1.0, -200000.0, 1.0
D = b**2 - 4*a*c
x1 = (-b + np.sqrt(D)) / (2*a)
x2_naive = (-b - np.sqrt(D)) / (2*a)   # катастрофическая отмена
x2_stable = c / (a * x1)               # устойчивая формула
print(f"x1 (большой корень)      = {x1:.12f}")
print(f"x2 наивная формула       = {x2_naive:.12e}   <- почти все разряды потеряны")
print(f"x2 устойчивая (c/(a*x1)) = {x2_stable:.12e}")
print(f"относительная ошибка наивной формулы: "
      f"{abs(x2_naive - x2_stable)/x2_stable:.3e}")

# Другой наглядный пример: вычитание близких float64
u = np.float64(1.0)
v = np.float64(1.0 + 2**-52)     # ближайшее число к 1.0
print(f"\n1.0 и ближайшее к нему float64 отличаются на {v - u:.3e}")
print("После вычитания из 53-х значащих бит остается ~1 — относительная "
      "погрешность результата ~100%.")

# =====================================================================
# ЗАДАЧА 2. Индексация, срезы и broadcasting
# =====================================================================
line("ЗАДАЧА 2.1 — базовые срезы массива A формы (4, 5)")

A = np.arange(20).reshape(4, 5)
print("A =\n", A)
print("вторая строка        A[1, :]   =", A[1, :])
print("столбец с индексом 3 A[:, 3]   =", A[:, 3])
print("строки 1-3, столбцы 2-4 A[1:4, 2:5] =\n", A[1:4, 2:5])

line("ЗАДАЧА 2.2 — fancy-индексация")

rows = [0, 2, 3]
cols = [1, 4]
print("A[np.ix_(rows, cols)] =\n", A[np.ix_(rows, cols)])
# Пояснение: без np.ix_ пара списков дала бы лишь пары (0,1),(2,4),(3,1)
print("A[rows][:, cols] дает то же самое:\n", A[rows][:, cols])

line("ЗАДАЧА 2.3 — логическая индексация: отрицательные -> 0")

A2 = A.astype(float) - 7.5      # теперь есть отрицательные элементы
print("до замены:\n", A2)
A2[A2 < 0] = 0
print("после замены отрицательных на 0:\n", A2)

line("ЗАДАЧА 2.4 — broadcasting: B (4,1) + C (1,5)")

B = np.array([[1], [2], [3], [4]])        # форма (4, 1)
C = np.array([[10, 20, 30, 40, 50]])      # форма (1, 5)
D = B + C
print("B (4,1) =\n", B)
print("C (1,5) =\n", C)
print("B + C (4,5) =\n", D)
# Broadcasting: размерности сравниваются справа налево; формы (4,1) и (1,5)
# совместимы, итоговая форма (4,5). Ни один цикл не нужен.

# =====================================================================
# ЗАДАЧА 3. Memory-mapped файлы и оптимизация кэша
# =====================================================================
line("ЗАДАЧА 3.1 — создание memory-mapped файла 1000x1000 float64")

mmap_path = os.path.join(OUT, "lab1_array.dat")
mm = np.memmap(mmap_path, dtype=np.float64, mode="w+", shape=(1000, 1000))
mm[:] = np.random.rand(1000, 1000)   # заполняем случайными числами
mm.flush()
print(f"файл {mmap_path}, размер: {os.path.getsize(mmap_path)/1e6:.1f} МБ")
print(f"массив: shape={mm.shape}, dtype={mm.dtype}")

line("ЗАДАЧА 3.2 — чтение отдельного столбца и строки БЕЗ загрузки всего массива")

mm2 = np.memmap(mmap_path, dtype=np.float64, mode="r", shape=(1000, 1000))
col7 = mm2[:, 7]          # читается только один столбец с диска
row42 = mm2[42, :]        # читается только одна строка
print(f"столбец 7:  первые 5 значений = {col7[:5]},  sum = {col7.sum():.4f}")
print(f"строка 42:  первые 5 значений = {row42[:5]},  sum = {row42.sum():.4f}")
print("вся матрица в память НЕ загружалась")

line("ЗАДАЧА 3.3 — сравнение обхода по строкам (row-major) и по столбцам")

big = np.random.rand(4000, 4000)

t0 = time.perf_counter()
s = 0.0
for row in big:           # идем по строкам = по порядку в памяти
    s += row.sum()
t_row = time.perf_counter() - t0

t0 = time.perf_counter()
s = 0.0
for j in range(big.shape[1]):   # идем по столбцам = скачем по памяти
    s += big[:, j].sum()
t_col = time.perf_counter() - t0

print(f"обход по строкам (row-major, по порядку в памяти): {t_row:.4f} с")
print(f"обход по столбцам (column-major, скачки по памяти): {t_col:.4f} с")
print(f"обход по столбцам медленнее в ~{t_col/t_row:.1f} раза")
# Причина: NumPy хранит массивы в C-порядке (row-major). Последовательный
# обход дает попадания в кэш-память (cache hits), а обход по столбцам —
# постоянные промахи кэша (cache misses), т.к. элементы одного столбца
# лежат в памяти через stride = 4000*8 байт.

# =====================================================================
# ЗАДАЧА 4. Разреженные матрицы
# =====================================================================
line("ЗАДАЧА 4.1 — разреженная матрица 5x5, плотность ~20%, формат CSR")

from scipy import sparse

dense5 = np.zeros((5, 5))
dense5[0, 1] = 1.0
dense5[0, 4] = 2.0
dense5[2, 2] = 3.0
dense5[3, 0] = 4.0
dense5[4, 3] = 5.0                     # 5 ненулевых из 25 = 20%
csr = sparse.csr_matrix(dense5)
print("плотная версия:\n", dense5)
print("CSR: nnz =", csr.nnz, ", density =", csr.nnz / 25)
print("data   =", csr.data)            # значения ненулевых элементов
print("indices=", csr.indices)         # индексы столбцов
print("indptr =", csr.indptr)          # указатели начала строк

line("ЗАДАЧА 4.2 — преобразование CSR -> COO и CSC")

coo = csr.tocoo()
csc = csr.tocsc()
print("COO: row =", coo.row, ", col =", coo.col, ", data =", coo.data)
print("CSC: data =", csc.data, ", indices =", csc.indices, ", indptr =", csc.indptr)
print("CSR и CSC и COO при обратном преобразовании дают ту же матрицу:",
      np.allclose(csr.toarray(), coo.toarray()),
      np.allclose(csr.toarray(), csc.toarray()))

line("ЗАДАЧА 4.3 — умножение разреженной матрицы на плотный вектор")

v = np.array([1., 2., 3., 4., 5.])
prod = csr @ v
print("v          =", v)
print("CSR @ v    =", prod)
print("проверка (плотное умножение):", dense5 @ v)

line("ЗАДАЧА 4.4 — сравнение времени умножения: разреженная vs плотная (1000x1000, 1%)")

n = 1000
big_dense = np.random.rand(n, n)
big_dense[big_dense < 0.99] = 0.0      # плотность ~1%
big_csr = sparse.csr_matrix(big_dense)
print(f"плотность: nnz/n^2 = {big_csr.nnz / (n*n):.4f}")
vec = np.random.rand(n)

t0 = time.perf_counter(); big_csr @ vec; t_sparse = time.perf_counter() - t0
t0 = time.perf_counter(); big_dense @ vec; t_dense = time.perf_counter() - t0
print(f"разреженная (CSR) @ вектор: {t_sparse*1e3:.3f} мс")
print(f"плотная           @ вектор: {t_dense*1e3:.3f} мс")
print(f"разреженное умножение быстрее в ~{t_dense/t_sparse:.1f} раза")
print(f"память: плотная = {big_dense.nbytes/1e6:.1f} МБ, "
      f"CSR = {(big_csr.data.nbytes + big_csr.indices.nbytes + big_csr.indptr.nbytes)/1e6:.1f} МБ")

# =====================================================================
# ЗАДАЧА 5. Cython — ускорение циклов
# =====================================================================
line("ЗАДАЧА 5.1 — сумма квадратов на чистом Python")


def sum_sq_py(lst):
    """Сумма квадратов элементов списка (чистый Python, медленный цикл)."""
    s = 0.0
    for x in lst:
        s += x * x
    return s


N = 10**7
data_list = [0.5] * N          # список из 10^7 элементов
t0 = time.perf_counter()
res_py = sum_sq_py(data_list)
t_py = time.perf_counter() - t0
print(f"sum_sq_py(list из {N:,} элементов) = {res_py:.3f}, время = {t_py:.3f} с")

line("ЗАДАЧА 5.2 — Cython-версия (.pyx) и компиляция")

# Код Cython-модуля сохраняем в файл. Компиляция выполняется командой:
#   python setup.py build_ext --inplace
pyx_code = """# fastsum.pyx
# cython: language_level=3

def sum_sq_cython(list lst):
    cdef double s = 0.0
    cdef double x
    for x in lst:          # типизированный цикл -> чистый C-код
        s += x * x
    return s
"""

setup_code = """# setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("fastsum.pyx"))
"""

workdir = os.path.join(OUT, "cython_part")
os.makedirs(workdir, exist_ok=True)
with open(os.path.join(workdir, "fastsum.pyx"), "w", encoding="utf-8") as f:
    f.write(pyx_code)
with open(os.path.join(workdir, "setup.py"), "w", encoding="utf-8") as f:
    f.write(setup_code)
print("Сохранены файлы fastsum.pyx и setup.py в", workdir)
print("Для компиляции:  cd cython_part && python setup.py build_ext --inplace")

cython_ok = False
try:
    from Cython.Build import cythonize
    from setuptools import setup
    cwd = os.getcwd()
    os.chdir(workdir)
    setup(script_args=["build_ext", "--inplace"],
          ext_modules=cythonize("fastsum.pyx", quiet=True))
    os.chdir(cwd)
    cython_ok = True
    print("Cython-модуль успешно скомпилирован")
except Exception as e:
    print(f"Cython недоступен в этой среде ({e}).")
    print("Компиляция выполняется на машине с установленными Cython и gcc:")


line("ЗАДАЧА 5.3 — сравнение времени выполнения (10^7 элементов)")

if cython_ok:
    import sys
    # Добавляем папку с cython-модулем в пути поиска Python
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
        
    import fastsum
    t0 = time.perf_counter()
    res_cy = fastsum.sum_sq_cython(data_list)
    t_cy = time.perf_counter() - t0
    print(f"Cython:  {res_cy:.3f}, время = {t_cy:.3f} с  (ускорение x{t_py/t_cy:.1f})")
else:
    # Запасной вариант в среде без Cython: векторизованный NumPy как эталон
    arr = np.array(data_list)
    t0 = time.perf_counter()
    res_np = float((arr**2).sum())
    t_np = time.perf_counter() - t0
    print(f"(среда без Cython) NumPy-версия: {res_np:.3f}, время = {t_np:.3f} с")
    print(f"Векторизованный NumPy быстрее питоновского цикла в "
          f"~{t_py/t_np:.0f} раз.")
    print("На машине с Cython компилируемая функция из п.5.2 обычно "
          "еще в ~10-30 раз быстрее цикла на чистом Python "
          "(типизация double убирает boxing/unboxing).")

print("\nВсе задачи лабораторной работы выполнены.")
