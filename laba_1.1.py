import gc
import os
import time
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt


def _close_memmap(mm):
    """Надёжно освобождает дескриптор memory-mapped файла (актуально для Windows)."""
    if mm is None:
        return
    try:
        if getattr(mm, "_mmap", None) is not None:
            mm._mmap.close()
    except Exception:
        pass
    del mm
    gc.collect()


def task_6():
    print("=" * 60)
    print("ЗАДАЧА 6: Индексация, срезы и broadcasting")
    print("=" * 60)
    np.random.seed(42)

    # 6.1 Одномерный массив[cite: 4]
    arr = np.arange(20)
    print("Исходный arr:", arr)
    print("6.1.1 Каждый 3-й элемент с индекса 1:", arr[1::3])
    print("6.1.2 Элементы с 5-го по 15-й (не вкл.) в обратном порядке:", arr[14:4:-1])
    print("6.1.3 Последние 5 элементов в обратном порядке:", arr[-1:-6:-1])

    # 6.2 Матрица 6x6[cite: 4]
    mat = np.random.randint(0, 10, size=(6, 6))
    print("\nМатрица 6x6:\n", mat)
    print("6.2.1 Подматрица (строки 2-4, столбцы 1-3 вкл.):\n", mat[2:5, 1:4])
    print("6.2.2 Главная диагональ:", np.diagonal(mat))
    idx_sum = np.add.outer(np.arange(6), np.arange(6))
    print("6.2.3 Шахматный узор (чётная сумма индексов):", mat[idx_sum % 2 == 0])

    # 6.3 Продвинутая индексация[cite: 4]
    a = np.array([10, 20, 30, 40, 50, 60])
    m = np.arange(12).reshape(3, 4)
    print("\n6.3.1 Элементы a по индексам [0, 2, 4]:", a[[0, 2, 4]])
    print("6.3.2 Попарные элементы m[(0,1), (1,3), (2,0)]:", m[[0, 1, 2], [1, 3, 0]])
    print("6.3.3 Блок 2x2 из строк [0, 2] и столбцов [1, 3]:\n", m[np.ix_([0, 2], [1, 3])])

    # 6.4 Булева индексация и сдвиги[cite: 4]
    x = np.random.randn(1000)
    print("\n6.4.1 Количество элементов > 1:", np.sum(x > 1))
    x_no_neg = np.where(x < 0, 0, x)
    print("6.4.2 Минимум после замены отрицательных на 0:", x_no_neg.min())

    x_sign = x.copy()
    sign_change = (x[:-2] * x[2:]) < 0
    x_sign[1:-1][sign_change] = 0
    print("6.4.3 Обнулено элементов при смене знака соседей:", np.sum(sign_change))


def task_7():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 7: Memory-mapped файлы и оптимизация кэша")
    print("=" * 60)
    np.random.seed(42)

    # 7.1 Создание big.dat на диске с массивом (50000, 200) типа float32[cite: 6]
    shape = (50000, 200)
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "big.dat")
    batch_size = 5000

    fp_w = np.memmap(filename, dtype="float32", mode="w+", shape=shape)
    try:
        for start in range(0, shape[0], batch_size):
            end = min(start + batch_size, shape[0])
            fp_w[start:end] = np.random.rand(end - start, shape[1]).astype("float32")
        fp_w.flush()
    finally:
        _close_memmap(fp_w)
        fp_w = None

    actual_bytes = os.path.getsize(filename)
    expected_bytes = 50000 * 200 * 4
    print(f"7.1 Размер файла: {actual_bytes} байт (ожидалось {expected_bytes} байт)")

    # 7.2 Чтение батчами в режиме 'r' без загрузки всего массива через [:][cite: 6]
    g_min = np.inf
    g_max = -np.inf
    g_sum = 0.0
    total_elements = shape[0] * shape[1]

    fp_r = np.memmap(filename, dtype="float32", mode="r", shape=shape)
    try:
        for start in range(0, shape[0], batch_size):
            end = min(start + batch_size, shape[0])
            chunk = fp_r[start:end]
            g_min = min(g_min, float(chunk.min()))
            g_max = max(g_max, float(chunk.max()))
            g_sum += float(chunk.sum(dtype=np.float64))
    finally:
        _close_memmap(fp_r)
        fp_r = None

    g_mean = g_sum / total_elements
    print(f"7.2 Min: {g_min:.6f}, Max: {g_max:.6f}, Mean: {g_mean:.6f}")

    # Безопасное удаление big.dat после освобождения всех дескрипторов
    for _ in range(5):
        gc.collect()
        try:
            if os.path.exists(filename):
                os.remove(filename)
            break
        except PermissionError:
            time.sleep(0.1)

    # 7.3 Влияние порядка хранения (C-order vs F-order) на время обхода в Python-циклах[cite: 6]
    N = 3000
    arr_c = np.ones((N, N), dtype=np.float64, order="C")
    arr_f = np.ones((N, N), dtype=np.float64, order="F")

    def sum_by_rows(a):
        total = 0.0
        rows_n, cols_n = a.shape
        for i in range(rows_n):
            for j in range(cols_n):
                total += a[i, j]
        return total

    def sum_by_cols(a):
        total = 0.0
        rows_n, cols_n = a.shape
        for j in range(cols_n):
            for i in range(rows_n):
                total += a[i, j]
        return total

    t0 = time.perf_counter()
    s_c_row = sum_by_rows(arr_c)
    t_c_row = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_c_col = sum_by_cols(arr_c)
    t_c_col = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_f_row = sum_by_rows(arr_f)
    t_f_row = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_f_col = sum_by_cols(arr_f)
    t_f_col = time.perf_counter() - t0

    print(
        f"7.3 C-order: построчно={t_c_row:.3f} с (sum={s_c_row:.0f}), "
        f"постолбцово={t_c_col:.3f} с (sum={s_c_col:.0f})"
    )
    print(
        f"    F-order: построчно={t_f_row:.3f} с (sum={s_f_row:.0f}), "
        f"постолбцово={t_f_col:.3f} с (sum={s_f_col:.0f})"
    )

    # 7.4 Вычисление (a + b) / (a - b + 1) на 10 млн элементов float64[cite: 6]
    M = 10_000_000
    a = np.random.uniform(1.0, 10.0, size=M)
    b = np.random.uniform(0.1, 0.9, size=M)
    a_list = a.tolist()
    b_list = b.tolist()

    t0 = time.perf_counter()
    res_list = [(x + y) / (x - y + 1.0) for x, y in zip(a_list, b_list)]
    t_list = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_np = (a + b) / (a - b + 1.0)
    t_np = time.perf_counter() - t0

    num_buf = np.empty_like(a)
    den_buf = np.empty_like(a)
    out_buf = np.empty_like(a)

    t0 = time.perf_counter()
    np.add(a, b, out=num_buf)
    np.subtract(a, b, out=den_buf)
    np.add(den_buf, 1.0, out=den_buf)
    np.divide(num_buf, den_buf, out=out_buf)
    t_out = time.perf_counter() - t0

    max_diff = float(np.max(np.abs(res_np - out_buf)))
    print(
        f"7.4 Python list: {t_list:.4f} с | NumPy 1-line: {t_np:.4f} с | "
        f"NumPy out=: {t_out:.4f} с (len(list)={len(res_list)}, max_diff={max_diff:.2e})"
    )


def task_8():
    print("\n" + "=" * 60)
    print("ЗАДАЧА 8: Разреженные матрицы (SciPy Sparse)")
    print("=" * 60)
    np.random.seed(42)

    # 8.1 Сборка 5000x5000 в COO из 10 000 элементов и перевод в CSR[cite: 8]
    rows = np.random.randint(0, 5000, size=10000)
    cols = np.random.randint(0, 5000, size=10000)
    data = np.random.rand(10000)

    coo = sparse.coo_matrix((data, (rows, cols)), shape=(5000, 5000))
    csr = coo.tocsr()
    unique_pairs = len(set(zip(rows.tolist(), cols.tolist())))
    print(
        f"8.1 Записей в COO: {len(data)} | Уникальных координат: {unique_pairs} | "
        f"csr.nnz: {csr.nnz} | Совпадает: {csr.nnz == unique_pairs}"
    )

    # 8.2 Матрица 4000x4000 с плотностью 0.5%: память и умножение на вектор[cite: 8]
    csr_4k = sparse.random(
        4000, 4000, density=0.005, format="csr", dtype=np.float64, random_state=42
    )
    dense_4k = csr_4k.toarray()
    x_4k = np.random.rand(4000)

    mem_dense_mb = dense_4k.nbytes / (1024 ** 2)
    mem_csr_mb = (
        csr_4k.data.nbytes + csr_4k.indices.nbytes + csr_4k.indptr.nbytes
    ) / (1024 ** 2)

    # Прогрев перед замером
    _ = dense_4k @ x_4k
    _ = csr_4k @ x_4k

    repeats_4k = 50
    t0 = time.perf_counter()
    for _ in range(repeats_4k):
        _ = dense_4k @ x_4k
    t_dense_ms = (time.perf_counter() - t0) / repeats_4k * 1000

    t0 = time.perf_counter()
    for _ in range(repeats_4k):
        _ = csr_4k @ x_4k
    t_csr_ms = (time.perf_counter() - t0) / repeats_4k * 1000

    print(
        f"8.2 Память: Dense={mem_dense_mb:.2f} МБ vs CSR={mem_csr_mb:.2f} МБ "
        f"(экономия в {mem_dense_mb / mem_csr_mb:.1f}x)"
    )
    print(f"    Время A @ x: Dense={t_dense_ms:.4f} мс vs CSR={t_csr_ms:.4f} мс")

    # 8.3 Зависимость времени A @ x от плотности при n = 2000[cite: 8]
    densities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    x_2k = np.random.rand(2000)
    t_sp_list = []
    t_dn_list = []
    crossover_density = None

    print("\n8.3 Время умножения A @ x (n = 2000):")
    print(f"{'density':<10} | {'CSR (мс)':<12} | {'Dense (мс)':<12} | {'Быстрее'}")
    print("-" * 52)

    repeats_2k = 30
    for d in densities:
        sp_m = sparse.random(
            2000, 2000, density=d, format="csr", dtype=np.float64, random_state=42
        )
        dn_m = sp_m.toarray()

        _ = sp_m @ x_2k
        _ = dn_m @ x_2k

        t0 = time.perf_counter()
        for _ in range(repeats_2k):
            _ = sp_m @ x_2k
        t_sp = (time.perf_counter() - t0) / repeats_2k * 1000

        t0 = time.perf_counter()
        for _ in range(repeats_2k):
            _ = dn_m @ x_2k
        t_dn = (time.perf_counter() - t0) / repeats_2k * 1000

        t_sp_list.append(t_sp)
        t_dn_list.append(t_dn)

        faster = "CSR" if t_sp < t_dn else "Dense"
        if faster == "Dense" and crossover_density is None:
            crossover_density = d

        print(f"{d:<10} | {t_sp:<12.4f} | {t_dn:<12.4f} | {faster}")

    if crossover_density is not None:
        print(f"\nSparse перестаёт выигрывать у Dense начиная с плотности: {crossover_density}")
    else:
        print("\nНа выбранном диапазоне плотностей CSR оставался быстрее Dense.")

    chart_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sparse_vs_dense.png"
    )
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(densities, t_sp_list, marker="o", linewidth=2, label="CSR Sparse (A @ x)")
    plt.plot(densities, t_dn_list, marker="s", linewidth=2, label="Dense NumPy (A @ x)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Плотность матрицы (логарифмическая шкала)")
    plt.ylabel("Время умножения, мс (логарифмическая шкала)")
    plt.title("Время умножения A @ x в зависимости от плотности (n = 2000)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"График сохранён в: {chart_path}")


if __name__ == "__main__":
    task_6()
    task_7()
    task_8()