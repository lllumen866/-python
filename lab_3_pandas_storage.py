import os
import time
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("\n" + "=" * 70)
print("ЗАДАЧА 1. Подготовка тестового набора данных")
print("=" * 70)
np.random.seed(42)
n_rows = 10_000
departments = ["IT", "Analytics", "Marketing", "Finance", "HR"]
names_pool = ["Иван Иванов", "Анна Смирнова", "Пётр Кузнецов", "Елена Попова", "Дмитрий Соколов"]

df = pd.DataFrame({
    "id": np.arange(1, n_rows + 1, dtype="int64"),
    "name": np.random.choice(names_pool, size=n_rows),
    "age": np.random.randint(21, 60, size=n_rows, dtype="int32"),
    "salary": np.round(np.random.uniform(50_000, 250_000, size=n_rows), 2),
    "department": np.random.choice(departments, size=n_rows),
    "hiredate": pd.to_datetime("2018-01-01") + pd.to_timedelta(
        np.random.randint(0, 2500, size=n_rows), unit="D"
    ),
    "is_active": np.random.choice([True, False], size=n_rows, p=[0.85, 0.15])
})
print(df.head())
print("\nТипы данных исходного df:\n", df.dtypes)

print("\n" + "=" * 70)
print("ЗАДАЧА 2. CSV — чтение и запись")
print("=" * 70)
df.to_csv("employees_with_index.csv", encoding="utf-8-sig")
df.to_csv("employees_no_index.csv", index=False, encoding="utf-8-sig")
df_csv_raw = pd.read_csv("employees_no_index.csv")
print("Типы после обычного read_csv:\n", df_csv_raw.dtypes)
size_with_idx = os.path.getsize("employees_with_index.csv")
size_no_idx = os.path.getsize("employees_no_index.csv")
print(f"Размер с index: {size_with_idx} байт | Без index: {size_no_idx} байт")
df_csv_opt = pd.read_csv(
    "employees_no_index.csv",
    dtype={"id": "int64", "age": "int32", "salary": "float64",
           "department": "category", "is_active": "bool"},
    parse_dates=["hiredate"],
    usecols=["id", "name", "salary", "department", "hiredate", "is_active"]
)
print("\nТипы после оптимизированного чтения:\n", df_csv_opt.dtypes)

print("\n" + "=" * 70)
print("ЗАДАЧА 3. JSON — работа с иерархическими данными")
print("=" * 70)
df_nested = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Иван Иванов", "Анна Смирнова", "Пётр Кузнецов"],
    "skills": [["Python", "Pandas", "SQL"], ["Excel", "Power BI"],
               ["Python", "Spark", "Airflow", "Docker"]],
    "profile": [{"city": "Москва", "grade": "Senior"},
                {"city": "Санкт-Петербург", "grade": "Middle"},
                {"city": "Казань", "grade": "Lead"}]
})
for orient in ["records", "columns", "index", "split"]:
    fname = f"nested_{orient}.json"
    df_nested.to_json(fname, orient=orient, force_ascii=False, indent=2)
    print(f"orient='{orient}': {os.path.getsize(fname)} байт")
with open("nested_records.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
df_flat = pd.json_normalize(raw_data)
print("\nРазвёрнутая таблица (json_normalize):")
print(df_flat)

print("\n" + "=" * 70)
print("ЗАДАЧА 4. Parquet — колоночный формат для аналитики")
print("=" * 70)
parquet_available = True
try:
    import pyarrow  # noqa: F401
except ImportError:
    try:
        import fastparquet  # noqa: F401
    except ImportError:
        parquet_available = False

if parquet_available:
    codecs = ["snappy", "gzip", "brotli"]
    for codec in codecs:
        df.to_parquet(f"employees_{codec}.parquet", compression=codec, index=False)
    print(f"CSV (без индекса): {os.path.getsize('employees_no_index.csv')} байт")
    for codec in codecs:
        print(f"Parquet ({codec}): {os.path.getsize(f'employees_{codec}.parquet')} байт")
    t0 = time.perf_counter()
    _ = pd.read_csv("employees_no_index.csv")
    t_csv = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    df_pq = pd.read_parquet("employees_snappy.parquet")
    t_pq = (time.perf_counter() - t0) * 1000
    print(f"\nВремя чтения CSV: {t_csv:.2f} мс | Parquet (snappy): {t_pq:.2f} мс")
    t0 = time.perf_counter()
    df_pq_subset = pd.read_parquet("employees_snappy.parquet", columns=["id", "department", "salary"])
    t_pq_sub = (time.perf_counter() - t0) * 1000
    print(f"Время чтения 3 столбцов Parquet: {t_pq_sub:.2f} мс")
    print("\nТипы данных после read_parquet:\n", df_pq.dtypes)
else:
    print("Parquet пропущен: установите backend командой pip install pyarrow")

print("\n" + "=" * 70)
print("ЗАДАЧА 5. HDF5 — иерархическое хранилище")
print("=" * 70)
try:
    import tables  # noqa: F401
    df_agg = df.groupby("department", as_index=False).agg(
        emp_count=("id", "count"), avg_salary=("salary", "mean"), max_salary=("salary", "max")
    )
    hdf_fixed_file = "company_fixed.h5"
    df.to_hdf(hdf_fixed_file, key="employees", mode="w", format="fixed")
    df_agg.to_hdf(hdf_fixed_file, key="dept_summary", mode="a", format="fixed")
    df_dept_loaded = pd.read_hdf(hdf_fixed_file, key="dept_summary")
    print("Загруженная сводка по отделам:\n", df_dept_loaded)

    hdf_table_file = "company_table.h5"
    df.to_hdf(hdf_table_file, key="employees", mode="w", format="table",
              data_columns=["department", "salary", "is_active"])
    df_filtered = pd.read_hdf(
        hdf_table_file, key="employees",
        where="department == 'IT' & salary > 200000 & is_active == True"
    )
    print(f"\nНайдено сотрудников по условию where: {len(df_filtered)}")
    print(df_filtered.head())
    print(f"\nCSV: {os.path.getsize('employees_no_index.csv')} байт")
    print(f"HDF5 (fixed): {os.path.getsize(hdf_fixed_file)} байт")
    print(f"HDF5 (table): {os.path.getsize(hdf_table_file)} байт")
except ImportError:
    print("HDF5 пропущен: установите PyTables командой pip install tables")

print("\n" + "=" * 70)
print("ЗАДАЧА 6. Итоговое сравнение форматов и выводы")
print("=" * 70)
results = []

def benchmark_format(name, filename, write_fn, read_fn):
    t0 = time.perf_counter()
    write_fn(filename)
    write_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    _ = read_fn(filename)
    read_ms = (time.perf_counter() - t0) * 1000
    size_kb = os.path.getsize(filename) / 1024
    results.append({
        "Формат": name,
        "Размер (КБ)": round(size_kb, 1),
        "Запись (мс)": round(write_ms, 2),
        "Чтение (мс)": round(read_ms, 2)
    })

benchmark_format("CSV (index=False)", "bench_test.csv",
                 lambda f: df.to_csv(f, index=False), lambda f: pd.read_csv(f))
benchmark_format("JSON (records)", "bench_test.json",
                 lambda f: df.to_json(f, orient="records"),
                 lambda f: pd.read_json(f, orient="records"))

if parquet_available:
    benchmark_format("Parquet (snappy)", "bench_test_snappy.parquet",
                    lambda f: df.to_parquet(f, compression="snappy", index=False),
                    lambda f: pd.read_parquet(f))
    benchmark_format("Parquet (gzip)", "bench_test_gzip.parquet",
                    lambda f: df.to_parquet(f, compression="gzip", index=False),
                    lambda f: pd.read_parquet(f))

try:
    import tables  # noqa: F401
    benchmark_format("HDF5 (fixed)", "bench_test.h5",
                    lambda f: df.to_hdf(f, key="data", mode="w", format="fixed"),
                    lambda f: pd.read_hdf(f, key="data"))
except ImportError:
    pass

report_df = pd.DataFrame(results)
print(report_df.to_string(index=False))
print("\nВыводы по форматам:")
print("- Parquet: предпочтителен для аналитики, сжатия и выборочного чтения столбцов.")
print("- CSV: универсален для плоских таблиц и обмена.")
print("- JSON: удобен для API и вложенных структур.")
print("- HDF5: подходит для нескольких таблиц и индексируемых запросов в одном контейнере.")
