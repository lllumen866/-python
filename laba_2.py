import numpy as np
import pandas as pd


def main():
    # ==========================================================
    # ЗАДАЧА 1: Создание Series и DataFrame, базовый осмотр
    # ==========================================================
    print("=" * 60)
    print("ЗАДАЧА 1: Создание Series и DataFrame")
    print("=" * 60)
    s = pd.Series([100, 250, 175, 420, 310], index=["Пн", "Вт", "Ср", "Чт", "Пт"], name="Sales")
    print("1.1 Series:\n", s)

    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"],
        "department": ["IT", "HR", "IT", "Finance", "HR", "IT", "Finance", "HR"],
        "salary": [80000, 60000, 90000, 75000, np.nan, 85000, 70000, 62000],
        "bonus": [5000, 3000, np.nan, 4000, 2000, 6000, 3500, np.nan],
        "hire_date": pd.to_datetime([
            "2020-01-15", "2019-03-22", "2021-07-01", "2018-11-05",
            "2022-02-10", "2020-09-30", "2019-12-12", "2021-05-20"
        ])
    }
    df = pd.DataFrame(data)
    print("\n1.3 Первые 3 строки:\n", df.head(3))
    print("\nПоследние 3 строки:\n", df.tail(3))
    print("\nИнформация о типах данных:")
    df.info()
    print("\nОписательная статистика:\n", df.describe())

    # ==========================================================
    # ЗАДАЧА 2: Индексация, выборка, фильтрация
    # ==========================================================
    print("\n" + "=" * 60)
    print("ЗАДАЧА 2: Индексация, выборка, фильтрация")
    print("=" * 60)
    df_t2 = df.copy()
    print("2.1 Сотрудники IT (name, salary):\n", df_t2.loc[df_t2["department"] == "IT", ["name", "salary"]])
    print("\n2.2 Первые 3 строки и 2 столбца (iloc):\n", df_t2.iloc[:3, :2])
    print("\n2.3 Сотрудники с зарплатой > 70000:\n", df_t2[df_t2["salary"] > 70000])

    df_t2["total_comp"] = df_t2["salary"] + df_t2["bonus"].fillna(0)
    df_t2 = df_t2.drop(columns=["hire_date"])
    print("\n2.4-2.5 После добавления total_comp и удаления hire_date:\n", df_t2)

    # ==========================================================
    # ЗАДАЧА 3: Обработка пропущенных значений
    # ==========================================================
    print("\n" + "=" * 60)
    print("ЗАДАЧА 3: Обработка пропущенных значений")
    print("=" * 60)
    print("3.1 Количество пропусков по столбцам:\n", df.isnull().sum())

    df["salary"] = df["salary"].fillna(df.groupby("department")["salary"].transform("mean"))
    df["bonus"] = df["bonus"].fillna(0)
    df = df.dropna(subset=["salary"]).copy()
    df["total_comp"] = df["salary"] + df["bonus"]
    print("\n3.2-3.4 DataFrame после заполнения пропусков:\n", df)

    # ==========================================================
    # ЗАДАЧА 4: Группировка и агрегация
    # ==========================================================
    print("\n" + "=" * 60)
    print("ЗАДАЧА 4: Группировка и агрегация")
    print("=" * 60)
    print("4.1 Средняя зарплата по отделам:\n", df.groupby("department")["salary"].mean())

    dept_stats = df.groupby("department").agg(
        mean_salary=("salary", "mean"),
        max_bonus=("bonus", "max"),
        emp_count=("name", "count")
    )
    print("\n4.2 Агрегация по отделам:\n", dept_stats)

    pivot = pd.pivot_table(df, values="salary", index="department", aggfunc="mean")
    print("\n4.3 Сводная таблица:\n", pivot)
    print("\n4.4 Сортировка отделов по средней зарплате (убыв.):\n",
          dept_stats.sort_values(by="mean_salary", ascending=False))

    # ==========================================================
    # ЗАДАЧА 5: Объединение данных
    # ==========================================================
    print("\n" + "=" * 60)
    print("ЗАДАЧА 5: Объединение данных (merge и concat)")
    print("=" * 60)
    managers = pd.DataFrame({
        "department": ["IT", "HR", "Finance"],
        "manager": ["Иванов И.И.", "Петрова А.С.", "Сидоров К.В."]
    })
    df_merged = pd.merge(df, managers, on="department", how="left")

    new_emp = pd.DataFrame([{
        "name": "Ivan",
        "department": "IT",
        "salary": 88000.0,
        "bonus": 4500.0,
        "hire_date": pd.to_datetime("2023-04-01"),
        "total_comp": 92500.0,
        "manager": "Иванов И.И."
    }])
    df_final = pd.concat([df_merged, new_emp], ignore_index=True)
    print("5.1-5.4 Итоговая таблица после merge и concat:\n", df_final)

    # ==========================================================
    # ЗАДАЧА 6: Комплексный мини-анализ
    # ==========================================================
    print("\n" + "=" * 60)
    print("ЗАДАЧА 6: Комплексный мини-анализ")
    print("=" * 60)
    top3 = df_final.nlargest(3, "total_comp")[["name", "department", "salary", "bonus", "total_comp"]]
    print("6.1 Топ-3 сотрудника по total_comp:\n", top3)

    avg_comp = df_final.groupby("department")[["salary", "bonus"]].mean().rename(
        columns={"salary": "mean_salary", "bonus": "mean_bonus"}
    )
    print("\n6.2 Средняя зарплата и средний бонус по отделам:\n", avg_comp)

    share_high = df_final.groupby("department")["salary"].apply(lambda s: (s > 75000).mean())
    print("\n6.3 Доля сотрудников с зарплатой > 75000 по отделам:\n", share_high)


if __name__ == "__main__":
    main()