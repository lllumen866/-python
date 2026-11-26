from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ORDERS_FILE = BASE_DIR / "orders.csv"
OUTPUT_FILE = BASE_DIR / "orders_final.csv"


def create_demo_orders(path: Path) -> None:
    """Создать учебный orders.csv, если исходный файл не предоставлен."""
    rows = [
        # order_id, client, city, category, product, qty, price, discount,
        # order_date, delivery_days, status, promo_code, manager
        (1, " Ivan Petrov ", " moscow ", " Electronics ", " laptop ", 1, 150000, .10, "2024-01-05", 3, "paid", "SAVE10", "Ivan Sidorov"),
        (2, "Ivan Petrov", "Moscow", "electronics", "laptop", 1, 76000, .10, "2024-01-05", 5, "paid", "SAVE10", "Ivan Sidorov"),
        (3, "Anna Smirnova", "Saint Petersburg", "home", "chair", 3, 5000, .10, "2024-01-10", 4, "new", "WELCOME", "Petr Kuznetsov"),
        (4, "Pavel Ivanov", "Kazan", "electronics", "monitor", 1, 17000, .10, "2024-01-12", 3, "paid", "SAVE10", "Anna Volkova"),
        (5, "Maria Sokolova", "Novosibirsk", "office", "desk", 1, 24000, .20, "2024-01-15", 5, "shipped", "NO_PROMO", "Olga Morozova"),
        (6, "Sergey Volkov", "Moscow", "electronics", "headphones", 1, 10000, 0, "2024-01-17", 6, "cancelled", "NO_PROMO", "Dmitry Orlov"),
        (7, "Elena Petrova", "Kazan", "electronics", "tablet", 1, 55000, .25, "2024-01-18", 2, "paid", "VIP25", "Ivan Sidorov"),
        (8, "Dmitry Smirnov", "Moscow", "home", "sofa", 1, 25000, .20, "2024-01-20", 7, "returned", "NO_PROMO", "Petr Kuznetsov"),
        (9, "Olga Kuznetsova", "Saint Petersburg", "electronics", "phone", 1, 60000, .05, "2024-01-22", 3, "paid", "SAVE5", "Anna Volkova"),
        (10, "Anna Smirnova", "Saint Petersburg", "electronics", "phone", 1, 50000, .10, "2024-02-14", 4, "paid", "SAVE10", "Anna Volkova"),
        (11, "Anna Smirnova", "Saint Petersburg", "electronics", "phone", 1, 120000, .20, "2024-02-14", 4, "paid", "VIP20", "Petr Kuznetsov"),
        (12, "Alexey Popov", "Moscow", "electronics", "camera", 1, 50000, .10, "2024-02-20", 3, "shipped", "SAVE10", "Olga Morozova"),
        (13, "Natalia Fedorova", "Kazan", "office", "printer", 1, 18000, .10, "2024-02-21", 2, "new", "WELCOME", "Dmitry Orlov"),
        (14, "Alexey Popov", "Moscow", "electronics", "monitor", 1, 50000, .09, "2024-02-22", 3, "paid", "SAVE10", "Ivan Sidorov"),
    ]
    rng = np.random.default_rng(42)
    clients = ["Ivan Petrov", "Anna Smirnova", "Pavel Ivanov", "Maria Sokolova",
               "Sergey Volkov", "Elena Petrova", "Dmitry Smirnov", "Olga Kuznetsova",
               "Alexey Popov", "Natalia Fedorova"]
    cities = ["Moscow", "Saint Petersburg", "Kazan", "Novosibirsk"]
    categories = ["electronics", "home", "office"]
    products = ["laptop", "phone", "chair", "monitor", "desk", "tablet", "camera", "printer"]
    statuses = ["new", "paid", "shipped", "cancelled", "returned"]
    promos = ["NO_PROMO", "SAVE10", "WELCOME", "VIP25"]
    managers = ["Ivan Sidorov", "Petr Kuznetsov", "Anna Volkova", "Olga Morozova", "Dmitry Orlov"]
    for order_id in range(15, 31):
        qty = int(rng.integers(1, 4))
        price = int(rng.choice([1200, 3500, 8000, 15000, 25000, 45000, 70000]))
        discount = float(rng.choice([0, .05, .10, .15, .20]))
        rows.append((
            order_id, rng.choice(clients), rng.choice(cities), rng.choice(categories),
            rng.choice(products), qty, price, discount,
            (pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 90)))).strftime("%Y-%m-%d"),
            int(rng.integers(1, 8)), rng.choice(statuses), rng.choice(promos), rng.choice(managers)
        ))
    columns = ["order_id", "client", "city", "category", "product", "qty", "price",
               "discount", "order_date", "delivery_days", "status", "promo_code", "manager"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Создан демонстрационный файл: {path.name} (30 заказов)")


if not ORDERS_FILE.exists():
    create_demo_orders(ORDERS_FILE)

print("\n" + "=" * 70)
print("ЗАДАЧА 7. Нормализация текста и расчёт выручки (df1)")
print("=" * 70)
df1 = pd.read_csv(ORDERS_FILE)
df1["client_clean"] = df1["client"].astype("string").str.strip().str.title()
df1["city_clean"] = df1["city"].astype("string").str.strip().str.title()
df1["category_clean"] = df1["category"].astype("string").str.strip().str.lower()
df1["product_clean"] = df1["product"].astype("string").str.strip().str.upper()
df1["revenue"] = (df1["qty"] * df1["price"] * (1 - df1["discount"])).round(2)
print(df1[["order_id", "client_clean", "city_clean", "category_clean", "product_clean", "revenue"]].head())

print("\n" + "=" * 70)
print("ЗАДАЧА 8. Сортировка и ранжирование по выручке (df2)")
print("=" * 70)
df2 = df1.copy()
df2["revenue_rank"] = df2["revenue"].rank(ascending=False, method="dense").astype(int)
df2 = df2.sort_values(by=["revenue", "order_date"], ascending=[False, True]).reset_index(drop=True)
print(df2[["order_id", "client_clean", "order_date", "revenue", "revenue_rank"]].head(6))

print("\n" + "=" * 70)
print("ЗАДАЧА 9. Даты и календарные признаки (df3)")
print("=" * 70)
df3 = df2.copy()
df3["order_date"] = pd.to_datetime(df3["order_date"])
df3["order_year"] = df3["order_date"].dt.year
df3["order_month"] = df3["order_date"].dt.month
df3["order_quarter"] = df3["order_date"].dt.quarter
df3["order_weekday"] = df3["order_date"].dt.day_name()
df3["is_weekend"] = df3["order_date"].dt.dayofweek >= 5
df3["delivery_date"] = df3["order_date"] + pd.to_timedelta(df3["delivery_days"], unit="D")
df3["days_to_month_end"] = (df3["order_date"] + pd.offsets.MonthEnd(0) - df3["order_date"]).dt.days
print(df3[["order_id", "order_date", "order_quarter", "order_weekday", "is_weekend",
            "delivery_date", "days_to_month_end"]].head())

print("\n" + "=" * 70)
print("ЗАДАЧА 10. Поиск и устранение дублей (df4)")
print("=" * 70)
df4 = df3.copy()
dup_cols = ["client_clean", "order_date", "product_clean"]
df4["duplicate_group_flag"] = df4.duplicated(subset=dup_cols, keep=False)
df4 = (
    df4.sort_values(by="revenue", ascending=False)
       .drop_duplicates(subset=dup_cols, keep="first")
       .sort_values(by="order_id", ascending=True)
       .reset_index(drop=True)
)
print(f"Строк до удаления дублей: {len(df3)}, после: {len(df4)}")
print(df4[df4["duplicate_group_flag"]][["order_id", "client_clean", "order_date",
      "product_clean", "revenue", "duplicate_group_flag"]])

print("\n" + "=" * 70)
print("ЗАДАЧА 11. Категоризация и замена значений (df5)")
print("=" * 70)
df5 = df4.copy()
df5["revenue_segment"] = pd.cut(
    df5["revenue"], bins=[0, 1000, 5000, 20000, np.inf],
    labels=["low", "medium", "high", "vip"], right=False
)
status_map = {
    "new": "новый", "paid": "оплачен", "shipped": "отправлен",
    "cancelled": "отменён", "returned": "возврат"
}
df5["status_ru"] = df5["status"].map(status_map)
df5["promo_used"] = df5["promo_code"] != "NO_PROMO"
print(df5[["order_id", "revenue", "revenue_segment", "status", "status_ru",
            "promo_code", "promo_used"]].head(6))

print("\n" + "=" * 70)
print("ЗАДАЧА 12. Построчное применение функции и итоговый отчёт (df6)")
print("=" * 70)
df6 = df5.copy()

def assign_priority(row):
    if row["status"] in ["cancelled", "returned"]:
        return "archive"
    elif row["revenue"] >= 20000 and row["delivery_days"] <= 3:
        return "A"
    elif row["revenue"] >= 5000:
        return "B"
    else:
        return "C"

df6["priority"] = df6.apply(assign_priority, axis=1)
df6["manager_email"] = (
    df6["manager"].astype("string").str.strip().str.lower()
       .str.replace(" ", ".", regex=False) + "@shop.ru"
)
df6["final_comment"] = df6.apply(
    lambda r: f"{r['client_clean']}: {r['priority']}, {r['revenue_segment']}", axis=1
)
df6.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(df6[["order_id", "priority", "manager_email", "final_comment"]].head(8))
print(f"\nИтоговый файл сохранён: {OUTPUT_FILE.name}")
