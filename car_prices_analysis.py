"""
Used Car Prices - Exploratory Data Analysis (Hero Vired Assignment)
Run:  python car_prices_analysis.py
Needs car_prices.csv in the same folder. Charts are saved to ./charts/
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save charts to files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
os.makedirs("charts", exist_ok=True)

def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

# ---------------------------------------------------------------- PART 1
section("1.1 LOAD & INSPECT")
df = pd.read_csv("car_prices.csv")
print(df.head())
print("\nRecord count:", len(df))
df.info()

section("1.2 STRUCTURE")
print("Shape (rows, columns):", df.shape)
print("Columns:", df.columns.tolist())
print(df.dtypes)

section("1.3 MISSING VALUES & DUPLICATES")
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_summary = pd.DataFrame({"null_count": null_counts, "null_pct": null_pct})
print(null_summary.sort_values("null_pct", ascending=False))

nz = null_summary[null_summary.null_count > 0].sort_values("null_pct", ascending=False)
plt.figure(figsize=(10, 5))
ax = nz["null_pct"].plot(kind="bar", color="indianred")
for i, v in enumerate(nz["null_pct"]):
    ax.text(i, v + 0.15, f"{v}%", ha="center", fontsize=9)
plt.title("Percentage of Missing Values by Column")
plt.ylabel("% missing")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("charts/1_3_missing_values_bar.png", dpi=130)
plt.close()

plt.figure(figsize=(12, 6))
sns.heatmap(df.sample(20000, random_state=1).isnull(), cbar=False,
            yticklabels=False, cmap="Reds")
plt.title("Missing Value Heatmap (random 20,000 rows; red = missing)")
plt.tight_layout()
plt.savefig("charts/1_3_missing_values_heatmap.png", dpi=130)
plt.close()

rows_before = len(df)
# Essential fields: row is useless without them -> drop (tiny %)
df = df.dropna(subset=["vin", "sellingprice", "mmr", "saledate"])
# Numeric -> median (robust to outliers)
df["condition"] = df["condition"].fillna(df["condition"].median())
df["odometer"] = df["odometer"].fillna(df["odometer"].median())
# Categorical -> 'unknown'
for c in ["make", "model", "trim", "body", "transmission", "color", "interior"]:
    df[c] = df[c].fillna("unknown")
print("\nRows dropped for missing essential fields:", rows_before - len(df))
print("Nulls remaining:", int(df.isnull().sum().sum()))

dupes = df.duplicated().sum()
print("Duplicate records found:", dupes)
df = df.drop_duplicates()
print("Shape after null + duplicate handling:", df.shape)

section("1.4 ANOMALY CLEAN-UP (problems discovered while profiling)")
# (a) '—' is used as a placeholder for "no value" in color / interior
for c in ["color", "interior"]:
    n = int((df[c] == "—").sum())
    df[c] = df[c].replace("—", "unknown")
    print(f"'—' placeholders in {c}: {n} -> set to 'unknown'")

# (b) invalid state codes (a VIN got shifted into the state column)
bad_state = ~df["state"].str.fullmatch(r"[a-z]{2}")
print("Rows with an invalid state code:", int(bad_state.sum()), "-> removed")
df = df[~bad_state]

# (c) junk values in transmission (only 'automatic' / 'manual' are valid)
bad_tr = df["transmission"].str.lower().eq("sedan")
print("Rows with 'sedan' typed in transmission:", int(bad_tr.sum()), "-> 'unknown'")
df.loc[bad_tr, "transmission"] = "unknown"

# (d) same brand/model typed in different letter-case (e.g. 'bmw' vs 'BMW')
def canonical(series):
    """Map every spelling to the most common spelling of the same lower-case text."""
    key = series.str.strip().str.lower()
    best = (pd.DataFrame({"key": key, "orig": series})
            .groupby(["key", "orig"]).size().reset_index(name="n")
            .sort_values("n", ascending=False).drop_duplicates("key")
            .set_index("key")["orig"])
    return key.map(best)

makes_raw = df.loc[df["make"] != "unknown", "make"].nunique()
models_raw = df.loc[df["model"] != "unknown", "model"].nunique()
df["make"] = canonical(df["make"])
df["model"] = canonical(df["model"])
print(f"Distinct makes : {makes_raw} -> {df.loc[df['make']!='unknown','make'].nunique()} after merging spellings")
print(f"Distinct models: {models_raw} -> {df.loc[df['model']!='unknown','model'].nunique()} after merging spellings")

# (e) odometer = 999,999 is a placeholder, not a real reading
raw_max_odo = df.groupby("year")["odometer"].max().sort_values(ascending=False)  # keep for Q2.8
n999 = int((df["odometer"] == 999999).sum())
df.loc[df["odometer"] == 999999, "odometer"] = np.nan
df["odometer"] = df["odometer"].fillna(df["odometer"].median())
print(f"Odometer placeholders (999,999): {n999} -> replaced with median")

# (f) NOTE: condition is stored on two scales (1-5 and 10-49). Kept as-is because the
#     assignment asks for condition cut-offs (>=48) and bins; flagged again in 3.5 / 3.6.
print("Rows with condition on the 1-5 scale:", int((df["condition"] <= 5).sum()))
print("Final shape after cleaning:", df.shape)

# ---------------------------------------------------------------- PART 2
section("2.1 AVERAGE / MIN / MAX PRICE")
print(df["sellingprice"].agg(["mean", "min", "max"]).round(2))

section("2.2 UNIQUE COLORS")
print(sorted(df["color"].unique()))

section("2.3 UNIQUE BRANDS & MODELS")
print("Unique makes  (cleaned spellings):", df.loc[df["make"] != "unknown", "make"].nunique(), f"[raw: {makes_raw}]")
print("Unique models (cleaned spellings):", df.loc[df["model"] != "unknown", "model"].nunique(), f"[raw: {models_raw}]")

section("2.4 SELLING PRICE > $165,000")
hp = df[df["sellingprice"] > 165000]
print("Count:", len(hp))
print(hp[["year", "make", "model", "trim", "condition", "odometer", "state", "sellingprice"]])

section("2.5 TOP 5 MODELS")
print(df["model"].value_counts().head(5))

section("2.6 AVG PRICE BY MAKE")
avg_make = df.groupby("make")["sellingprice"].mean().sort_values(ascending=False).round(2)
print(avg_make.head(15)); print("..."); print(avg_make.tail(5))

section("2.7 MIN PRICE BY INTERIOR")
print(df.groupby("interior")["sellingprice"].min().sort_values())

section("2.8 MAX ODOMETER PER YEAR (high -> low)")
print("As recorded (includes 999,999 placeholders):")
print(raw_max_odo.head(12))
print("\nAfter replacing the placeholders:")
print(df.groupby("year")["odometer"].max().sort_values(ascending=False))

section("2.9 CAR AGE COLUMN")
df["car_age"] = 2025 - df["year"]
print(df[["year", "car_age"]].head())

section("2.10 CONDITION >= 48 AND ODOMETER > 90000")
print("Count:", len(df[(df["condition"] >= 48) & (df["odometer"] > 90000)]))

section("2.11 STATE WITH HIGHER PRICES FOR NEWER CARS (year > 2013)")
newer = df[df["year"] > 2013]
print("Model years in this group:", sorted(newer["year"].unique()))
by_state_year = (newer.groupby(["state", "year"])["sellingprice"]
                 .agg(["mean", "count"]).reset_index())
by_state_year = by_state_year[by_state_year["count"] >= 100]   # reliable samples only
pivot = by_state_year.pivot(index="state", columns="year", values="mean").dropna()
pivot["overall_avg"] = pivot.mean(axis=1)
for y in [c for c in pivot.columns if c != "overall_avg"]:
    pivot[f"rank_{y}"] = pivot[y].rank(ascending=False).astype(int)
pivot = pivot.sort_values("overall_avg", ascending=False).round(0)
print(pivot.head(8))

section("2.12 EXCELLENT CONDITION (TOP 20%) - LOWEST AVG PRICE BY MAKE")
thr = df["condition"].quantile(0.80)
print("Top-20% condition threshold:", thr)
exc = df[df["condition"] >= thr]
g = exc.groupby("make")["sellingprice"].agg(["mean", "count"])
g = g[g["count"] >= 30].sort_values("mean")        # ignore makes with tiny samples
print(g.head(10).round(0))

# ---------------------------------------------------------------- PART 3
section("3.1 CORRELATION")
num_cols = ["year", "condition", "odometer", "mmr", "sellingprice", "car_age"]
corr = df[num_cols].corr()
print(corr.round(3))
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation of Numeric Features")
plt.tight_layout(); plt.savefig("charts/3_1_correlation.png", dpi=130); plt.close()

section("3.2 AVG PRICE BY YEAR")
avg_year = df.groupby("year")["sellingprice"].mean().sort_index()
print(avg_year.round(0))
plt.figure(figsize=(12, 5))
avg_year.plot(kind="bar", color="steelblue")
plt.title("Average Selling Price by Model Year")
plt.xlabel("Model year"); plt.ylabel("Average selling price ($)")
plt.tight_layout(); plt.savefig("charts/3_2_price_by_year.png", dpi=130); plt.close()

section("3.3 AVG PRICE BY ODOMETER")
df["odo_bucket"] = pd.cut(df["odometer"], bins=range(0, 400001, 20000))
avg_odo = df.groupby("odo_bucket", observed=True)["sellingprice"].agg(["mean", "count"])
print(avg_odo.round(0))
avg_odo = avg_odo[avg_odo["count"] >= 100]
plt.figure(figsize=(12, 5))
plt.plot(range(len(avg_odo)), avg_odo["mean"], marker="o", color="darkorange")
plt.xticks(range(len(avg_odo)),
           [f"{int(i.left/1000)}-{int(i.right/1000)}k" for i in avg_odo.index],
           rotation=45, ha="right")
plt.title("Average Selling Price by Odometer Range (miles)")
plt.xlabel("Odometer range"); plt.ylabel("Average selling price ($)")
plt.tight_layout(); plt.savefig("charts/3_3_price_by_odometer.png", dpi=130); plt.close()

section("3.4 CARS SOLD PER STATE")
by_state = df["state"].value_counts()
print(by_state.head(10))
plt.figure(figsize=(14, 5))
by_state.plot(kind="bar", color="seagreen")
plt.title("Number of Cars Sold by State")
plt.xlabel("State"); plt.ylabel("Cars sold")
plt.tight_layout(); plt.savefig("charts/3_4_cars_by_state.png", dpi=130); plt.close()
print("TOP 3 STATES:", by_state.head(3).to_dict())

section("3.5 AVG PRICE BY CONDITION RANGE (size 5)")
df["cond5"] = pd.cut(df["condition"], bins=range(0, 55, 5))
avg_c5 = df.groupby("cond5", observed=True)["sellingprice"].mean()
print(avg_c5.round(0))
plt.figure(figsize=(10, 5))
avg_c5.plot(kind="bar", color="mediumpurple")
plt.title("Average Selling Price by Condition Score Range (width 5)")
plt.xlabel("Condition range"); plt.ylabel("Average selling price ($)")
plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig("charts/3_5_price_by_condition5.png", dpi=130); plt.close()

# Extra view: drop the rows scored on the separate 1-5 scale so the trend is clean
tenplus = df[df["condition"] >= 10]
avg_c5b = tenplus.groupby("cond5", observed=True)["sellingprice"].mean().dropna()
print("\nSame chart using only the 10-49 scale:"); print(avg_c5b.round(0))
plt.figure(figsize=(10, 5))
avg_c5b.plot(kind="bar", color="mediumpurple")
plt.title("Average Selling Price by Condition Range (10-49 scale only)")
plt.xlabel("Condition range"); plt.ylabel("Average selling price ($)")
plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig("charts/3_5b_price_by_condition5_clean.png", dpi=130); plt.close()

section("3.6 CARS SOLD BY CONDITION RANGE (size 10)")
df["cond10"] = pd.cut(df["condition"], bins=range(0, 60, 10))
cnt_c10 = df["cond10"].value_counts(sort=False)
print(cnt_c10)
plt.figure(figsize=(10, 5))
cnt_c10.plot(kind="bar", color="goldenrod")
plt.title("Number of Cars Sold by Condition Range (width 10)")
plt.xlabel("Condition range"); plt.ylabel("Cars sold")
plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig("charts/3_6_count_by_condition10.png", dpi=130); plt.close()

section("3.7 BOX PLOT PRICE BY COLOR")
colors_known = df[df["color"] != "unknown"]
order = colors_known["color"].value_counts().index.tolist()
plt.figure(figsize=(13, 6))
sns.boxplot(data=colors_known, x="color", y="sellingprice", order=order)
plt.title("Selling Price by Color (with outliers)")
plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig("charts/3_7a_boxplot_with_outliers.png", dpi=130); plt.close()

def remove_outliers_iqr(data, col):
    q1, q3 = data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return data[(data[col] >= lo) & (data[col] <= hi)], lo, hi

# remove outliers within each color group so each color is judged on its own spread
parts = []
for c, grp in colors_known.groupby("color"):
    cleaned, lo, hi = remove_outliers_iqr(grp, "sellingprice")
    parts.append(cleaned)
no_out = pd.concat(parts)
print(f"Rows before: {len(colors_known):,}  after outlier removal: {len(no_out):,}")
print(no_out.groupby("color")["sellingprice"].median().sort_values(ascending=False).round(0))

plt.figure(figsize=(13, 6))
sns.boxplot(data=no_out, x="color", y="sellingprice", order=order)
plt.title("Selling Price by Color (outliers removed)")
plt.xticks(rotation=45)
plt.tight_layout(); plt.savefig("charts/3_7b_boxplot_no_outliers.png", dpi=130); plt.close()

print("\nDONE. Charts saved in ./charts/")
