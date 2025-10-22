"""
titanic_eda.py
Exploratory Data Analysis (EDA) script for the Titanic dataset.
Author: Generated for user
Requirements:
    pip install pandas numpy matplotlib seaborn plotly statsmodels scikit-learn
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional interactive plotting (works if you want HTML interactive charts)
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# For VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder

# -------------------------
# User settings
# -------------------------
# Update this path if your CSV is somewhere else:
file_path = "/mnt/data/Titanic-Dataset (3).csv"

# Where to save output plots
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Limit pairplot size (pairplot can be slow for many rows; sampling keeps responsiveness)
PAIRPLOT_SAMPLE = 500  # set to None to use full dataset

# -------------------------
# Utility functions
# -------------------------
def save_fig(fig, fname, dpi=150, tight=True):
    path = output_dir / fname
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    print(f"Saved: {path}")

def show_and_save(fig, fname, dpi=150):
    save_fig(fig, fname, dpi=dpi)
    plt.close(fig)

# -------------------------
# Load data
# -------------------------
if not Path(file_path).exists():
    print(f"ERROR: file not found: {file_path}")
    sys.exit(1)

df = pd.read_csv(file_path)
pd.set_option("display.max_columns", None)
print("\n--- Dataset loaded ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.head().T)

# -------------------------
# Basic info & summary stats
# -------------------------
print("\n--- dtypes & non-null counts ---")
print(df.info())

print("\n--- Summary statistics (numeric) ---")
print(df.describe().T)

print("\n--- Summary statistics (object/categorical) ---")
print(df.describe(include=["object", "category"]).T)

# Missing values
missing = df.isna().sum().sort_values(ascending=False)
print("\n--- Missing values (count) ---")
print(missing[missing > 0])

# Percentage missing
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
print("\n--- Missing values (%) ---")
print(missing_pct[missing_pct > 0].round(2))

# -------------------------
# Data cleaning for EDA (non-destructive)
# -------------------------
df_eda = df.copy()

# Common Titanic columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
# Convert certain types if needed:
if "Survived" in df_eda.columns:
    df_eda["Survived"] = df_eda["Survived"].astype("Int64")  # keep NA if any

# Fill or mark missing Embarked with 'Missing'
if "Embarked" in df_eda.columns:
    df_eda["Embarked"] = df_eda["Embarked"].astype("object")
    df_eda["Embarked"] = df_eda["Embarked"].fillna("Missing")

# Simple feature extraction: Title from Name (if present)
if "Name" in df_eda.columns:
    df_eda["Title"] = df_eda["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    # group rare titles
    title_counts = df_eda["Title"].value_counts()
    rare_titles = title_counts[title_counts < 10].index
    df_eda["Title"] = df_eda["Title"].replace(rare_titles, "Rare")

# Create Age group
if "Age" in df_eda.columns:
    df_eda["Age_group"] = pd.cut(df_eda["Age"], bins=[-1, 0, 12, 18, 35, 60, 200],
                                 labels=["Unknown/0","Child","Teen","YoungAdult","Adult","Senior"])

# -------------------------
# Numeric feature EDA: histograms, boxplots, skewness
# -------------------------
numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns found:", numeric_cols)

# Summary stats: mean, median, std, skew
num_stats = df_eda[numeric_cols].agg(["count","mean","median","std","min","max","skew"]).T
print("\n--- Numeric summary with skewness ---")
print(num_stats.round(3))

# Histograms
for col in numeric_cols:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df_eda[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Histogram: {col}")
    save_fig(fig, f"hist_{col}.png")
    plt.close(fig)

# Boxplots (global) and by Survived if Survived exists
for col in numeric_cols:
    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x=df_eda[col], ax=ax)
    ax.set_title(f"Boxplot: {col}")
    save_fig(fig, f"box_{col}.png")
    plt.close(fig)

    if "Survived" in df_eda.columns:
        fig, ax = plt.subplots(figsize=(7,4))
        sns.boxplot(x="Survived", y=col, data=df_eda, ax=ax)
        ax.set_title(f"Boxplot of {col} by Survived")
        save_fig(fig, f"box_{col}_by_survived.png")
        plt.close(fig)

# -------------------------
# Categorical feature EDA: counts and barplots
# -------------------------
cat_cols = df_eda.select_dtypes(include=["object","category"]).columns.tolist()
print("\nCategorical columns found:", cat_cols)

for col in cat_cols:
    counts = df_eda[col].value_counts(dropna=False)
    print(f"\n--- Value counts for {col} ---")
    print(counts.head(20))
    # barplot (top categories)
    top = counts.iloc[:12]
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title(f"Top categories in {col}")
    save_fig(fig, f"bar_{col}.png")
    plt.close(fig)

# If Survived exists: survival rates by categories
if "Survived" in df_eda.columns:
    for col in ["Sex","Pclass","Embarked","Title","Age_group"]:
        if col in df_eda.columns:
            surv_rate = df_eda.groupby(col)["Survived"].mean().sort_values(ascending=False)
            print(f"\nSurvival rate by {col}:\n", surv_rate.round(3))
            fig, ax = plt.subplots(figsize=(7,4))
            surv_rate.plot(kind="barh", ax=ax)
            ax.set_xlabel("Survival rate")
            ax.set_title(f"Survival rate by {col}")
            save_fig(fig, f"survival_rate_by_{col}.png")
            plt.close(fig)

# -------------------------
# Pairplot & Correlation matrix
# -------------------------
# Pairplot: choose a subset of numeric features to keep it readable
pairplot_cols = numeric_cols.copy()
# remove many-id-like columns if present
for c in ["PassengerId", "Ticket", "Fare"]:
    if c in pairplot_cols and len(pairplot_cols) > 3:
        pairplot_cols.remove(c)

print("\nUsing columns for pairplot:", pairplot_cols)
try:
    sample_df = df_eda[pairplot_cols].dropna()
    if PAIRPLOT_SAMPLE and len(sample_df) > PAIRPLOT_SAMPLE:
        sample_df = sample_df.sample(PAIRPLOT_SAMPLE, random_state=42)
    sns.pairplot(sample_df)
    fig = plt.gcf()
    save_fig(fig, "pairplot_numeric.png")
    plt.close(fig)
except Exception as e:
    print("Pairplot error (maybe too many columns/NaNs):", e)

# Correlation matrix (Pearson) for numeric variables
corr = df_eda[numeric_cols].corr()
print("\n--- Correlation matrix (numeric) ---")
print(corr.round(3))

fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
ax.set_title("Correlation matrix (numeric)")
save_fig(fig, "correlation_matrix.png")
plt.close(fig)

# -------------------------
# Skewness detection (numeric)
# -------------------------
skewness = df_eda[numeric_cols].skew().sort_values(ascending=False)
print("\n--- Skewness (numeric) ---")
print(skewness)
# Note: positive skew => long right tail; negative skew => long left tail

# -------------------------
# Multicollinearity check: VIF
# -------------------------
# Prepare numeric-only DataFrame for VIF; drop rows with NA in selected numeric columns
vif_df = df_eda[numeric_cols].dropna()
if vif_df.shape[0] > 0 and vif_df.shape[1] > 1:
    # add small constant to avoid division by zero issues
    X = vif_df.copy()
    # If any constant columns exist, drop them
    X = X.loc[:, X.apply(pd.Series.nunique) > 1]
    # compute VIF for each column
    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception as e:
            vif = np.nan
        vif_data.append((col, vif))
    vif_table = pd.DataFrame(vif_data, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
    print("\n--- VIF table ---")
    print(vif_table)
else:
    print("\nNot enough numeric data to compute VIF.")

# -------------------------
# Quick inferences & anomaly checks (printed)
# -------------------------
print("\n--- Quick inferences & anomalies ---")
# Example automatic insights:
if "Survived" in df_eda.columns and "Sex" in df_eda.columns:
    # survival by sex
    surv_by_sex = df_eda.groupby("Sex")["Survived"].mean().sort_values(ascending=False)
    print("Survival by Sex:\n", surv_by_sex.round(3))

if "Survived" in df_eda.columns and "Pclass" in df_eda.columns:
    surv_by_pclass = df_eda.groupby("Pclass")["Survived"].mean().sort_values(ascending=False)
    print("Survival by Pclass:\n", surv_by_pclass.round(3))

# fare anomalies (very high fares)
if "Fare" in df_eda.columns:
    high_fares = df_eda["Fare"].nlargest(5)
    print("Top 5 fares:\n", high_fares)

# missing age patterns
if "Age" in df_eda.columns and "Pclass" in df_eda.columns:
    missing_age_by_pclass = df_eda[df_eda["Age"].isna()].groupby("Pclass").size()
    print("Missing Age counts by Pclass:\n", missing_age_by_pclass)

# Detect duplicates if ticket or name duplicates
if "Name" in df_eda.columns:
    dup_names = df_eda["Name"].duplicated().sum()
    print(f"Duplicate Name rows: {dup_names}")
if "Ticket" in df_eda.columns:
    dup_tickets = df_eda["Ticket"].duplicated().sum()
    print(f"Duplicate Ticket rows: {dup_tickets}")

# -------------------------
# Optional: interactive Plotly checks (if plotly installed)
# -------------------------
if PLOTLY_AVAILABLE:
    try:
        # Survival by Pclass and Sex
        if {"Pclass","Sex","Survived"}.issubset(df_eda.columns):
            fig = px.histogram(df_eda, x="Pclass", color="Sex", facet_col="Survived",
                               title="Count by Pclass, Sex and Survived")
            outname = output_dir / "plotly_pclass_sex_survived.html"
            fig.write_html(outname)
            print(f"Saved interactive plot: {outname}")
    except Exception as e:
        print("Plotly demonstration failed:", e)

# -------------------------
# Save a short EDA report CSVs
# -------------------------
try:
    num_stats.to_csv(output_dir / "numeric_summary.csv")
    vif_table.to_csv(output_dir / "vif_table.csv") if 'vif_table' in locals() else None
    print(f"Saved numeric summary and VIF (if computed) to {output_dir}")
except Exception as e:
    print("Error saving CSV summaries:", e)

print("\nEDA complete. Check the 'plots/' directory for generated figures and the console output for printed summaries.")
