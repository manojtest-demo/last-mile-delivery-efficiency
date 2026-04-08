# =============================================================================
# PROJECT   : Evolution of Last-Mile Delivery Efficiency & E-Commerce Logistics
# DATA      : World Bank LPI
# TARGET    : LMEI = Mean(Timeliness, Tracking & Tracing, Logistics Competence)
# MODEL     : Weighted Polynomial Regression (degree 2)
# OUTPUTS   : CSVs + Plots saved to analysis_artifacts/
# =============================================================================

# ------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------------
OUTPUT = "analysis_artifacts"
os.makedirs(OUTPUT, exist_ok=True)

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/manojtest-demo/last-mile-delivery-efficiency/main/Combined_LPI_Data.csv"
)

# Remove Rank columns
rank_cols = [c for c in df.columns if 'Rank' in c]
df.drop(columns=rank_cols, inplace=True)

# Identify score columns
score_cols = [c for c in df.columns if 'Score' in c]

# Fill missing values using country-wise mean
df[score_cols] = df.groupby('Country')[score_cols].transform(lambda x: x.fillna(x.mean()))

# Clip outliers using IQR
for col in score_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# ------------------------------------------------------------------
# TARGET VARIABLE
# ------------------------------------------------------------------
LMEI_DIMS = [
    'Timeliness Score',
    'Tracking & tracing Score',
    'Logistics competence Score'
]

df["LMEI"] = df[LMEI_DIMS].mean(axis=1)

df.to_csv(f"{OUTPUT}/cleaned_lpi_data.csv", index=False)

countries = df["Country"].unique().tolist()

# ------------------------------------------------------------------
# INTERPOLATION (VISUAL ONLY)
# ------------------------------------------------------------------
def interpolate_country(cdf):
    cdf = cdf.sort_values("Year")
    years_known = cdf["Year"].values
    years_full = np.arange(2007, 2024)

    result = {}
    for col in score_cols + ["LMEI"]:
        cs = CubicSpline(years_known, cdf[col].values)
        result[col] = np.clip(cs(years_full), 1.0, 5.0)

    out = pd.DataFrame(result)
    out.insert(0, "Year", years_full)
    out.insert(0, "Country", cdf["Country"].iloc[0])
    return out

df_interp = pd.concat(
    [interpolate_country(df[df["Country"] == c]) for c in countries],
    ignore_index=True
)

df_interp.to_csv(f"{OUTPUT}/interpolated_lmei.csv", index=False)

# ------------------------------------------------------------------
# STYLE
# ------------------------------------------------------------------
PALETTE = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
           '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4']

CMAP = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(countries)}

plt.rcParams.update({
    'figure.facecolor': '#141422',
    'axes.facecolor': '#1c1c32',
    'axes.edgecolor': '#555',
    'axes.labelcolor': 'white',
    'xtick.color': '#ccc',
    'ytick.color': '#ccc',
    'text.color': 'white',
    'grid.color': '#2a2a4a',
})

# ------------------------------------------------------------------
# PLOT: HISTORICAL TRENDS
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 5, figsize=(20, 11))
fig.suptitle('LMEI Historical Trend per Country (2007-2023)', fontsize=14)

for idx, country in enumerate(countries):
    ax = axes[idx//5, idx%5]

    col = CMAP[country]
    h = df_interp[df_interp["Country"] == country].sort_values("Year")
    r = df[df["Country"] == country].sort_values("Year")

    ax.plot(h["Year"], h["LMEI"], color=col, lw=2)
    ax.scatter(r["Year"], r["LMEI"], color='white', s=25, edgecolors=col)

    ax.set_title(country, fontsize=9)
    ax.grid()

plt.tight_layout()
fig.savefig(f"{OUTPUT}/plot4_historical_trends.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
def weighted_poly2_model(X, y, weights=None):
    poly = PolynomialFeatures(degree=2, include_bias=True)
    Xp = poly.fit_transform(X)
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xp, y, sample_weight=weights)

    class Model:
        def __init__(self, poly, lr):
            self.poly = poly
            self.lr = lr

        def predict(self, X_new):
            return self.lr.predict(self.poly.transform(X_new))

    return Model(poly, lr)

# ------------------------------------------------------------------
# FORECAST
# ------------------------------------------------------------------
FORECAST_YEARS = np.arange(2024, 2029)
forecast_rows = []

for c in countries:
    raw = df[df["Country"] == c].sort_values("Year")

    X = raw["Year"].values.reshape(-1, 1)
    y = raw["LMEI"].values

    weights = np.arange(1, len(y) + 1)

    mdl = weighted_poly2_model(X, y, weights=weights)

    y_pred = mdl.predict(X)
    ci = 1.5 * np.std(y - y_pred)

    y_fore = np.clip(mdl.predict(FORECAST_YEARS.reshape(-1, 1)), 1.0, 5.0)

    for yr, val in zip(FORECAST_YEARS, y_fore):
        forecast_rows.append({
            "Country": c,
            "Year": int(yr),
            "Forecast": float(val),
            "Lower": float(val - ci),
            "Upper": float(val + ci)
        })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(f"{OUTPUT}/forecast_results.csv", index=False)

print("Pipeline executed successfully.")