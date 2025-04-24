import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Qt5Agg")

PRODUCT = "23C"

# %%

excel_file = pd.read_excel("./Hackaton DB Final 04.21.xlsx", sheet_name=None)
supply_demand_page = excel_file["Supply_Demand"]

demand: pd.DataFrame = (
    supply_demand_page.iloc[[3, 9, 15]]
    .drop(["Attribute"], axis=1)
    .set_index("Product ID")
    .transpose()
)
sum = demand.sum(axis=1)

# %%

demand.plot()
sum.plot(
    linestyle="--",
)
plt.show()

# %%

weekly_ratios = excel_file["Weekly Demand Ratio"].transpose()
weekly_ratios

weekly_conversion = []
for quarter, row in demand.iterrows():
    for week, ratio in weekly_ratios.iterrows():
        weekly_row = row * ratio.values
        weekly_row.name = f"{quarter} {week}"
        weekly_conversion.append(weekly_row)
weekly_demand = pd.DataFrame(weekly_conversion)
weekly_sum = weekly_demand.sum(axis=1)

# %%

weekly_demand.plot()
weekly_sum.plot(linestyle="--")
plt.show()


# %%

from statsmodels.tsa.seasonal import seasonal_decompose

demand_adecomposition = [
    seasonal_decompose(
        demand[product_id],
        model="additive",
        period=4,
        extrapolate_trend="freq",
    )
    for product_id in weekly_demand.columns
    if product_id != "Total Demand"
]

demand_adecomposition[0].plot()
plt.show()

# %%

n = len(weekly_demand)
train = weekly_demand.iloc[: int(n * 0.8)]
test = weekly_demand.iloc[int(n * 0.8) :]

# %%

plt.plot(train[PRODUCT])
plt.plot(test[PRODUCT])
plt.show()

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

ya = train[PRODUCT]
ya_test = test[PRODUCT]

# auto_model = auto_arima(
#     ya,
#     seasonal=True,       # Try with False if you don’t expect seasonality
#     m=52,                # Set to 12 for monthly, 4 for quarterly, 52 for weekly (depends on your data)
#     trace=True,
#     suppress_warnings=True,
#     stepwise=True
# )
#
# print(auto_model.summary())
#
# order = auto_model.order
# seasonal_order = auto_model.seasonal_order

# Apply to your train and test sets
train_weeks = pd.Series([x.endswith("12") for x in train.index])
test_weeks = pd.Series([x.endswith("12") for x in test.index])

# Create spike week binary flag (spike at week 11)
train_spike_flag = (train_weeks).astype(int).values.reshape(-1, 1)
test_spike_flag = (test_weeks).astype(int).values.reshape(-1, 1)

order = (3, 0, 52)
seasonal_order = (2, 0, 0, 52)

from statsmodels.tsa.statespace.sarimax import SARIMAX

sarimax_model = SARIMAX(
    train[PRODUCT], order=order, seasonal_order=seasonal_order, exog=train_spike_flag
)

sarimax_fit = sarimax_model.fit(disp=False)

y_pred = sarimax_fit.get_forecast(steps=len(ya_test), exog=test_spike_flag)
y_pred_df = y_pred.conf_int()
y_pred_df["Predictions"] = y_pred.predicted_mean
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"]

rmse = np.sqrt(mean_squared_error(test[PRODUCT].values, y_pred_out))
print("RMSE: ", rmse)

plt.figure(figsize=(16, 6))
plt.plot(train[PRODUCT], label="Train")
plt.plot(test[PRODUCT], label="Test")
plt.plot(y_pred_out, color="yellow", label="Forecast")
plt.legend()
plt.show()

# %%

# froecast ahead of current data

weekly_demand

last: str = test.index[-1]
quarter, year, week = last.split(" ")
quarter_num = int(quarter[1:])
year_num = int(year)
week_num = int(week[2:])

week_num
year_num

next_n = 52 * 3

period = []
for i, nn in enumerate(range(next_n)):
    week_num += 1

    if week_num > 13:
        quarter_num += 1
        week_num = 1
        if quarter_num > 4:
            year_num += 1
            quarter_num = 1
    print(year_num)

    period.append(f"Q{quarter_num} {year_num:02} Wk{week_num}")

period


forecast_weeks = pd.Series([x.endswith("12") for x in period])
forecast_spike_flag = (forecast_weeks).astype(int).values.reshape(-1, 1)

y_pred = sarimax_fit.get_forecast(steps=next_n, exog=forecast_spike_flag)
y_pred_df = y_pred.conf_int()
y_pred_df["Predictions"] = y_pred.predicted_mean
y_pred_df.index = period
y_pred_out = y_pred_df["Predictions"]

plt.figure(figsize=(16, 6))
plt.plot(train[PRODUCT], label="Train")
plt.plot(test[PRODUCT], label="Test")
plt.plot(y_pred_out, color="yellow", label="Forecast")
plt.legend()
plt.show()

# %%

test.index
period

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

custom_quarter_starts = {
    "Q1": (3, 9),  # March
    "Q2": (6, 3),  # June 3
    "Q3": (9, 3),  # September 3
    "Q4": (12, 2),  # December 2
}


def parse_custom_date(label: str) -> pd.Timestamp:
    parts = label.split()
    quarter = parts[0]
    year_suffix = int(parts[1])
    year = 1900 + year_suffix if year_suffix > 50 else 2000 + year_suffix
    week = int(parts[2][2:])

    month, day = custom_quarter_starts[quarter]
    start_date = pd.Timestamp(year=year, month=month, day=day)
    return start_date + pd.DateOffset(weeks=week - 1)


df_train = train[PRODUCT].reset_index()
df_train.columns = ["period", "y"]
df_train["ds"] = df_train["period"].apply(parse_custom_date)
df_train = df_train[["ds", "y"]]

df_test = test[PRODUCT].reset_index()
df_test.columns = ["period", "y"]
df_test["ds"] = df_test["period"].apply(parse_custom_date)
df_test = df_test[["ds", "y"]]

spike_weeks = train.index.to_series().apply(lambda x: int(x.split()[2][2:]) == 11)
spike_dates = df_train.loc[spike_weeks.values, "ds"]

holidays_df = pd.DataFrame(
    {"holiday": "spike", "ds": spike_dates, "lower_window": 0, "upper_window": 0}
)


# Convert to datetime
forecast_ds = [parse_custom_date(p) for p in period]

model = Prophet(holidays=holidays_df)
model.fit(df_train)

future = pd.DataFrame({"ds": forecast_ds})
forecast = model.predict(future)

forecast


# %%


def remap_quarter_weeks_to_absolute_year_weeks(
    periods: list[str], base="Q3 95 Wk1", base_week=9
) -> list[str]:
    """
    Converts 'Qx YY WkN' (week-in-quarter) to 'Qx YY WW_nn' (week-in-year),
    starting from a base period that defines the absolute week number.

    Example:
        remap_quarter_weeks_to_absolute_year_weeks(["Q3 95 Wk1", "Q1 96 Wk1"])
        → ["Q3 95 WW_09", "Q1 96 WW_35"]
    """

    # Week offsets of each quarter within a year (no gaps)
    quarter_offsets = {
        "Q1": 0,
        "Q2": 13,
        "Q3": 26,
        "Q4": 39,
    }

    # Parse the base reference (e.g. "Q3 95 Wk1")
    base_q, base_y, base_w = base.split()
    base_y = int(base_y)
    base_y_full = 1900 + base_y if base_y >= 50 else 2000 + base_y
    base_q_offset = quarter_offsets[base_q]
    base_wq = int(base_w[2:])  # 'Wk1' → 1
    base_absolute = base_y_full * 52 + base_q_offset + base_wq - 1

    result = []
    for period in periods:
        q, y, w = period.split()
        y = int(y)
        y_full = 1900 + y if y >= 50 else 2000 + y
        wq = int(w[2:])

        current_absolute = y_full * 52 + quarter_offsets[q] + wq - 1
        relative_week = (current_absolute - base_absolute + base_week) % 52 + 1
        result.append(f"{q} {str(y).zfill(2)} WW_{relative_week:02}")

    return result


# $$

all_labels = remap_quarter_weeks_to_absolute_year_weeks(
    list(train.index) + list(test.index) + period
)
train_labels = all_labels[0 : len(train.index)]
test_labels = all_labels[len(train.index) : len(train.index) + len(test.index)]
period_labels = all_labels[len(train.index) + len(test.index) :]

len(train_labels)
len(train.index)
len(test_labels)
len(test.index)

plt.figure(figsize=(12, 5))
plt.plot(train_labels, df_train["y"], label="Train")
plt.plot(test_labels, df_test["y"], label="Test")
plt.plot(period_labels, forecast["yhat"], label="Forecast", color="orange")
# plt.gca().set_xticks(period)
plt.xticks(all_labels, rotation=45, ha="right")
plt.legend()
plt.title("Prophet Forecast with Custom Period Array")
plt.tight_layout()
plt.show()

pd.DataFrame([period_labels, forecast["yhat"]]).to_csv(f"prophet{PRODUCT}.csv")


len(forecast["yhat"])
len(period_labels)
