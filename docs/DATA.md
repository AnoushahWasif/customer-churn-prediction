# Data dictionary

**Source:** IBM Telco Customer Churn sample, [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d), pinned to commit `1fd6fd7`. It's a fictional telecom company, so there's no real personal data.

**Size:** 7,043 customers, 21 columns. **Target:** `Churn`, where 1,869 customers (26.5%) left in the last month.

`make data` downloads the file to `data/raw/telco_churn.csv` and checks its SHA-256. `churn.data.load()` returns it with the fixes below applied.

## Columns

| Column | Type | Values | Notes |
|--------|------|--------|-------|
| `customerID` | id | `7590-VHVEG` … | Dropped on load. It's an identifier with no signal |
| `gender` | category | Male 3,555 · Female 3,488 | |
| `SeniorCitizen` | 0/1 | 16.2% are 1 | Stored as 0/1, unlike the Yes/No columns |
| `Partner` | category | Yes · No | |
| `Dependents` | category | Yes · No | |
| `tenure` | int, months | 0–72, median 29 | 0 = joined this month |
| `PhoneService` | category | Yes 6,361 · No 682 | |
| `MultipleLines` | category | Yes · No · No phone service | "No phone service" repeats `PhoneService = No` |
| `InternetService` | category | Fiber optic 3,096 · DSL 2,421 · No 1,526 | |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | category | Yes · No · No internet service | "No internet service" (1,526 rows each) repeats `InternetService = No` |
| `Contract` | category | Month-to-month 3,875 · One year 1,473 · Two year 1,695 | |
| `PaperlessBilling` | category | Yes · No | |
| `PaymentMethod` | category | Electronic check 2,365 · Mailed check 1,612 · Bank transfer (automatic) 1,544 · Credit card (automatic) 1,522 | |
| `MonthlyCharges` | float, $ | 18.25–118.75, median 70.35 | |
| `TotalCharges` | float, $ | 0–8,684.80, median 1,394.55 | Read as text in the raw file because of blanks (see below) |
| `Churn` | target | Yes → 1, No → 0 | |

## Quirks and how they're handled

- **`TotalCharges` blanks.** 11 rows have a single space instead of a number, so pandas reads the whole column as text. All 11 have `tenure = 0`: brand-new customers who haven't been billed. They're set to 0, which is the true value, not imputed.
- **Repeated "No service" values.** "No internet service" in six columns and "No phone service" in one say the same thing as `InternetService` / `PhoneService`. They're kept as their own category for now. Whether to merge them into "No" gets decided in preprocessing (Day 4).
- **`TotalCharges` ≈ `tenure × MonthlyCharges`.** It's mostly derived from the other two, so expect strong correlation. This matters for reading logistic regression coefficients later.
