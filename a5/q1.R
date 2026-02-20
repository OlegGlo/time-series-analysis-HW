# Load libraries
library(rugarch)  # GARCH modeling
library(xts)      # time-series objects with date indices

# --- Load data ---
# read.csv() reads a CSV file into a data frame (like a table)
df <- read.csv("VIX_SP.csv")

# Convert the DATE column from text to actual R Date objects
# format tells R how the dates are written in the file: month/day/year
df$DATE <- as.Date(df$DATE, format = "%m/%d/%Y")

# Sort by date (just in case the file isn't already sorted)
df <- df[order(df$DATE), ]

# --- Plot the data ---
# par(mfrow=c(1,2)) splits the plot window into 1 row, 2 columns (side by side)
par(mfrow = c(1, 2))
plot(df$DATE, df$VIX, type = "l", col = "blue",
     main = "VIX Level", xlab = "Date", ylab = "VIX")
plot(df$DATE, df$ret, type = "l", col = "darkred",
     main = "S&P 500 Returns", xlab = "Date", ylab = "Return")
par(mfrow = c(1, 1))  # reset to single plot

# --- Prepare returns as a time-series object ---
# xts() creates a time-series: first arg is the data, order.by is the dates
# We multiply returns by 100 to work in percentage (helps GARCH numerics)
ret <- xts(df$ret * 100, order.by = df$DATE)
ret <- na.omit(ret)  # drop any missing values

# ============================================================
# Q1A: Plain GARCH(1,1)
# Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
# ============================================================
spec11 <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)
fit11 <- ugarchfit(spec11, data = ret)
print(fit11)

# Squared standardized residuals diagnostics
std_resid <- residuals(fit11, standardize = TRUE)
std_resid_sq <- std_resid^2

par(mfrow = c(1, 2))
acf(as.numeric(std_resid_sq), lag.max = 20, main = "ACF — Sq Std Resid GARCH(1,1)")
pacf(as.numeric(std_resid_sq), lag.max = 20, main = "PACF — Sq Std Resid GARCH(1,1)")
par(mfrow = c(1, 1))

# ============================================================
# Q1B: GARCH(1,1)-X with Monday dummy in variance equation
# Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + δ·Monday_t
# ============================================================

# --- Create Monday dummy variable ---
# format(..., "%u") extracts day-of-week as a number: 1=Monday, ..., 7=Sunday
# We compare to "1" to get TRUE/FALSE, then as.numeric converts to 1/0
monday <- as.numeric(format(index(ret), "%u") == "1")

# Wrap it in an xts object with the same dates as returns
monday_xts <- xts(monday, order.by = index(ret))

# --- Specify the GARCH-X model ---
# ugarchspec() defines the model BEFORE fitting it
spec_x <- ugarchspec(
  variance.model = list(
    model    = "sGARCH",       # standard GARCH
    garchOrder = c(1, 1),      # GARCH(1,1): 1 ARCH lag, 1 GARCH lag
    # external.regressors puts the Monday dummy INTO the variance equation
    external.regressors = as.matrix(monday_xts)
  ),
  mean.model = list(
    armaOrder    = c(0, 0),    # no AR or MA terms in the mean
    include.mean = TRUE        # include a constant (mu) in the mean
  ),
  distribution.model = "norm"  # normal distribution for errors
)

# By default rugarch constrains variance params >= 0 to keep variance positive.
# But the Monday dummy CAN have a negative effect, so we allow it.
setbounds(spec_x) <- list(vxreg1 = c(-5, 5))

# --- Fit the model to data ---
# ugarchfit() estimates all parameters by maximum likelihood
fit_x <- ugarchfit(spec_x, data = ret)

# Print the full summary: coefficients, std errors, t-stats, p-values
print(fit_x)

# --- Diagnostic: squared standardized residuals ---
# If the model is correct, these should look like white noise (no patterns)
# residuals(..., standardize=TRUE) gives ε_t / σ_t
std_resid_x    <- residuals(fit_x, standardize = TRUE)
std_resid_x_sq <- std_resid_x^2  # square them to check for remaining volatility clustering

# Plot ACF and PACF — bars outside the blue dashed lines indicate problems
par(mfrow = c(1, 2))
acf(as.numeric(std_resid_x_sq), lag.max = 20,
    main = "ACF of Sq Std Residuals — GARCH-X")
pacf(as.numeric(std_resid_x_sq), lag.max = 20,
     main = "PACF of Sq Std Residuals — GARCH-X")
par(mfrow = c(1, 1))

# ============================================================
# Q1C: GARCH(1,1)-X with Monday AND Friday dummies
# Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + δ₁·Monday_t + δ₂·Friday_t
# ============================================================

# --- Create Friday dummy variable ---
# Day-of-week 5 = Friday
friday <- as.numeric(format(index(ret), "%u") == "5")
friday_xts <- xts(friday, order.by = index(ret))

# Combine Monday and Friday into a matrix with 2 columns
# external.regressors expects a matrix where each column is one regressor
dummies <- merge(monday_xts, friday_xts)
colnames(dummies) <- c("Monday", "Friday")

# --- Specify and fit the GARCH-X model with both dummies ---
spec_mf <- ugarchspec(
  variance.model = list(
    model    = "sGARCH",
    garchOrder = c(1, 1),
    external.regressors = as.matrix(dummies)
  ),
  mean.model = list(
    armaOrder    = c(0, 0),
    include.mean = TRUE
  ),
  distribution.model = "norm"
)

# Allow negative coefficients on day-of-week dummies
setbounds(spec_mf) <- list(vxreg1 = c(-5, 5), vxreg2 = c(-5, 5))

fit_mf <- ugarchfit(spec_mf, data = ret)
print(fit_mf)

# --- Diagnostics ---
std_resid_mf    <- residuals(fit_mf, standardize = TRUE)
std_resid_mf_sq <- std_resid_mf^2

par(mfrow = c(1, 2))
acf(as.numeric(std_resid_mf_sq), lag.max = 20,
    main = "ACF of Sq Std Residuals — GARCH-X (Mon+Fri)")
pacf(as.numeric(std_resid_mf_sq), lag.max = 20,
     main = "PACF of Sq Std Residuals — GARCH-X (Mon+Fri)")
par(mfrow = c(1, 1))

# ======================================z======================
# Q1D: GARCH(1,1)-X with lagged VIX daily variance
# Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + δ·VIXvar_{t-1}
# ============================================================

# --- Convert VIX to daily variance ---
# VIX is annualized implied volatility in percentage points (e.g. 16 = 16%).
# To get daily variance in pct^2 (matching our ret*100 scale):
#   daily_var = VIX^2 / 252
vix <- xts(df$VIX, order.by = df$DATE)
vix <- vix[index(ret)]                   # align with returns
vix_daily_var <- vix^2 / 252             # daily variance in pct^2

# --- Lag by 1 day ---
# We must use YESTERDAY's VIX to forecast TODAY's variance (no look-ahead).
# lag(x, k=1) in xts shifts the series forward by 1 period.
vix_var_lag <- lag(vix_daily_var, k = 1)

# Drop the first observation (NA from lagging) and align returns
common_idx <- index(na.omit(vix_var_lag))
ret_d      <- ret[common_idx]
vix_var_d  <- vix_var_lag[common_idx]

# --- Specify and fit ---
spec_vix <- ugarchspec(
  variance.model = list(
    model    = "sGARCH",
    garchOrder = c(1, 1),
    external.regressors = as.matrix(vix_var_d)
  ),
  mean.model = list(
    armaOrder    = c(0, 0),
    include.mean = TRUE
  ),
  distribution.model = "norm"
)

# Allow negative coefficient on VIX variance regressor
setbounds(spec_vix) <- list(vxreg1 = c(-5, 5))

fit_vix <- ugarchfit(spec_vix, data = ret_d)
print(fit_vix)

# --- Diagnostics ---
std_resid_vix    <- residuals(fit_vix, standardize = TRUE)
std_resid_vix_sq <- std_resid_vix^2

par(mfrow = c(1, 2))
acf(as.numeric(std_resid_vix_sq), lag.max = 20,
    main = "ACF of Sq Std Residuals — GARCH-X (VIX)")
pacf(as.numeric(std_resid_vix_sq), lag.max = 20,
     main = "PACF of Sq Std Residuals — GARCH-X (VIX)")
par(mfrow = c(1, 1))

Box.test(as.numeric(std_resid_vix_sq), lag = 10, type = "Ljung-Box")
Box.test(as.numeric(std_resid_vix_sq), lag = 20, type = "Ljung-Box")

# --- Interpretation ---
coefs_vix <- coef(fit_vix)
cat("\n===== Lagged VIX Daily Variance Effect =====\n")
cat(sprintf("VIX variance coef (vxreg1): %.6f\n", coefs_vix["vxreg1"]))
cat(sprintf("alpha1: %.6f\n", coefs_vix["alpha1"]))
cat(sprintf("beta1:  %.6f\n", coefs_vix["beta1"]))


