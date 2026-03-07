# Question 1: VAR Analysis of High-Frequency Trade Data (Airgas / ARG)

## The Question

We analyze transaction-level data for the stock Airgas (ARG) with three variables:

| Variable | Description |
|---|---|
| `dmidprice` | Change in the midpoint of the bid-ask spread (the "return") |
| `dur` | Time elapsed between the i-th and (i-1)-th trade (duration) |
| `xx` | Trade direction indicator: +1 = buyer-initiated, -1 = seller-initiated |

The economic setup is that each trade decision is made *after* observing the current quotes but *before* observing the direction of trade. If prices contemporaneously affect both trades and duration, but duration only affects trades (not prices directly at the same instant), the causal ordering of shocks is:

**Returns → Durations → Trades**

This ordering is encoded in the Cholesky decomposition used for structural identification.

---

## Theory

### Vector Autoregression (VAR)

A VAR(p) model treats k time series jointly as:

$$Y_t = A_1 Y_{t-1} + A_2 Y_{t-2} + \cdots + A_p Y_{t-p} + u_t$$

where $Y_t = [\text{dmidprice}_t,\ \text{dur}_t,\ \text{xx}_t]'$ and $u_t \sim (0, \Sigma)$ is a vector of reduced-form residuals. The $A_i$ are $3 \times 3$ coefficient matrices capturing how each variable depends on lagged values of all variables.

### Why a VAR for microstructure?

VAR models are a natural fit for market microstructure because:
- Trades, quotes, and time are interdependent — a trade at time t affects prices and the timing of the next trade.
- The reduced-form VAR captures these dynamic feedbacks without imposing a full structural model.
- IRFs from the VAR can reveal the "price impact" of trades — a core quantity in market microstructure theory.

### Structural Identification via Cholesky Decomposition

The reduced-form shocks $u_t$ are correlated. To study the effect of a single "pure" shock, we orthogonalize them using the Cholesky decomposition of $\Sigma$:

$$\Sigma = P P'$$

The structural shocks $\varepsilon_t = P^{-1} u_t$ are uncorrelated by construction. The ordering of variables in $Y_t$ determines who affects whom contemporaneously:
- **dmidprice first**: Returns can contemporaneously affect durations and trades, but not vice versa.
- **dur second**: Durations can contemporaneously affect trade direction, but not returns.
- **xx last**: Trade direction is contemporaneously affected by both returns and durations.

This matches the economic story: the market maker posts quotes (setting the mid-price), the duration elapses, and then a trade arrives with direction xx.

### Impulse Response Functions (IRFs)

The IRF $\Psi_h$ gives the response of each variable at horizon $h$ to a one-standard-deviation structural shock at time 0. For orthogonalized shocks:

$$\Psi_h = \Phi_h P$$

where $\Phi_h$ are the reduced-form moving-average coefficients. We specifically look at:
- Response of `dmidprice` to a shock in `xx` (price impact of trade direction)
- Response of `dmidprice` to a shock in `dur` (price impact of trade timing)

### Cumulative IRF and Permanent Price Impact

The **cumulative IRF** sums the individual-period responses:

$$C_H = \sum_{h=0}^{H} \Psi_h$$

This measures the total accumulated effect over H periods. In the limit ($H \to \infty$), it measures the **permanent price impact** of a shock. This is economically important:
- A **nonzero long-run level** means the shock has a permanent effect — prices incorporate new information permanently. This is the prediction of the Kyle (1985) and Glosten-Milgrom (1985) adverse selection models: informed trades move prices permanently.
- A **cumulative IRF that returns to zero** implies a purely transitory effect (e.g., inventory adjustment or bid-ask bounce with no permanent information).

---

## Solutions

### Part a: VAR(1) Estimation

We estimate a VAR(1) model with the variable ordering `[dmidprice, dur, xx]`. Each equation regresses the current value on one lag of all three variables plus a constant. The model produces:
- 3 equations × (1 intercept + 3 lag coefficients) = 12 slope parameters
- A $3 \times 3$ covariance matrix $\hat{\Sigma}$ of residuals

Key coefficients to examine: does lagged `xx` predict current `dmidprice`? (Cross-equation predictability = Granger causality from trades to returns.)

### Part b: Residual Diagnostics

For a well-specified VAR, residuals should be:
1. **Serially uncorrelated** — ACF of residuals should lie within the confidence bands; the portmanteau (Ljung-Box) test should not reject whiteness.
2. **Homoskedastic** — ACF of squared residuals should show no patterns (though high-frequency financial data often has ARCH effects, so this test frequently fails).

VAR(1) for high-frequency trade data often fails the whiteness test because:
- The dynamics are richer than one lag can capture.
- The `xx` variable is a binary/ternary indicator, so normal residual assumptions break down badly.
- ARCH/GARCH effects are common in high-frequency data.

### Part c: Improved Model via Lag Selection

We use AIC/BIC to select the lag order. In practice, AIC tends to select more lags than BIC (AIC penalizes less). A higher lag order captures more of the short-run autocorrelation in trade sequences.

**Why the xx equation is hard to fit:**
The trade direction indicator takes values in {-1, +1}. A VAR assumes linear dynamics with Gaussian errors, but the true DGP for xx is discrete. Even a well-fitted VAR will have residuals with poor distributional properties for this equation. Alternative models (probit VAR, logistic models) would be more appropriate but are beyond a simple VAR framework.

### Part d: IRF — Price Impact of Trade Direction

The orthogonalized IRF of `dmidprice` to a one-standard-deviation shock in `xx` answers:
*"If a buyer-initiated trade arrives unexpectedly, how does the mid-price respond over time?"*

**Expected finding:**
- **Immediate positive response** at period 0 or 1: A buyer-initiated trade signals that the buyer believes the stock is undervalued. The market maker, fearing adverse selection, raises both bid and ask quotes, moving the midpoint up.
- **Partial reversal** in subsequent periods: Some of the initial price impact is reversed as the market maker adjusts for inventory effects (the bounce component).
- **Remaining permanent level**: The part of the price impact that does not reverse is the information component — the trade revealed genuine private information.

The dur IRF is typically smaller or less clear-cut. Longer durations might signal lower information flow, potentially pulling quotes slightly back toward fundamental value, but the effect is ambiguous and often statistically insignificant.

### Part e: Cumulative IRF — Permanent Price Impact

The cumulative IRF of `dmidprice` to a shock in `xx` converges to the **total permanent price impact of a trade**.

**Interpretation:**
- If the cumulative IRF stabilizes at a positive value (say, 0.02 for a +1 shock in xx): a buyer-initiated trade permanently raises the mid-price by 2 cents, on average. This is the information content of the trade.
- The **speed of convergence** tells us how quickly the market incorporates the information: a fast-converging cumulative IRF means information is quickly impounded; a slow convergence implies gradual learning or inventory adjustments that take many trades to resolve.
- The fraction of the total price impact that is permanent (cumulative IRF at infinity / IRF at period 0) is called the **permanent component** and is related to the adverse selection component of the bid-ask spread in models like Glosten (1987).

This decomposition — transitory vs. permanent price impact — is one of the central empirical quantities in market microstructure research.

---

## Key References

- **Kyle (1985)**: Strategic trading by an informed trader moves prices permanently.
- **Glosten & Milgrom (1985)**: Adverse selection by informed traders widens the bid-ask spread and causes permanent price impact.
- **Hasbrouck (1991)**: Empirically decomposes trade price impact using VAR methods — exactly the approach used here.
- **Engle & Russell (1998)**: Autoregressive Conditional Duration (ACD) model for the `dur` variable, a natural extension.
