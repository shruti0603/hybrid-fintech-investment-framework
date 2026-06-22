import yfinance as yf

ticker = "GOOGL"

stock = yf.Ticker(ticker)

info = stock.info
cashflow = stock.cashflow
balance_sheet = stock.balance_sheet

current_price = info.get("currentPrice")
shares_outstanding = info.get("sharesOutstanding")

free_cash_flow = cashflow.loc["Free Cash Flow"].iloc[0]
cash = balance_sheet.loc["Cash Cash Equivalents And Short Term Investments"].iloc[0]
debt = balance_sheet.loc["Total Debt"].iloc[0]

discount_rate = 0.10
terminal_growth_rate = 0.03
projection_years = 5

growth_rates = [0.06, 0.08, 0.10, 0.12, 0.15]

print("Company:", ticker)
print("Current Price:", round(current_price, 2))
print("Free Cash Flow:", round(free_cash_flow / 1_000_000_000, 2), "B")
print("Cash:", round(cash / 1_000_000_000, 2), "B")
print("Debt:", round(debt / 1_000_000_000, 2), "B")
print("Shares Outstanding:", shares_outstanding)

print("\nDCF Sensitivity Analysis")
print("Discount Rate:", discount_rate * 100, "%")
print("Terminal Growth Rate:", terminal_growth_rate * 100, "%")
print("-" * 80)

print(f"{'Growth Rate':<15}{'Intrinsic Value':<20}{'Difference %':<15}{'Status'}")
print("-" * 80)

for growth_rate in growth_rates:
    discounted_fcfs = []

    for year in range(1, projection_years + 1):
        projected_fcf = free_cash_flow * ((1 + growth_rate) ** year)
        discounted_fcf = projected_fcf / ((1 + discount_rate) ** year)
        discounted_fcfs.append(discounted_fcf)

    terminal_fcf = free_cash_flow * ((1 + growth_rate) ** projection_years) * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)

    enterprise_value = sum(discounted_fcfs) + discounted_terminal_value
    equity_value = enterprise_value + cash - debt

    intrinsic_value_per_share = equity_value / shares_outstanding
    difference = ((intrinsic_value_per_share - current_price) / current_price) * 100

    if difference > 15:
        status = "Potentially Undervalued"
    elif difference < -15:
        status = "Potentially Overvalued"
    else:
        status = "Fairly Valued"

    print(
        f"{growth_rate * 100:<15.2f}"
        f"${intrinsic_value_per_share:<19.2f}"
        f"{difference:<15.2f}"
        f"{status}"
    )