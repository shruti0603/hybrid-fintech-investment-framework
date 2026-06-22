import pandas as pd
from scipy.stats import pearsonr

data = {
    "Stock": ["NVDA","GOOGL","MSFT","AAPL","AMZN","TSLA"],
    "HybridScore": [78.50,77.88,65.75,52.38,52.00,42.50],
    "AvgReturn": [4.66,4.49,1.68,1.51,0.01,2.20]
}

df = pd.DataFrame(data)

corr, p_value = pearsonr(
    df["HybridScore"],
    df["AvgReturn"]
)

print("\nCorrelation:", round(corr,4))
print("P Value:", round(p_value,4))