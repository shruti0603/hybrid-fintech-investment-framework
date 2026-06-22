import requests
from datetime import datetime, timedelta, timezone
from transformers import pipeline

API_KEY = "d9aabc1d2d384a0680558fdbef1c27f6"

company_key = "TSLA"  # MSFT, AAPL, NVDA, AMZN, GOOGL, TSLA

COMPANY_CONFIG = {
    "MSFT": {
        "company": "Microsoft",
        "keywords": ["Microsoft", "MSFT", "Azure", "Microsoft Corporation"],
        "query": '"Microsoft Corporation" OR "Microsoft earnings" OR "Microsoft stock" OR "Microsoft Azure" OR "MSFT"',
        "blocked": ["Pypi", ".py", "package", "plugin"]
    },
    "AAPL": {
        "company": "Apple",
        "keywords": ["Apple", "AAPL", "iPhone", "Mac", "Apple Inc"],
        "query": '"Apple Inc" OR "Apple earnings" OR "Apple stock" OR "AAPL" OR "iPhone"',
        "blocked": ["how to", "tips", "charger", "coupon", "deal", "discount", "free shipping", "Slickdeals"]
    },
    "NVDA": {
        "company": "Nvidia",
        "keywords": ["Nvidia", "NVDA", "GPU", "Blackwell", "Nvidia Corporation"],
        "query": '"Nvidia Corporation" OR "Nvidia earnings" OR "Nvidia stock" OR "NVDA" OR "Blackwell"',
        "blocked": ["Pypi", ".py", "package", "plugin", "gaming laptop"]
    },
    "AMZN": {
        "company": "Amazon",
        "keywords": ["Amazon earnings", "Amazon stock", "AMZN", "Amazon revenue", "Amazon profit", "Amazon investors"],
        "query": '"Amazon earnings" OR "Amazon stock" OR "AMZN" OR "Amazon revenue" OR "Amazon profit" OR "Amazon investors"',
        "blocked": ["Prime Day", "deal", "discount", "coupon", "free shipping", "Slickdeals", "EC2", "ECS", "SageMaker", "CloudWatch", "Pypi", ".py", "WEB-DL", "1080p", "Rlsbb"]
    },
    "GOOGL": {
        "company": "Google",
        "keywords": ["Google", "Alphabet", "GOOGL", "GOOG", "Gemini", "Google Cloud", "Alphabet Inc"],
        "query": '"Alphabet Inc" OR "Google earnings" OR "Google stock" OR "GOOGL" OR "Google Cloud" OR "Gemini AI"',
        "blocked": ["Pypi", ".py", "plugin", "Chrome extension", "coupon", "deal", "discount"]
    },
    "TSLA": {
        "company": "Tesla",
        "keywords": ["Tesla", "TSLA", "Elon Musk", "Cybertruck", "Model Y", "Model 3", "Tesla Inc"],
        "query": '"Tesla Inc" OR "Tesla earnings" OR "Tesla stock" OR "TSLA" OR "Elon Musk" OR "Cybertruck"',
        "blocked": ["toy", "coupon", "deal", "discount", "free shipping", "Pypi", ".py", "plugin"]
    }
}

financial_keywords = [
    "earnings", "revenue", "profit", "growth", "forecast", "guidance",
    "stock", "stocks", "shares", "investors", "shareholders", "valuation",
    "market cap", "cash flow", "quarter", "results", "analyst", "rating",
    "upgrade", "downgrade", "lawsuit", "probe", "investigation", "regulation",
    "SEC", "FTC", "DOJ", "antitrust", "acquisition", "partnership",
    "investment", "dividend", "buyback", "debt", "price target", "market",
    "losses", "drop", "fell", "rally", "buy", "sell"
]

config = COMPANY_CONFIG[company_key]
company = config["company"]

seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

url = (
    f"https://newsapi.org/v2/everything?"
    f"q={config['query']}&"
    f"from={seven_days_ago}&"
    f"language=en&"
    f"sortBy=publishedAt&"
    f"pageSize=50&"
    f"apiKey={API_KEY}"
)

data = requests.get(url).json()

if data.get("status") != "ok":
    print("API Error:", data)
    exit()

articles = data.get("articles", [])
finbert = pipeline("text-classification", model="ProsusAI/finbert")

weighted_sum = 0
used_articles = 0
seen_titles = set()

print(f"\nConfidence-Weighted FinBERT Sentiment for {company}\n")

for article in articles:
    title = str(article.get("title") or "")
    description = str(article.get("description") or "")
    source = article.get("source", {}).get("name", "Unknown")
    published_at = article.get("publishedAt", "Unknown")

    full_text = f"{title}. {description}"

    if not title:
        continue

    if title.lower() in seen_titles:
        continue

    seen_titles.add(title.lower())

    if any(word.lower() in full_text.lower() for word in config["blocked"]):
        continue

    if not any(keyword.lower() in full_text.lower() for keyword in config["keywords"]):
        continue

    if not any(word.lower() in full_text.lower() for word in financial_keywords):
        continue

    result = finbert(full_text[:512])[0]
    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        article_score = confidence
    elif label == "negative":
        article_score = -confidence
    else:
        article_score = 0

    weighted_sum += article_score
    used_articles += 1

    print(f"{used_articles}. {title}")
    print("Source:", source)
    print("Published:", published_at)
    print("Label:", label.upper())
    print("Confidence:", round(confidence, 4))
    print("Article Weighted Score:", round(article_score * 100, 2))
    print("-" * 80)

print("\n==========================")
print(f"Company: {company}")
print("==========================")
print("Articles Used:", used_articles)

if used_articles > 0:
    weighted_sentiment_score = (weighted_sum / used_articles) * 100
    print("Weighted FinBERT Sentiment Score:", round(weighted_sentiment_score, 2))
else:
    print("No financially relevant articles found.")