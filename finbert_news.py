import requests
from datetime import datetime, timedelta, timezone
from transformers import pipeline

API_KEY = "d9aabc1d2d384a0680558fdbef1c27f6"

# Change only this each time:
company_key = "TSLA"
# Options: MSFT, AAPL, NVDA, AMZN, GOOGL, TSLA

COMPANY_CONFIG = {
    "MSFT": {
        "company": "Microsoft",
        "keywords": ["Microsoft", "MSFT", "Azure", "Xbox", "LinkedIn", "Microsoft Corporation"],
        "query": '"Microsoft Corporation" OR "Microsoft earnings" OR "Microsoft stock" OR "Microsoft Azure" OR "MSFT"',
        "blocked": ["Pypi", ".py", "package", "plugin"]
    },
    "AAPL": {
        "company": "Apple",
        "keywords": ["Apple", "AAPL", "iPhone", "Mac", "Apple Inc"],
        "query": '"Apple Inc" OR "Apple earnings" OR "Apple stock" OR "AAPL" OR "iPhone"',
        "blocked": [
            "how to", "tips", "wallpaper", "charger", "case", "coupon",
            "deal", "discount", "free shipping", "Slickdeals", "Apple Support"
        ]
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
        "blocked": [
            "Prime Day", "deal", "discount", "coupon", "free shipping",
            "Slickdeals", "AWS Local Zone", "EC2", "ECS", "SageMaker",
            "CloudWatch", "Bedrock", "JumpStart", "Pypi", ".py",
            "awscli", "boto3", "WEB-DL", "1080p", "Rlsbb"
        ]
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
    "earnings", "revenue", "profit", "profits", "margin", "growth",
    "forecast", "guidance", "stock", "stocks", "shares", "investors",
    "shareholders", "valuation", "market cap", "cash flow",
    "free cash flow", "quarter", "quarterly", "results", "analyst",
    "rating", "upgrade", "downgrade", "lawsuit", "probe",
    "investigation", "regulation", "SEC", "FTC", "DOJ", "antitrust",
    "acquisition", "deal", "partnership", "investment", "dividend",
    "buyback", "debt", "price target", "market", "losses", "drop",
    "fell", "rally", "buy", "sell"
]

config = COMPANY_CONFIG[company_key]

company = config["company"]
keywords = config["keywords"]
query = config["query"]
blocked_words = config["blocked"]

seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

url = (
    f"https://newsapi.org/v2/everything?"
    f"q={query}&"
    f"from={seven_days_ago}&"
    f"language=en&"
    f"sortBy=publishedAt&"
    f"pageSize=50&"
    f"apiKey={API_KEY}"
)

response = requests.get(url)
data = response.json()

if data.get("status") != "ok":
    print("API Error:", data)
    exit()

articles = data.get("articles", [])

finbert = pipeline("text-classification", model="ProsusAI/finbert")

positive = 0
negative = 0
neutral = 0
seen_titles = set()
used_articles = 0

print(f"\nFinBERT Financial-Relevance Sentiment Analysis for {company}\n")

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

    if any(word.lower() in full_text.lower() for word in blocked_words):
        continue

    if not any(keyword.lower() in full_text.lower() for keyword in keywords):
        continue

    if not any(word.lower() in full_text.lower() for word in financial_keywords):
        continue

    result = finbert(full_text[:512])[0]
    label = result["label"].lower()
    confidence = result["score"]

    if label == "positive":
        positive += 1
    elif label == "negative":
        negative += 1
    else:
        neutral += 1

    used_articles += 1

    print(f"{used_articles}. {title}")
    print("Source:", source)
    print("Published:", published_at)
    print("FinBERT Sentiment:", label.upper())
    print("Confidence:", round(confidence, 4))
    print("-" * 80)

total = positive + negative + neutral

print("\n==========================")
print(f"Company: {company}")
print("==========================")
print("Model: FinBERT + Financial Relevance Filter")
print("Search Window: Last 7 days")
print(f"Articles Retrieved: {len(articles)}")
print(f"Articles Used After Filtering: {total}")

if total > 0:
    positive_pct = (positive / total) * 100
    negative_pct = (negative / total) * 100
    neutral_pct = (neutral / total) * 100

    sentiment_score = positive_pct - negative_pct

    print(f"\nPositive Articles: {positive}")
    print(f"Negative Articles: {negative}")
    print(f"Neutral Articles: {neutral}")

    print("\nSentiment Distribution")
    print(f"Positive: {positive_pct:.2f}%")
    print(f"Negative: {negative_pct:.2f}%")
    print(f"Neutral : {neutral_pct:.2f}%")

    print("\nFiltered FinBERT Sentiment Score")
    print(f"Sentiment Score: {sentiment_score:.2f}")

    if sentiment_score > 20:
        print("Overall Sentiment: Positive")
    elif sentiment_score < -20:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Mixed / Neutral")
else:
    print("No financially relevant articles found.")