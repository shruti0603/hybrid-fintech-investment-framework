import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, timezone

API_KEY = "d9aabc1d2d384a0680558fdbef1c27f6"

company = "Amazon"
keywords = [
    "Amazon earnings",
    "Amazon stock",
    "AMZN",
    "Amazon revenue",
    "Amazon profit",
    "Amazon valuation",
    "Amazon investors"
]

query = (
    '"Amazon earnings" OR '
    '"Amazon stock" OR '
    '"AMZN" OR '
    '"Amazon revenue" OR '
    '"Amazon profit" OR '
    '"Amazon investors"'
)

blocked_words = [
    # Shopping / Deals
    "Prime Day",
    "deal",
    "discount",
    "coupon",
    "free shipping",
    "Slickdeals",

    # AWS Technical Releases
    "AWS Local Zone",
    "EC2",
    "ECS",
    "SageMaker",
    "CloudWatch",
    "Bedrock",
    "JumpStart",
    "MSK",
    "Datadog",
    "DevOps Agent",
    "containerd",
    "Local Zone",

    # Package Releases
    "Pypi",
    ".py",
    "awscli",
    "boto3",
    "dagster",
    "amplify",
    "workbench",
    "cleancloud",
    "iam-policy-validator",

    # Misc Noise
    "WEB-DL",
    "1080p",
    "Rlsbb",
    "movie",
    "TV series",
    "streaming"
]
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

analyzer = SentimentIntensityAnalyzer()

positive = 0
negative = 0
neutral = 0
used_articles = 0
seen_titles = set()

print(f"\nFiltered {company} Articles from last 7 days\n")

for article in articles:
    title = str(article.get("title") or "")
    description = str(article.get("description") or "")
    source = article.get("source", {}).get("name", "Unknown")
    published_at = article.get("publishedAt", "Unknown")

    full_text = f"{title} {description}"

    if not title:
        continue

    if title.lower() in seen_titles:
        continue

    seen_titles.add(title.lower())

    if any(word.lower() in full_text.lower() for word in blocked_words):
        continue

    if not any(keyword.lower() in full_text.lower() for keyword in keywords):
        continue

    score = analyzer.polarity_scores(full_text)
    compound = score["compound"]

    if compound >= 0.05:
        sentiment = "POSITIVE"
        positive += 1
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
        negative += 1
    else:
        sentiment = "NEUTRAL"
        neutral += 1

    used_articles += 1

    print(f"{used_articles}. {title}")
    print("Source:", source)
    print("Published:", published_at)
    print("Sentiment:", sentiment)
    print("Compound Score:", compound)
    print("-" * 80)

total = positive + negative + neutral

print("\n==========================")
print(f"Company: {company}")
print("==========================")
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

    print("\nOverall Sentiment Score")
    print(f"Sentiment Score: {sentiment_score:.2f}")

    if sentiment_score > 20:
        print("Overall Sentiment: Positive")
    elif sentiment_score < -20:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Mixed / Neutral")
else:
    print("No relevant articles found.")