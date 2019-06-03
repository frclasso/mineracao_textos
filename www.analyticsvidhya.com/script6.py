from textblob import TextBlob

blob = TextBlob("Analytics Vidhya is a great platform to leran Data Science.\nIt"
                " helps community  through blogs, hackathons, discussions, etc")

print(blob)
print()
print(blob.sentiment)