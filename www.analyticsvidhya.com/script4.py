from textblob import TextBlob

blob = TextBlob("Analytics Vidhya is a great platform to leran Data Science.\nIt helps community  through blogs, hackathons, discussions, etc")

print(blob.sentences[1].words[1])
print(blob.sentences[1].words[1].singularize())
