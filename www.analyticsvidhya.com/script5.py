from textblob import TextBlob

blob = TextBlob("Analytics Vidhya is a great platform to leran Data Science.\nIt helps community  through blogs, hackathons, discussions, etc")

for word,pos in blob.tags:
    if pos == 'NN':
        print(word.pluralize())