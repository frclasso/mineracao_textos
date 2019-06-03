from textblob import TextBlob

blob = TextBlob("Analytics Vidhya is a great platform to leran Data Science.\n It helps"
                "community  through blogs, hackathons, discussions, etc")

#print(blob.sentences)

for words in blob.sentences[0].words:
    print(words) # printing words of first sentence