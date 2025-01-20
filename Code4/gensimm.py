import gensim.downloader

model = gensim.downloader.load('glove-wiki-gigaword-100')

# Vektoren extrahieren
vector_s = model["germany"]
vector_t = model["italy"]
vector_n = model["mossulini"]

# Arithmetische Operation
result_vector = vector_s - vector_t + vector_n


# Ähnlichstes Wort finden
similar_words = model.most_similar(positive=[result_vector], topn=5)

# Ergebnisse ausgeben
print("Ähnlichste Wörter:", similar_words)




