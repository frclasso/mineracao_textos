#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:30:07 2019

@author: moises
"""

from string import punctuation
from collections import defaultdict
from heapq import nlargest
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

total_sentencas_mais_importantes = 4
nome_arq_entrada = 'corpus 1.txt'
nome_arq_saida = 'corpus 1___(resumido).txt'

print('Iniciando sumarização do corpus "{}"...'.format(nome_arq_entrada))

texto = ''
with open(nome_arq_entrada, "r") as arquivo_entrada: 
    
    texto = arquivo_entrada.read()
    sentencas = sent_tokenize(texto)
    palavras = word_tokenize(texto.lower())

    stopwords = set(stopwords.words('portuguese') + list(punctuation))
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stopwords]
    
    frequencia = FreqDist(palavras_sem_stopwords)
    sentencas_importantes = defaultdict(int)
    
    for i, sentenca in enumerate(sentencas):
        for palavra in word_tokenize(sentenca.lower()):
            if palavra in frequencia:
                sentencas_importantes[i] += frequencia[palavra]
    
    idx_sentencas_importantes = nlargest(total_sentencas_mais_importantes, sentencas_importantes, sentencas_importantes.get)
    
    i_sentenca = 0
    
    print()
    print('*' * 50)
    for i in idx_sentencas_importantes:
        i_sentenca += 1
        print('{}a. Sentença Mais Importante: {}'.format(i_sentenca, i))
    print('*' * 50)

    with open(nome_arq_saida, "w") as arquivo_saida:
        for i in sorted(idx_sentencas_importantes):
            print('\n\n################# Sentença Importante ({}) #################\n\n'.format(i))
            print(sentencas[i])
            arquivo_saida.write(sentencas[i])
    
print('\n\nSumarização finalizada.')















    