# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:15:50 2019

@author: Labinfor_2
"""

from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import nltk
from nltk.downloader import *


train_set = [
    ('Se você traçar metas absurdamente altas e falhar, seu fracasso será muito melhor '
     'que o sucesso de todos', 'positivo'),
    ('A vida é melhor para aqueles que fazem o possível para ter o melhor', 'positivo'),
    ('Os empreendedores falham, em média, três, oito vezes antes do sucesso final. O que '
     'separa os bem-sucedidos dos outros é a persistência', 'positivo'),
    ('Se você não está disposto a arriscar, esteja disposto a uma vida comum', 'positivo'),
    ('Escolha uma ideia. Faça dessa ideia a sua vida. Pense nela, sonhe com ela, viva pensando nela. '
     'Deixe cérebro, músculos, nervos, todas as partes do seu corpo serem preenchidas com essa ideia. '
     'Esse é o caminho para o sucesso', 'positivo'),
    ('Para de perseguir o dinheiro e comece a perseguir o sucesso', 'positivo'),
    ('Todos os seus sonhos podem se tornar realidade se você tem coragem para persegui-los', 'positivo'),
    ('Ter sucesso é falhar repetidamente, mas sem perder o entusiasmo', 'positivo'),
    ('Sempre que você vir uma pessoa de sucesso, você sempre verá as glórias, nunca os sacrifícios que'
     ' a levaram até ali', 'positivo'),
    ('Sucesso? Eu não sei o que isso significa. Eu sou feliz. A definição de sucesso varia de pessoa'
     ' para pessoa. Para mim, sucesso é paz anterior', 'positivo'),
    ('Oportunidades não surgem. É você que as cria', 'positivo'),
    ('Não tente ser uma pessoa de sucesso. Em vez disso, seja uma pessoa de valor','positivo'),
    ('Não é o mais forte que sobrevive, nem o mais inteligente. Quem sobrevive é o mais disposto'
     ' à mudança','positivo'),
    ('A melhor vingança é um sucesso estrondoso','positivo'),
    ('Eu não falhei. Só descobri dez mil caminhos que não eram o certo','positivo'),
    ('Um homem de sucesso é aquele que cria uma parede com os tijolos que jogaram nele','positivo'),
    ('O grande segredo de uma boa vida é encontrar qual é o seu destino. E realizá-lo','positivo'),
    ('O que nos parece uma provação amarga pode ser uma bênção disfarçada','positivo'),
    ('A distância entre a insanidade e a genialidade é medida pelo sucesso  ','positivo'),
    ('Não tenha medo de desistir do bom para perseguir o ótimo','negativo'),
    ('A felicidade é uma borboleta que, sempre que perseguida, parecerá inatingível; no'
     ' entanto, se você for paciente, ela pode pousar no seu ombro','positivo'),
    ('Se você não pode explicar algo de forma simples, então você não entendeu muito bem '
     'o que tem a dizer','positivo'),
    ('Há dois tipos de pessoa que vão te dizer que você não pode fazer a diferença neste mundo: as'
     ' que têm medo de tentar e as que têm medo de que você se dê bem   ','positivo'),
    ('Comece de onde você está. Use o que você tiver. Faça o que você puder','positivo'),
    ('As pessoas me perguntam qual é o papel que mais gostei de interpretar. Eu sempre respondo: o'
     ' próximo','positivo'),
    ('Descobri que, quanto mais eu trabalho, mais sorte eu pareço ter','positivo'),
    ('O ponto de partida de qualquer conquista é o desejo','positivo'),
    ('O sucesso é a soma de pequenos esforços repetidos dia após dia','positivo'),
    ('Todo progresso acontece fora da zona de conforto','positivo'),
    ('Coragem é a resistência e o domínio do medo, não a ausência dele','positivo'),
    ('Só evite fazer algo hoje se você quiser morrer e deixar assuntos inacabados','positivo'),
    ('O único lugar em que o sucesso vem antes do trabalho é no dicionário','positivo'),
    ('Sonhar grande e sonhar pequeno dá o mesmo trabalho','positivo'),
    ('Embora ninguém possa voltar atrás e começar tudo de novo, qualquer um pode ter um ótimo'
     ' final','positivo'),
    ('Descobri que, se você tem vontade de viver e curiosidade, dormir não é a coisa mais importante',
     'positivo'),
    ('Daqui a vinte anos, você não terá arrependimento das coisas que fez, mas das que deixou de fazer.'
     ' Por isso, veleje longe do seu porto seguro. Pegue os ventos. Explore. Sonhe.','positivo'),
    ('Sempre que você se encontrar do lado da maioria, é hora de parar e refletir','positivo'),
    ('O primeiro passo rumo ao sucesso é dado quando você se recusa a ser um refém do ambiente em'
     ' que se encontra','positivo'),
    ('Continue andando. Haverá a chance de você ser barrado por um obstáculo, talvez por algo que'
     ' você nem espere. Mas siga, até porque eu nunca ouvi falar de ninguém que foi barrado enquanto'
     ' estava parado','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo'),
    ('...','positivo')
    
    ]

test_set = [
    ('40 positivos', 'positivo'),
    ('40 negativos', 'negativo'),
    ('40 neutro', 'negativo'),
    
]

cl = NaiveBayesClassifier(train_set)
accuracy = cl.accuracy(test_set)

frase1 = 'Eu não odeio todo mundo'
frase2 = 'Eu não odeio todo mundo'
frase3 = 'Eu não odeio todo mundo'
frase4 = 'Eu não odeio todo mundo'
frase5 = 'Eu não odeio todo mundo'
frase6 = 'Eu não odeio todo mundo'


blob = TextBlob(frase1,frase2, frase3, frase4, frase5, frase6, classifier=cl)

print('Esta frase é de caráter:{}'.format(blob.classify()))
print('Precisão da previsão:{}'.format(accuracy))
