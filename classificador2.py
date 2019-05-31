#!/usr/bin/env python 3

# -*-coding:utf-8-*-

from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob


# criando um dataset de treinamento
train_set = [

('Se você traçar metas absurdamente altas e falhar, seu fracasso será muito'
 ' melhor que o sucesso de todos', 'positivo'),
    ('O sucesso normalmente vem para quem está ocupado demais'
     ' para procurar por ele','positivo'),
    ('A vida é melhor para aqueles que fazem o possível para ter o melhor','positivo'),
    ('Os empreendedores falham, em média, três, oito vezes antes do sucesso final.'
     ' O que separa os bem-sucedidos dos outros é a persistência','positivo'),
    ('Se você não está disposto a arriscar, esteja disposto a uma vida comum',
     'positivo'),
    ('Escolha uma ideia. Faça dessa ideia a sua vida. Pense nela, sonhe com ela,'
     ' viva pensando nela. Deixe cérebro, músculos, nervos, todas as partes do'
     ' seu corpo serem preenchidas com essa ideia. Esse é o caminho para o '
     'sucesso','positivo'),
    ('Para de perseguir o dinheiro e comece a perseguir o sucesso','positivo'),
    ('Todos os seus sonhos podem se tornar realidade se você tem coragem para'
     ' persegui-los','positivo'),
    ('er sucesso é falhar repetidamente, mas sem perder o entusiasmo','positivo'),
    ('Sempre que você vir uma pessoa de sucesso, você sempre verá as glórias, '
     'nunca os sacrifícios que a levaram até ali','positivo'),
    ('Sucesso? Eu não sei o que isso significa. Eu sou feliz. A definição de '
     'sucesso varia de pessoa para pessoa. Para mim, sucesso é paz anterior',
     'positivo'),
    ('Oportunidades não surgem. É você que as cria','positivo'),
    ('Não tente ser uma pessoa de sucesso. Em vez disso, seja uma pessoa de valor',
     'positivo'),
    ('Não é o mais forte que sobrevive, nem o mais inteligente. '
     'Quem sobrevive é o mais disposto à mudança','positivo'),
    ('A melhor vingança é um sucesso estrondoso','positivo'),
    ('Eu não falhei. Só descobri dez mil caminhos que não eram o certo','positivo'),
    ('Um homem de sucesso é aquele que cria uma parede com os tijolos que '
     'jogaram nele','positivo'),
    ('O grande segredo de uma boa vida é encontrar qual é o seu destino. '
     'E realizá-lo','positivo'),
    ('O que nos parece uma provação amarga pode ser uma bênção disfarçada','positivo'),
    ('A distância entre a insanidade e a genialidade é medida pelo sucesso','positivo'),
    ('Não tenha medo de desistir do bom para perseguir o ótimo','positivo'),
    ('A felicidade é uma borboleta que, sempre que perseguida, parecerá inatingível;'
     ' no entanto, se você for paciente, ela pode pousar no seu ombro','positivo'),
    ('Se você não pode explicar algo de forma simples, então você não entendeu muito'
     ' bem o que tem a dizer','positivo'),
    ('Há dois tipos de pessoa que vão te dizer que você não pode fazer a diferença'
     ' neste mundo: as que têm medo de tentar e as que têm medo de que você se dê'
     ' bem','positivo'),
    ('Comece de onde você está. Use o que você tiver. Faça o que você puder',
     'positivo'),
    ('As pessoas me perguntam qual é o papel que mais gostei de interpretar.'
     ' Eu sempre respondo: o próximo','positivo'),
    ('Descobri que, quanto mais eu trabalho, mais sorte eu pareço ter','positivo'),
    ('O ponto de partida de qualquer conquista é o desejo','positivo'),
    ('O sucesso é a soma de pequenos esforços repetidos dia após dia','positivo'),
    ('Todo progresso acontece fora da zona de conforto','positivo'),
    ('Coragem é a resistência e o domínio do medo, não a ausência dele','positivo'),
    ('Só evite fazer algo hoje se você quiser morrer e deixar assuntos inacabados','positivo'),
    ('O único lugar em que o sucesso vem antes do trabalho é no dicionário','positivo'),
    ('Sonhar grande e sonhar pequeno dá o mesmo trabalho','positivo'),
    ('Embora ninguém possa voltar atrás e começar tudo de novo,'
     ' qualquer um pode ter um ótimo final','positivo'),
    ('Descobri que, se você tem vontade de viver e curiosidade, '
     'dormir não é a coisa mais importante','positivo'),
    ('Daqui a vinte anos, você não terá arrependimento das coisas que fez, '
     'mas das que deixou de fazer. Por isso, veleje longe do seu porto seguro. '
     'Pegue os ventos. Explore. Sonhe. Descubra','positivo'),
    ('O primeiro passo rumo ao sucesso é dado quando você se recusa a ser um'
     ' refém do ambiente em que se encontra','positivo'),
    ('Sempre que você se encontrar do lado da maioria, é hora de parar e refletir',
     'positivo'),
    ('Continue andando. Haverá a chance de você ser barrado por um obstáculo, '
     'talvez por algo que você nem espere. Mas siga, até porque eu nunca ouvi '
     'falar de ninguém que foi barrado enquanto estava parado','positivo'),
    ('Se você realmente quer algo, não espere. Ensine você mesmo a ser impaciente',
     'positivo'),
    ('e você quer uma mudança permanente, pare de focar no tamanho de seus problemas'
     ' e comece a focar no seu tamanho','positivo'),
    ('Pessoas de sucesso fazem o que pessoas mal sucedidas não querem fazer. '
     'Não queira que a vida seja mais fácil. Deseje que você seja ainda melhor',
     'positivo'),
    ('A primeira razão para o fracasso de alguém é escutar amigos, família e'
     ' vizinhos','positivo'),
    ('O sucesso não consiste em não errar, mas não cometer os mesmos equívocos'
     ' mais de uma vez','positivo'),
    ('A motivação é o que faz o empreendedor começar e o hábito é o que nos'
     ' faz continuar','positivo'),
    ('Nosso maior medo não deve ser o fracasso, mas ser bem-sucedido em algo'
     ' que não importa','positivo'),
    ('Se você não traçou um plano para você mesmo, é possível que você caia no '
     'plano de outra pessoa. E adivinha o que ele planejou para você? Não muito',
     'positivo'),
    ('Você deve lutar mais de uma batalha para se tornar um vencedor','positivo'),
    ('Eu devo meu sucesso a meu hábito de respeitosamente ouvir conselhos'
     ' e fazer exatamente o contrário','positivo'),
    ('Muitas das falhas da vida ocorrem quando não percebemos o quão próximos'
     ' estávamos do sucesso na hora em que desistimos','positivo'),
    ('Quanto maior o artista, maior a dúvida. Confiança grande demais é algo'
     ' destinados aos menos talentosos como um prêmio de consolação','positivo'),
    ('Tenha em mente que o seu desejo em atingir o sucesso é mais importante que'
     ' qualquer coisa','positivo'),
    ('Fique contente em agir. Deixe a fala para os outros','positivo'),
    (' Para conquistar o sucesso, você precisa aceitar todos os desafios que '
     'vierem na sua frente. Você não pode apenas aceitar os que você preferir',
     'positivo'),
    ('O guerreiro de sucesso é um homem médio, mas com um foco apurado como'
     ' um raio laser','positivo'),
    ('A lógica pode levar de um ponto A a um ponto B. A imaginação pode levar'
     ' a qualquer lugar','positivo'),
    ('Sonhe como se você fosse viver para sempre. Viva como se você fosse '
     'morrer hoje','positivo'),
    ('Fazer o que você gosta é liberdade. Gostar do que você faz é'
     ' felicidade','positivo'),
    ('Seja feliz com o que você tem, mas fique animado com a chance de ter'
     ' mais','positivo'),
    ('Seu tempo é curto. Por isso, não o desperdice vivendo a vida de outra'
     ' pessoa','positivo'),
    ('Somos nós que forjamos as correntes que usamos em nossas vidas','positivo'),
    ('A arte de viver bem não consiste em eliminar o que nos faz sofrer, '
     'mas crescer com esses problemas','positivo'),
    ('Você nunca se arrependerá de ser gentil','positivo'),
    ('Em nossas vidas, a mudança é inevitável. A perda é inevitável. A felicidade'
     ' reside na nossa adaptabilidade em sobreviver a tudo de ruim','positivo'),
    ('Para cuidar de si mesmo, use a cabeça. Para cuidar dos outros, use seu '
     'coração','positivo'),
    ('Apenas um entre mil é um líder de outros homens – os outros 999 '
     'seguem suas mulheres','positivo'),
    ('Mantenha seus medos consigo, mas compartilhe sua coragem com os '
     'outros','positivo'),
    ('A felicidade é uma borboleta que, sempre que perseguida, parecerá'
     ' inatingível. No entanto, se você for paciente, ela pode pousar no '
     'seu ombro','positivo'),
    ('Faça ou não faça. Tentativas não existem','positivo'),
    ('Se alguém não se sente agradecido pelo que tem, ele provavelmente nunca'
     ' será agradecido pelo que conseguir','positivo'),
    ("Se você ouvir uma voz dizendo 'não faça', isso significa que você deve fazê-lo,"
     " acima de tudo. A voz vai se calar",'positivo'),
    ('Você não se preocuparia tanto sobre o que pensam de você se você soubesse'
     ' que poucos perdem tempo com isso','positivo'),
    ('Autoconfiança é muito importante para alcançar o sucesso. E para se tornar'
     ' confiante, é importante estar preparado','positivo'),
    ('Se todos se propusessem a fazer o que são capazes, ficaríamos impressionados'
     ' com nossas criações','positivo'),
    ('Sempre se lembre de que você tem mais fibra que acredita, é mais forte que'
     ' parece e mais esperto do que você pensa que é','positivo'),
    ('É difícil liderar uma cavalaria se você não sabe montar a cavalo','positivo'),
    ('"80% do necessário para o sucesso é aparecer','positivo'),
    ('Nunca é tarde para ser o que você poderia ter sido','positivo'),
    ('A vida é uma jornada. Se você se apaixonar pela jornada, você será um ser'
     ' apaixonado até o fim dos tempos','positivo'),
    ('Realize seus próprios sonhos. Do contrário, você será contratado para '
     'realizar os de outras pessoas','positivo'),
    ('Assim que você acreditar em si mesmo, você saberá como viver','positivo'),
    ('Você jamais se sentirá sozinho se gostar de si mesmo','positivo'),
    ('Não desperdice sua energia tentando mudar opiniões. Faça seu trabalho '
     'e não ligue tanto para os outros','positivo'),
    ('Não espere até que tudo esteja perfeito. Nunca estará tudo bem. '
     'Sempre haverá desafios e obstáculos. E daí? Comece agora. A cada passo'
     ' dado, você estará mais forte, mais habilidoso, mais confiante e mais '
     'bem-sucedido','positivo'),
    ('A falta de autoconfiança não é uma pena perpétua. A autoconfiança pode'
     ' ser aprendida, praticada e dominada. Assim que você acreditar em si'
     ' mesmo, tudo em sua vida mudará para melhor','positivo'),
    ('Uma chave para o sucesso é a confiança. E uma chave para a confiança'
     ' é a preparação','positivo'),
    ('É a confiança em nossos corpos e mentes que nos permite buscar novas '
     'aventuras','positivo'),
    ('Ser você mesmo em um mundo que está sempre tentando te mudar é a maior'
     ' conquista possível','positivo'),
    ('Nenhum projeto disruptivo foi realizado sem risco. Deve-se estar disposto'
     ' a arriscar sempre','positivo'),
    ('Pessoas são como janelas de vidro. Elas brilham quando o sol está lá fora. '
     'Mas é só quando a escuridão chega é que vemos sua luz interior e o quão '
     'incríveis elas são','positivo'),
    ('A confiança não vem de sempre estar certo. Ela vem do ato de nunca temermos'
     ' nos equivocar','positivo'),
    ('Trate a si mesmo como você trata aqueles que ama','positivo'),
    ('Confie em si mesmo. Você pode mais do que pensa','positivo'),
    ('Pessoas bem-sucedidas têm dúvidas e preocupações. Elas só não deixam esses'
     ' sentimentos dominarem sua vida','positivo'),
    ('Você pode ter o que quiser. Basta deixar de lado o pensamento de que seus'
     ' sonhos são inatingíveis','positivo'),
    ('A confiança te deixa sexy','positivo'),
    ('Você, assim como qualquer pessoa em todo o universo, merece amor e '
     'afeição','positivo'),
    ('A inação ajuda a desenvolver a dúvida e o medo. A ação ajuda a ter'
     ' confiança e coragem. Se você quer ter algo na vida, não fique sentado.'
     'Levante-se e vá se ocupar','positivo'),
    ('Nada pode parar um homem com a atitude mental certa. E nada pode ajudar'
     ' quem tem a atitude errada','positivo'),
    ('Ao parar por um minuto e contar todas as suas conquistas, você perceberá'
     ' como é bem-sucedido','positivo'),
    ('Todos os que disseram que você não é bom... tampouco são bons','positivo'),
    ('Um líder de verdade tem confiança para ficar sozinho, coragem para tomar'
     ' decisões difíceis e compaixão para escutar a necessidade dos '
     'outros','positivo'),
    ('Se todo mundo está pensando igual, então tem gente que não está '
     'pensando em nada','positivo'),
    ('Temos que ajustar nosso caminho rumo às estrelas, não para as luzes de '
     'todo navio que passa no mar','positivo'),
    ('Enquanto uma pessoa hesita por se sentir inferior, outra está ocupada '
     'cometendo erros e se tornando superior','positivo'),
    ('Nada é menos produtivo do que tornar eficiente algo que nem deveria ser '
     'feito','positivo'),
    ('Produtividade nunca é um acidente. É sempre o resultado de comprometimento'
     ' com a excelência, planejamento inteligente e esforço focado','positivo'),
    ('Estar ocupado nem sempre significa trabalho de verdade. O objetivo de todo'
     ' trabalho é produção ou conquista, e para qualquer um desses objetivos deve'
     ' haver previsão, sistematização, planejamento, inteligência e propósito '
     'honesto, assim como transpiração. Parecer estar fazendo não é fazer','positivo'),
    ('Saber o que torna seus funcionários felizes irá não só aumentar produtividade'
     ' e moral, mas também fará com eles pensem menos em desistir','positivo'),
    ('“Um trabalhador sem genialidade é melhor do que um gênio que não quer'
     ' trabalhar','positivo'),
    ('Quando tudo parecer estar contra você, lembre-se que o avião decola contra o'
     ' vento, não com a ajuda dele','positivo'),
    ('Falhe sete vezes. Levante-se oito','positivo'),
    ('Você não falhará se não subir a montanha. Mas não tem graça nenhuma viver'
     ' sempre com o pé no chão','positivo'),
    ('Você não aprende a andar seguindo regras. Você aprende fazendo e '
     'caindo','positivo'),
    ('Eu tenho medo em todos os momentos da minha vida e isso nunca me impediu'
     ' de fazer nada que eu quisesse fazer','positivo'),
    ('Você não escolhe as suas paixões. Suas paixões escolhem você','positivo'),
    ('Escolha um trabalho que você ama e você nunca terá que trabalhar '
     'um dia sequer na vida','positivo'),
    ('Obstáculos não podem parar você. Se você achar uma parede, não desista. '
     'Ache uma maneira de escalá-la, atravessá-la ou derrubá-la','positivo'),
    ('Uma pessoa que nunca cometeu erros nunca tentou nada novo','positivo'),
    ('Eu sofro da crença que todo produto da minha imaginação não é só possível,'
     ' mas que fatalmente se tornará real','positivo'),
    ('Tenha a coragem de seguir seu coração e sua intuição. O resto é '
     'secundário','positivo'),
    ('Não ter medo é como fazer musculação. Quanto mais eu me exercito, '
     'menor a chance de meus temores me dominarem','positivo'),
    ('Sentir prazer no que faz torna o trabalho perfeito','positivo'),
    ('Eu quase ia dizer para você fazer o que ama, mas não é bem assim. '
     'As pessoas mais felizes e bem-sucedidas não amam o que fazem. '
     'Elas são obcecadas em resolver algo que importa a elas','positivo'),
    ('Você pode conseguir tudo o que quer ou simplesmente ficar velho','positivo'),
    ('O homem que remove montanhas sempre começa retirando pequenas pedras '
     'do caminho','positivo'),
    ('Você pode procurar uma resposta no Google. Você pode procurar um relacionamento, '
     'ou uma carreira, no Google. Mas você nunca vai encontrar o que está em '
     'seu coração em uma busca online','positivo'),
    ('Um homem é um sucesso se ele acorda pela manhã, vai para a cama à noite e,'
     ' nesse ínterim, fez o que quis fazer','positivo'),
    ('Eu prefiro morrer de paixão do que de tédio','positivo'),
    ('Muitos dos nossos sonhos parecem impossíveis a princípio. Depois, '
     'improváveis. E aí, em um certo momento, eles se tornarão inevitáveis','positivo'),
    ('Ame o que faz e faça o que ama','positivo'),
    ('As marcas mais poderosas e duradouras são as que ficam no coração das '
     'pessoas','positivo'),
    ('Não deixe o que você não pode fazer interferir no que você pode '
     'fazer','positivo'),
    ('A única coisa pior do que tentar algo e falhar é... não tentar algo','positivo'),
    ('Tudo começou do nada','positivo'),
    ('Quando eu desisto de ser quem eu sou, posso me tornar o que poderia '
     'ser','positivo'),
    ('Os dois dias mais importantes da sua vida são o dia em que você nasceu'
     ' e o dia em que você descobre por que você nasceu','positivo'),
    ('Quanto mais eu quero que algo seja feito, menor é a chance de eu chamar'
     ' aquilo de trabalho','positivo'),
    ('Todo grande sonho começa na mente de um sonhador. Lembre-se de que você tem, '
     'dentro de você, a garra e a paciência para atingir as estrelas e'
     ' mudar o mundo','positivo'),
    ('Vencedores nunca desistem e quem desiste nunca vence','positivo'),
    ('A chave para a felicidade é encontrar sua vocação e assegurar uma oportunidade'
     ' para segui-la','positivo'),
    ('A humanidade, na busca pelo que fazer na vida, deixa de viver','positivo'),
    ('Nunca continue em um trabalho que você não gosta. Se você está satisfeito '
     'com o que faz, você estará satisfeito consigo mesmo e terá paz interior. '
     'E com isso, terá mais sucesso do que jamais poderia ter imaginado','positivo'),
    ('Um líder é um vendedor de esperança','positivo'),
    ('Minha definição de liderança é a seguinte: a capacidade e o desejo de guiar,'
     ' com caráter, homens e mulheres rumo a um objetivo','positivo'),
    ('Nunca dê uma ordem que não pode ser executada','positivo'),
    ('A liderança é a arte de fazer uma pessoa querer fazer algo que, na verdade,'
     ' vai ajudar você','positivo'),
    ('Não se lidera acertando a cabeça das pessoas. O nome disso não é liderança;'
     ' é violência','positivo'),
    ('Se não fosse duro qualquer um o faria','negativo'),
    ('O único dia fácil foi ontem','negativo'),
    ('Só os fracos desistem. Ninguém disse que ia ser fácil','negativo'),
    ('The harder you fall, the higher you bounce','negativo'),
    ('Obstáculos aparecem para testar se o que queres vale mesmo a pena','negativo'),
    ('É das dificuldades que nascem os milagres','negativo'),
    ('Não estou a dizer que vai ser fácil, estou a dizer que vai valer a '
     'pena','negativo'),
    ('Aceite os seus limites sem desacreditar na sua capacidade de superação','negativo'),
    ('Podes desistir se quiseres e ninguém se vai importar. Mas vais saber para '
     'o resto da tua vida','negativo'),
    ('O céu não é o meu limite. Eu sou','negativo'),
    ('Livrai-me, Senhor, Das pessoas maldosas.','negativo'),
    ('Livrai-me, Senhor, Dos pensamentos negativos.','negativo'),
    ('Livrai-me, Senhor, Das palavras que machucam.','negativo'),
    ('Livrai-me, Senhor, Dos medos que aprisionam.','negativo'),
    ('Livrai-me, Senhor, Das escolhas erradas.','negativo'),
    ('Livrai-me, Senhor, Da tristeza que dói','negativo'),
    ('Livrai-me, Senhor, Dos dias sombrios.','negativo'),
    ('Livrai-me, Senhor, E dos obstáculos que me impedem de chegar a Ti','negativo'),
    ('Vocês que pensam que sabem tudo, se incomodem conosco que sabemos!','negativo'),
    ('A inteligência artificial não é páreo para a estupidez natural!','negativo'),
    ('Fujões nunca vencem. Vencedores nunca fogem. Mas aqueles que nunca vencem e '
     'continuam sem fugir são idiotas!','negativo'),
    ('Finalmente ela olhou pra você. E te achou feio','negativo'),
    ('Uma jornada de milhares de quilômetros, as vezes termina mal. '
     'Muito mal','negativo'),
    ('Sonhos são como arco-íris. Somente os bobos correm atrás deles!','negativo'),
    ('Você venceu seus inimigos. Playstation!','negativo'),
    ('Está tudo dando certo". Ai você acorda','negativo'),
    ('Ambição. Uma jornada de milhares de quilômetros, as vezes termina mal. '
     'Muito mal!','negativo'),
    ('Bloqueie suas fotos no Facebook!. Evite que mais pessoas sintam nojo de '
     'você!','negativo'),
    ('Ele ainda se lembra de você". Com raiva','negativo'),
    ('Deixe Namorando no Facebook". Engane as pessoas','negativo'),
    ('Se você não conseguir de primeira, remova todas as evidências que'
     ' você ao menos tentou!','negativo'),
    ('Tudo saiu Errado!!','negativo'),
    ('Ainda não existe maquiagem para disfarçar os olhos fundos, '
     'tristes e desmotivados','negativo'),
    ('Saboreie os momentos sob os holofotes. Eles não irão durar muito','negativo'),
    ('Um vendedor deprimido ou desmotivado jamais alcançará a sua meta '
     'profissional, porque se agarra a um passado de lembranças e ações frustradas','negativo'),
    ('Existem aqueles que tentam lhe desmotivar com absurdas palavras. '
     'E eles mau sabem que você os acham uns tremendos idiotas delinquentes','negativo'),
    ('A gestão de pessoas tem que lidar com colaboradores e pessoas desmotivadas'
     ' e intransigentes as mudanças, fazendo com que a ingestão cógnita por falta'
     ' de capacidades, habilidades e atitudes venham ser usadas como um agente'
     ' patogênico e venenoso para a organização e seu gestor.','negativo'),
    ('Por menor que seja a iluminação para sua inspiração, '
     'jamais se entregue para desmotivação se escondendo na escuridão','negativo'),
    ('Uma pessoa desmotivada se torna um peso demasiado grande para outra carregar. '
     'E o não que recebemos não é apenas para nos desmotivar, para confirmar'
     ' as nossas crenças limitantes que alimentamos durante nossa vida','negativo'),
    ('Desmotivar o interesse de outros por nós mesmos não os valorizarmos é '
     'como sentenciar nossa própria alma a uma eternidade de solidão.','negativo'),
    ('Mesmo sabendo que não estamos sozinhos, há um universo de pessoas vivendo '
     'os mesmos dramas, ainda sim somos capazes de pensar que ninguém na face'
     ' da terra está, neste exato momento, mais triste do que nós.','negativo'),
    ('Não desanime com uma derrota. Amanhã tem mais','negativo'),
    ('Talvez as coisas mudem. Para a pior.','negativo'),
    ('Nada é tão horrivel, que não possa piorar muito.','negativo'),
    ('O não voce ja tem, agora falta buscar a humilhação','negativo'),
    ('Não deixe para desistir amanhã, do que voce pode desistir hoje.','negativo'),
    ('A morte é inevitavel','negativo'),
    ('O caminho  é longo, mas a derrota é certa.','negativo'),
    ('Errar não é uma opção','negativo'),
    ('Tenho dias ruins. Fico mal humorada, irônica e bruta com qualquer um. '
     'Respondo com grosseria e não suporto ouvir besteiras. Falo demais e me '
     'estresso sem motivos. Fico com raiva, choro, grito, fecho a cara. Porque '
     'eu sou humana, cara. E isso é normal na vida de qualquer um. Será que dá '
     'pra entender?','negativo'),
    ('Basta um dia ruim para reduzir o mais são dos homens a um lunático.','negativo'),
    ('Lembra-te do teu Criador nos dias da tua mocidade, antes que venham os maus '
     'dias, e cheguem os anos dos quais dirás: não tenho neles prazer.','negativo'),
    ('Calma, respira. Dias ruins também chegam ao fim.','negativo'),
    ('Lembrarei dos dias ruins, pois eles trazem a inspiração, o que antes era só '
     'dor, hoje move minha emoção','negativo'),
    ('Os dias ruins todo mundo tem… Já jurei pra mim não desanimar e não ter mais '
     'pressa, pois sei que o mundo vai girar, e eu espero a minha vez...','negativo'),
    ('Quando o Desânimo Vier, vai Encontrar Alguém Corajoso e com fé .','negativo'),
    ('Quando os dias Ruins chegarem , Não me faltará Esperança ','negativo'),
    ('E Quando perder tudo, Ainda terei a Mim ','negativo'),
    ('Gosto de gente que insiste em mim, que apesar dos dias ruins, de eu ser '
     'complicado de lidar, não desiste e tenta me entender e falar '
     'comigo.','negativo'),
    ('Viver é o pior video game de todos. Você só tem uma vida, os poderes são '
     'ruins e algumas fases duram anos.','negativo'),
    ('No momento ruim da vida, prefiro fingir que estou tranqüilo e vencer, '
     'do que transmitir meu medo e perder','negativo'),
    ('Tempestades são apenas fases ruins da vida…','negativo'),
    ('A vida é feita de fases... Uma fase ruim não significa ser o fim. Entenda: '
     'Dias difíceis passam e logo o que é bom chega para você.','negativo'),
    ('A vida é feita de momentos bons e ruins. Estes últimos servem para você '
     'valorizar os primeiros','negativo'),
    ('A vida quando não é bem administrada, pode se tornar um mar de '
     'decepções.','negativo'),
    ('Passar por maus momentos faz parte da vida. Só não faz parte dela você se '
     'deixar abalar por eles.','negativo'),
    ('O homem que insiste em apagar os maus momentos de sua vida nunca '
     'evoluirá','negativo'),
    ('Quando o cenário está ruim, uma corda puxa a outra, e em algum momento'
     ' você terá que escolher uma.','negativo'),
    ('A graça da vida são os momentos ruins e o que você faz deles.','negativo'),
    ('A vida é assim a gente passa por momentos ruins, as vezes pensamos em nos '
     'destruir, mas a vida não é tão ruim assim','negativo'),
    ('Devemos deixar que os momentos ruins passem, pois a vida CONTINUA!','negativo'),
    ('A grande beleza da vida, Em nossos momentos ruins, Está em vencer as '
     'batalhas.','negativo'),
    ('A vida tem momentos ruins, mas devemos pegar os momentos runis e usar '
     'como experiencia na vida.','negativo'),
    ('A vida não é so coisas ruins, todos temos nossos momentos de sofrimento, '
     'mais depois a gente aprende, a ferida cura e voltamos ao sofrimento que a vida ja é.','negativo'),
    ('Há "momentos " na nossa vida que apesar de presenciamos "coisas ruins" '
     'queremos que durem eternamente, Infelizmente poucos entendem.','negativo'),
    ('Bom mesmo é quando a vida nos surpreende com grandes momentos,quando as coisas'
     ' ruins vão ficando para trás.Pois é: Isto acontece','negativo'),
    ('A vida só é ruim para aqueles que não sabem aproveitar.','negativo'),
    ('A vida é um circulo vicioso, se estivermos em um momento ruim da '
     'vida e permanecermos inertes corremos o risco de dar voltas pra '
     'sempre.','negativo'),
    ('O ruim da vida é saber que ela tem um fim, você passa a perceber que cada '
     'momento pode não ficar na memória. Ao menos não na sua memória','negativo'),
    ('Pessoas passam por nossa vida a todo momento, e cabe a nós escolher quem'
     ' deve ficar, o ruim é quando escolhemos errado','negativo'),
    ('Sorria, mesmo em meio aos caos da vida, mesmo em momentos ruim vale mais'
     ' apena sorrir','negativo'),
    ('Não julgue um momento ruim da sua vida como um ponto final','negativo'),
    ('Pare de parar sua vida em qualquer momento ruim que aconteceu. '
     'A maioria, você que permitiu.','negativo'),
    ('O que torna as fases ruins da vida suportáveis, é compreender que assim '
     'como tudo, elas também são passageiras.','negativo'),
    ('A vida não é uma competição de problemas. Cada um sabe exatamente o quanto'
     ' pesa a sua dor','negativo'),
    ('As más companhias são como um mercado de peixe; acabamos por nos acostumar'
     ' ao mau cheiro','negativo'),
    ('Críticos são sujeitos que têm mau hálito no pensamento','negativo'),
    ('As discussões devem ser evitadas; são sempre de mau tom e muitas vezes'
     ' convincentes','negativo'),
    ('Se a escravatura não é má, nada é mau','negativo'),
    ('A melhor maneira de responder a um mau argumento é deixá-lo '
     'continuar.','negativo'),
    ('A sua irritação não solucionará problema algum. O seu mau humor não '
     'modifica a vida','negativo'),
    ('Não estrague o seu dia.','negativo'),
    ('A Matemática não mente. Mente quem faz mau uso dela','negativo'),
    ('Não existe nada tão mau, selvagem e cruel, na natureza, quanto os '
     'homens normais.','negativo'),
    ('A ausência de dinheiro nos torna pobres, mas o mau uso dele nos '
     'torna miseráveis.','negativo'),
    ('Mentes criativas são conhecidas por sobreviverem a qualquer tipo '
     'de mau treinamento.','negativo'),
    ('Não é mau este costume de escrever o que se pensa e o que se vê, '
     'e dizer isso mesmo quando não se vê nem pensa nada.','negativo'),
    ('Nenhum inimigo é pior do que um mau conselho.','negativo'),
    ('Há mais do que uma sabedoria, e todas elas são necessárias ao mundo; '
     'não é mau que elas se vão alternando.','negativo'),
    ('Não há um lado mau da vida: a vida é una','negativo'),
    ('Para mau pagador, más garantias','negativo'),
    ('Só se é verdadeiramente mau quando se tem consciência disso.','negativo'),
    ('A vida é muito perigosa, não só pelas pessoas que fazem o mau, mas'
     ' pelas que se sentam para ver o que acontece.','negativo'),
    ('E, contudo, para cada mau há um pior.','negativo'),
    ('Os que falam mau dos outros em tua presença, na tua ausência falará '
     'mal de ti.','negativo'),
    ('As frases nos conquistam, quando nos roubam, um pouco do mau-humor '
     'que possuímos.','negativo'),
    ('A ironia é uma forma elegante de ser mau','negativo'),
    ('Não sou um completo inútil, posso servir de mau exemplo.','negativo'),
    ('Riqueza alguma poderá proporcionar a paz a um homem mau','negativo'),
    ('Pior do que fazer mau uso de uma palavra é vestir qualquer peça de '
     'indiferença.','negativo'),
    ('Quem é mau caráter, sempre vai achar uma desculpa para tornar legítimas '
     'suas ações','negativo'),
    ('O avarento, por um mau cálculo, sofre de presente os males que receia'
     ' no futuro.','negativo'),
    ('Um mau estilo é um pensamento imperfeito.','negativo'),
    ('O ideal é uma maneira de mostrarmos o mau humor','negativo'),
    ('Quando se tem caráter, ele é mau.','negativo'),
    ('Muitas vezes uma cidade inteira pagou por um homem mau.','negativo'),
    ('No mau é que está o seu próprio inferno.','negativo'),
    ('Se você está atravessando um inferno, continue atravessando','negativo'),
    ('Um sonho não vira realidade a partir de mágica. '
     'Você precisa de suor, determinação e trabalho duro','negativo'),
    ('Empreender é se jogar de um precipício e construir um avião '
     'durante a queda','negativo'),
    ('Ninguém pode fazer você se sentir inferior sem o seu consentimento','neutra'),
    ('','neutra'),
    ('O valor do Eu te amo nao esta na frase e sim no sentimento...','neutra'),
    ('Tudo na vida e transitorio','neutra'),
    ('Vida isso define respirar, imaginar e criar momentos.','neutra'),
    ('Estar vivo é estar aberto a todo tipo de sentimentos e sonhos...','neutra'),
    ('A vida nos ensina todos os dias, novas lições. Que sempre possamos estar com'
     ' o coração e a mente abertos para aprendermos','neutra'),
    ('Amigos verdadeiros são aqueles que estão em todos os momentos da sua vida','neutra'),
    ('Nem todo mundo será Astronauta quando crescer','neutra'),
    ('Faça um bom uso do objeto cilíndrico que você deve ter, sob a sua mesa!','neutra'),
    ('Se você não conseguir de primeira, remova todas as evidências de '
     'que você ao menos tentou!','neutra'),
    ('A Inabilidade de um Time, tem um impacto muito maior que a soma das '
     'inabilidades individuais de seus componentes!','neutra'),
    ('Aquela festa parece legal... Ninguém te convidou','neutra'),
    ('Uma mulher te cantou...era um travesti','neutra'),
    ('Se você vai se atrasar, então se atrase de verdade. Não dois minutinhos. '
     'Atrase-se uma hora e saboreie seu café da manhã!','neutra'),
    ('Sua cara metade virou gay!','neutra'),
    ('Faça um elogio. Não receba outro em troca','neutra'),
    ('O único lugar onde o sucesso vem antes do trabalho é no dicionário','neutra'),
    ('Ela disse que te ama. “Desculpe, janela errada','neutra'),
    ('Acaricie sua mão. Finja que é de uma garota','neutra'),
    ('Seu fim de semana. Também conhecido como “fail de semana”','neutra'),
    ('Uma mulher te abraçou". Era sua mãe','neutra'),
    ('Se A é o sucesso, então A é igual a X mais Y mais Z. O trabalho é '
     'X; Y é o lazer; e Z é manter a boca fechada.','neutra'),
    ('Seu amigo lembrou de você. Dinheiro emprestado','neutra'),
    ('Está tudo dando certo. Ai você acorda','neutra'),
    ('Esta é sua nova prioridade','neutra'),
    ('Tem muita gente que gostaria de ter o seu emprego','neutra'),
    ('Por vezes sentimos que aquilo que fazemos não é senão uma gota'
     ' de água no mar. Mas o mar seria menor se lhe faltasse uma gota.','neutra'),
    ('O mundo é como um espelho que devolve a cada pessoa o reflexo de seus próprios'
     ' pensamentos e seus atos. A maneira como você encara a vida é que faz toda '
     'diferença. A vida muda, quando "você muda.','neutra'),
    ('Se quer ir rápido, vá sozinho. Se quer ir longe, vá em grupo','neutra'),
    ('Sempre que lhe perguntarem se você sabe fazer um trabalho, diga que '
     'sim e apresse-se em descobrir como executá-lo','neutra'),
    ('Não importa aonde você está, e sim aonde quer chegar','neutra'),
    ('Dia dos Namorados, Dia do Amigo. Um dia inventam uma data de algo'
     ' que você tenha','neutra'),
    ('Você não é pago para pensar','neutra'),
    ('Eu só contrato pessoas que pensam como eu','neutra'),
    ('Não faça perguntas. Apenas faça o que estou dizendo','neutra'),

]

# data set de testes
test_set = [

    ('O ponto de partida de qualquer conquista é o desejo', 'positivo'),
    ('O sucesso é a soma de pequenos esforços repetidos dia após dia', 'positivo'),
    ('Todo progresso acontece fora da zona de conforto', 'positivo'),
    ('Coragem é a resistência e o domínio do medo, não a ausência dele', 'positivo'),
    ('Só evite fazer algo hoje se você quiser morrer e deixar assuntos inacabados', 'positivo'),
    ('O único lugar em que o sucesso vem antes do trabalho é no dicionário', 'positivo'),
    ('Sonhar grande e sonhar pequeno dá o mesmo trabalho', 'positivo'),
    ('Embora ninguém possa voltar atrás e começar tudo de novo,'
     ' qualquer um pode ter um ótimo final', 'positivo'),
    ('Descobri que, se você tem vontade de viver e curiosidade, '
     'dormir não é a coisa mais importante', 'positivo'),
    ('Daqui a vinte anos, você não terá arrependimento das coisas que fez, '
     'mas das que deixou de fazer. Por isso, veleje longe do seu porto seguro. '
     'Pegue os ventos. Explore. Sonhe. Descubra', 'positivo'),
    ('O primeiro passo rumo ao sucesso é dado quando você se recusa a ser um'
     ' refém do ambiente em que se encontra', 'positivo'),
    ('Sempre que você se encontrar do lado da maioria, é hora de parar e refletir',
     'positivo'),
    ('Continue andando. Haverá a chance de você ser barrado por um obstáculo, '
     'talvez por algo que você nem espere. Mas siga, até porque eu nunca ouvi '
     'falar de ninguém que foi barrado enquanto estava parado', 'positivo'),
    ('Se você realmente quer algo, não espere. Ensine você mesmo a ser impaciente',
     'positivo'),
    ('e você quer uma mudança permanente, pare de focar no tamanho de seus problemas'
     ' e comece a focar no seu tamanho', 'positivo'),
    ('Pessoas de sucesso fazem o que pessoas mal sucedidas não querem fazer. '
     'Não queira que a vida seja mais fácil. Deseje que você seja ainda melhor',
     'positivo'),
    ('A primeira razão para o fracasso de alguém é escutar amigos, família e'
     ' vizinhos', 'positivo'),
    ('O sucesso não consiste em não errar, mas não cometer os mesmos equívocos'
     ' mais de uma vez', 'positivo'),
    ('A motivação é o que faz o empreendedor começar e o hábito é o que nos'
     ' faz continuar', 'positivo'),
    ('Nosso maior medo não deve ser o fracasso, mas ser bem-sucedido em algo'
     ' que não importa', 'positivo'),
    ('Se você não traçou um plano para você mesmo, é possível que você caia no '
     'plano de outra pessoa. E adivinha o que ele planejou para você? Não muito',
     'positivo'),
    ('Você deve lutar mais de uma batalha para se tornar um vencedor', 'positivo'),
    ('Eu devo meu sucesso a meu hábito de respeitosamente ouvir conselhos'
     ' e fazer exatamente o contrário', 'positivo'),
    ('Fujões nunca vencem. Vencedores nunca fogem. Mas aqueles que nunca vencem e '
     'continuam sem fugir são idiotas!','negativo'),
    ('Finalmente ela olhou pra você. E te achou feio','negativo'),
    ('Uma jornada de milhares de quilômetros, as vezes termina mal. '
     'Muito mal','negativo'),
    ('Sonhos são como arco-íris. Somente os bobos correm atrás deles!','negativo'),
    ('Você venceu seus inimigos. Playstation!','negativo'),
    ('Está tudo dando certo". Ai você acorda','negativo'),
    ('Ambição. Uma jornada de milhares de quilômetros, as vezes termina mal. '
     'Muito mal!','negativo'),
    ('Bloqueie suas fotos no Facebook!. Evite que mais pessoas sintam nojo de '
     'você!','negativo'),
    ('Ele ainda se lembra de você". Com raiva','negativo'),
    ('Deixe Namorando no Facebook". Engane as pessoas','negativo'),
    ('Se você não conseguir de primeira, remova todas as evidências que'
     ' você ao menos tentou!','negativo'),
    ('Tudo saiu Errado!!','negativo'),
    ('Ainda não existe maquiagem para disfarçar os olhos fundos, '
     'tristes e desmotivados','negativo'),
    ('Saboreie os momentos sob os holofotes. Eles não irão durar muito','negativo'),
    ('Um vendedor deprimido ou desmotivado jamais alcançará a sua meta '
     'profissional, porque se agarra a um passado de lembranças e ações frustradas','negativo'),
    ('Existem aqueles que tentam lhe desmotivar com absurdas palavras. '
     'E eles mau sabem que você os acham uns tremendos idiotas delinquentes','negativo'),
    ('A gestão de pessoas tem que lidar com colaboradores e pessoas desmotivadas'
     ' e intransigentes as mudanças, fazendo com que a ingestão cógnita por falta'
     ' de capacidades, habilidades e atitudes venham ser usadas como um agente'
     ' patogênico e venenoso para a organização e seu gestor.','negativo'),
    ('Por menor que seja a iluminação para sua inspiração, '
     'jamais se entregue para desmotivação se escondendo na escuridão','negativo'),
    ('Uma pessoa desmotivada se torna um peso demasiado grande para outra carregar. '
     'E o não que recebemos não é apenas para nos desmotivar, para confirmar'
     ' as nossas crenças limitantes que alimentamos durante nossa vida','negativo'),
    ('Desmotivar o interesse de outros por nós mesmos não os valorizarmos é '
     'como sentenciar nossa própria alma a uma eternidade de solidão.','negativo'),
    ('Mesmo sabendo que não estamos sozinhos, há um universo de pessoas vivendo '
     'os mesmos dramas, ainda sim somos capazes de pensar que ninguém na face'
     ' da terra está, neste exato momento, mais triste do que nós.','negativo'),
    ('Não desanime com uma derrota. Amanhã tem mais','negativo'),
    ('Talvez as coisas mudem. Para a pior.','negativo'),
    ('Nada é tão horrivel, que não possa piorar muito.','negativo'),
('Estar vivo é estar aberto a todo tipo de sentimentos e sonhos...','neutra'),
    ('A vida nos ensina todos os dias, novas lições. Que sempre possamos estar com'
     ' o coração e a mente abertos para aprendermos','neutra'),
    ('Amigos verdadeiros são aqueles que estão em todos os momentos da sua vida','neutra'),
    ('Nem todo mundo será Astronauta quando crescer','neutra'),
    ('Faça um bom uso do objeto cilíndrico que você deve ter, sob a sua mesa!','neutra'),
    ('Se você não conseguir de primeira, remova todas as evidências de '
     'que você ao menos tentou!','neutra'),
    ('A Inabilidade de um Time, tem um impacto muito maior que a soma das '
     'inabilidades individuais de seus componentes!','neutra'),
    ('Aquela festa parece legal... Ninguém te convidou','neutra'),
    ('Uma mulher te cantou...era um travesti','neutra'),
    ('Se você vai se atrasar, então se atrase de verdade. Não dois minutinhos. '
     'Atrase-se uma hora e saboreie seu café da manhã!','neutra'),
    ('Sua cara metade virou gay!','neutra'),
    ('Faça um elogio. Não receba outro em troca','neutra'),
    ('O único lugar onde o sucesso vem antes do trabalho é no dicionário','neutra'),
    ('Ela disse que te ama. “Desculpe, janela errada','neutra'),
    ('Acaricie sua mão. Finja que é de uma garota','neutra'),
    ('Seu fim de semana. Também conhecido como “fail de semana”','neutra'),
    ('Uma mulher te abraçou". Era sua mãe','neutra'),
    ('Se A é o sucesso, então A é igual a X mais Y mais Z. O trabalho é '
     'X; Y é o lazer; e Z é manter a boca fechada.','neutra'),
    ('Seu amigo lembrou de você. Dinheiro emprestado','neutra'),
    ('Está tudo dando certo. Ai você acorda','neutra'),
    ('Esta é sua nova prioridade','neutra'),
    ('Tem muita gente que gostaria de ter o seu emprego','neutra'),
    ('Por vezes sentimos que aquilo que fazemos não é senão uma gota'
     ' de água no mar. Mas o mar seria menor se lhe faltasse uma gota.','neutra'),
    ('O mundo é como um espelho que devolve a cada pessoa o reflexo de seus próprios'
     ' pensamentos e seus atos. A maneira como você encara a vida é que faz toda '
     'diferença. A vida muda, quando "você muda.','neutra'),
    ('Se quer ir rápido, vá sozinho. Se quer ir longe, vá em grupo','neutra'),
    ('Sempre que lhe perguntarem se você sabe fazer um trabalho, diga que '
     'sim e apresse-se em descobrir como executá-lo','neutra'),
    ('Não importa aonde você está, e sim aonde quer chegar','neutra'),
    ('Dia dos Namorados, Dia do Amigo. Um dia inventam uma data de algo'
     ' que você tenha','neutra'),
    ('Você não é pago para pensar','neutra'),
    ('Eu só contrato pessoas que pensam como eu','neutra'),
    ('Não faça perguntas. Apenas faça o que estou dizendo','neutra'),
]



# criando um classificador
cl = NaiveBayesClassifier(train_set)

#criando uma variavel para medicao de precisao
accuracy = cl.accuracy(test_set)

##### frase utilizada na previsao ###

frases = [

    'Persiga um ideal, não o dinheiro. O dinheiro vai acabar indo atrás de você',
    'Você não precisa de uma equipe de 100 pessoas para desenvolver uma ideia',
    'Faça o que você puder, onde você está e com o que você tem',
    'Não faltam oportunidades para você viver do jeito que você quer. O que'
    ' falta é vontade de tomar o primeiro passo',
    'Hoje acordei cedo para ver o sol',
    'Se queres prever o futuro, estuda o passado'
]


def entrada():
    for frase in frases:
        blob = TextBlob(frase, classifier=cl)
        print('Esta frase e de carater: {}'.format(blob.classify()))
        print('Precisao de previsao: {}'.format(accuracy))

entrada()

# blob = TextBlob((f for f in frases),  classifier=cl)
#
#
# ## 0.7692307692307693
#
# print('Esta frase e de carater: {}'.format(blob.classify()))
# print('Precisao de previsao: {}'.format(accuracy))

