from textblob import TextBlob
from textblob import classifiers
from nltk import *
import nltk

training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]

classifiers = classifiers.NaiveBayesClassifier(training)
# dt_classifiers = classifiers.DecisionTreeClassifier(training)

dt_classifier = nltk.classify.NaiveBayesClassifier.train(training)
print(classifiers.accuracy(testing))
classifiers.show_informative_features(3)