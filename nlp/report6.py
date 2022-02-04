# pip show allennlp
# pip install allennlp
# pip show spacy
# pip install --user -U nltk
# pip show allennlp
# !pip install allennlp==2.1.0 allennlp-models==2.1.0
# !unzip /content/Mini-projects.zip
# pip install allennlp_models


from allennlp.predictors.predictor import  Predictor
import allennlp_models.tagging
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
predictor.predict(
    sentence="Did Uriah honestly think he could beat The Legend of Zelda in"
)

import pandas as pd
df = pd.read_csv("/content/Mini-projects/MSRP/dev.tsv", delimiter)


with open("/content/Mini-projects/MSRP/dev.tsv", "r") as f:
  lines=f.readlines()
line[1]



def verb_object_pairs(sentence):
  print('Sentence: ')
  print(sentence)

  prediction = predictor.predict(sentence=sentence)

  words = prediction['words']
  pred_dependencies = prediction['predicted_dependencies']
  pred_heads = prediction['predicted_heads']

  pairs = []
  for i in range(len(words)):
    if pred_dependencies[i] == 'dobj':
      verb =  words[pred_heads[i]-1] # -1 is bc head indices are one-indexed
      direct_object = words[i]
      pairs.append((verb, direct_object))
  return pairs

print(verb_object_pairs("Take the apple from the table and eat it."))
print(verb_object_pairs("Taunt the dragon before slaying him with my sword."))

