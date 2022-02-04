from sentence_transformers import SentenceTransformer, util
import numpy as np
model = SentenceTransformer('stsb-roberta-large')


# 1	1479718	1479546
# 0	1091286	1091304
# 1	2158854	2158971	Druce will face murder charges, Conte said.	Conte said Druce will be charged with murder.
# 1	2086152	2086347	"We're a quiet, peaceful town of 862 people and nothing ever happens," said Carolyn Greene Bennett, Cedar Grove's town recorder.	"We're a quiet, peaceful town of 862 people and nothing ever happens," Bennett said.
# 0	3107641	3107862	Nursing schools turned away more than 5,000 qualified applicants in the past year because of shortages of faculty and classroom space.	The American Association of Nursing Colleges reported that schools turned away more than 5,000 qualified applicants last year.
# 1	1909331	1909408	"This deal makes good sense for both companies," said Brian L. Halla, National's chairman, president and CEO.	"This deal makes sense for both companies," Halla said in a prepared statement.
# 0	841596	841803	The best the investigators can do is nitpick about the process and substance of isolated business decisions ... and question his competence as a manager."	Ebbers' lawyer, Reid Weingarten, said despite the thorough investigation, "the best the investigators can do is nitpick about the process and substance of isolated business decisions."
# 1	69773	69792	Cisco pared spending to compensate for sluggish sales.	In response to sluggish sales, Cisco pared spending.

# 1
sentence1 = "Cisco pared spending to compensate for sluggish sales."
sentence2 = "In response to sluggish sales, Cisco pared spending."
# 1
sentence3 = '"This deal makes good sense for both companies," said Brian L. Halla, National\'s chairman, president and CEO.'
sentence4 = '"This deal makes sense for both companies," Halla said in a prepared statement.'
# 0
sentence5 = 'The best the investigators can do is nitpick about the process and substance of isolated business decisions ... and question his competence as a manager."'
sentence6 = 'Ebbers\' lawyer, Reid Weingarten, said despite the thorough investigation, "the best the investigators can do is nitpick about the process and substance of isolated business decisions."'
# 0
sentence7 = "Nursing schools turned away more than 5,000 qualified applicants in the past year because of shortages of faculty and classroom space."
sentence8 = "The American Association of Nursing Colleges reported that schools turned away more than 5,000 qualified applicants last year."



# encode sentences to get their embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
embedding3 = model.encode(sentence3, convert_to_tensor=True)
embedding4 = model.encode(sentence4, convert_to_tensor=True)
embedding5 = model.encode(sentence5, convert_to_tensor=True)
embedding6 = model.encode(sentence6, convert_to_tensor=True)
embedding7 = model.encode(sentence7, convert_to_tensor=True)
embedding8 = model.encode(sentence8, convert_to_tensor=True)
# compute similarity scores of two embeddings
cosine_scores1 = util.pytorch_cos_sim(embedding1, embedding2)
cosine_scores2 = util.pytorch_cos_sim(embedding3, embedding4)
cosine_scores3 = util.pytorch_cos_sim(embedding5, embedding6)
cosine_scores4 = util.pytorch_cos_sim(embedding7, embedding8)

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print("Similarity score1,2:", cosine_scores1.item())
print("Sentence 3:", sentence3)
print("Sentence 4:", sentence4)
print("Similarity score:", cosine_scores2.item())
print("Sentence 5:", sentence5)
print("Sentence 6:", sentence6)
print("Similarity score:", cosine_scores3.item())
print("Sentence 7:", sentence7)
print("Sentence 8:", sentence8)
print("Similarity score:", cosine_scores4.item())
