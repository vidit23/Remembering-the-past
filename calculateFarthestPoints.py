print("starting script", flush=True)
import json
import spacy_sentence_bert
import pandas as pd
import numpy as np

#import sys
#import codecs

#sys.stdout = codecs.getwriter('utf8')(sys.stdout)
#sys.stderr = codecs.getwriter('utf8')(sys.stderr)

print("Started the program")
with open('./train-v1.1.json') as f:
  data = json.load(f)
print("loaded the dataset", flush=True)

nlp = spacy_sentence_bert.load_model('en_bert_base_nli_cls_token')
print("downloaded model", flush=True)
questionsWithVectors = pd.DataFrame([], columns=["questionID", "questionText", "questionVector", "contextTitle"])
contextWithVectors = pd.DataFrame([], columns=["contextText", "contextVector", "contextTitle"])

tidx = 0
for topic in data["data"]:
    sidx = 0
    tidx += 1
    for samples in topic["paragraphs"]:
        sidx += 1
        print(tidx, sidx, flush=True)
        # contextWithVectors = contextWithVectors.append({"contextText": samples["context"],
        #                                                "contextTitle": topic["title"],
        #                                                "contextVector": nlp(samples["context"]).vector
        #                                               }, ignore_index=True)
        for question in samples["qas"]:
            questionVector = nlp(question["question"])
            questionsWithVectors = questionsWithVectors.append({"questionID": question["id"],
                                         "questionText": question["question"],
                                         "contextTitle": topic["title"],
                                         "questionVector": questionVector.vector
                                        }, ignore_index=True)
    #print("finished the topic", topic["title"], flush=True)

print("Done calculating",flush=True)
questionVectors = np.stack(questionsWithVectors["questionVector"].to_numpy())
print("Converted to numpy",flush=True)

import random
random.seed(42)
from scipy.spatial.distance import cdist

startingIndex = random.randint(0, len(questionVectors))

farthestPointsId = []
farthestQuestions = []
farthestQuestionVectors = np.reshape(questionVectors[startingIndex], (1, -1))
print("Starting the iterations",flush=True)
while len(farthestPointsId) != 20000:
    print("Found these many points "+str(len(farthestPointsId)),flush=True)
    centroidOfQuestions = np.reshape(np.mean(farthestQuestionVectors, axis=0), (1, -1))

    similarities = 1 - cdist(centroidOfQuestions, questionVectors, metric='euclidean')
    pointDistanceSorted = np.argsort(similarities)
    farthestPoint = pointDistanceSorted[0][0] # most dissimilar
#     farthestPoint = pointDistanceSorted[0][::-1][0] # most similar


    farthestPointsId.append(questionsWithVectors.loc[farthestPoint]["questionID"])
    farthestQuestions.append(questionsWithVectors.loc[farthestPoint]["questionText"])
    farthestQuestionVectors = np.append(farthestQuestionVectors,
                                        np.reshape(questionVectors[farthestPoint], (1, -1)),
                                        axis = 0)

    questionVectors = np.delete(questionVectors, obj=farthestPoint, axis=0)
    questionsWithVectors = questionsWithVectors.drop([farthestPoint]).reset_index(drop=True)

print('dumping the file', flush=True)

with open('./trainFarthestQuestions-v1.1.json', 'w') as outfile:
    json.dump(farthestPointsId, outfile)