import os
import json

# TODO - this must be passed
model_indexes_path = 'model-indexes/'
model_index_reference_file = 'sentence-similarity.json'


def read_sentence_similarity_from_file():
  sentence_similarity = None
  try:
    with open(model_indexes_path + model_index_reference_file, 'r') as json_file:
      sentence_similarity = json.load(json_file)

  except Exception as e:
    print('Exception in read_sentence_similarity_from_file: {0}'.format(e))
    sentence_similarity = {}

  return sentence_similarity


def write_to_sentence_similarity_file(sentence_similarity):
  try:
    if(sentence_similarity != None):
      with open(model_indexes_path + model_index_reference_file, 'w+') as json_file:
        json.dump(sentence_similarity, json_file, indent=2)

  except Exception as e:
    print('Exception in write_to_sentence_similarity_file: {0}'.format(e))
    raise
