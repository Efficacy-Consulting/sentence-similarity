from flask import Flask, request
from flask_cors import CORS

import json
from json import JSONEncoder

import time

import pandas as pd

from app.src.utils import print_with_time, get_source_type, get_sentence_similarity_dict, put_sentence_similarity_dict, read_from_datafile
from app.src.local_file import read_sentence_similarity_from_file, write_to_sentence_similarity_file
from app.src.aws_s3 import download_from_s3, upload_to_s3, read_sentence_similarity_from_aws_s3, write_sentence_similarity_to_aws_s3

from app.src.get_index_files import get_index_files

from app.src.doc_indexing import create_index, load_index
from app.src.search import search_query
from app.src.find_similar_docs import find_similar_document


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config['DEBUG'] = True

# globals
VECTOR_SIZE = 512
default_k = 10
default_batch_size = 32
default_num_trees = 10

# g_columns = ['id', 'title', 'publication', 'content']
g_columns = ['GUID', 'CONTENT', 'ENTITY']
g_id_index = 0
g_content_index = 1
g_content_key = 'CONTENT'

g_df_docs = None
g_data_file = None
g_sentence_similarity = None

default_use_model = 'https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed'
default_csv_file_path = 'short-wiki.csv'
model_indexes_path = 'model-indexes/'
model_index_reference_file = 'sentence-similarity.json'
default_index_file = 'wiki.annoy.index'
default_index_filepath = model_indexes_path + default_index_file

@app.route('/', methods=['GET'])
def home():
  return '<h1>Sentense Analysis</h1><p>Simple sentense analysis. Use <a href="https://jaganlal.github.io/ui-sentence-similarity/">ui-sentence-similarity</a></p>'


@app.route('/get-model-indexes', methods=['GET'])
def get_model_indexes():
  print('get_model_indexes')
  result = get_index_files()
  return json.dumps(result)


@app.route('/train', methods=['GET', 'POST'])
def train_model():
  params = request.get_json()
  print('train_model', params)
  result = train(params)
  return json.dumps(result)


@app.route('/search', methods=['GET', 'POST'])
def search_string():
  params = request.get_json()
  print('search_string', params)
  result = search(params)
  return json.dumps(result)


@app.route('/similarity', methods=['POST'])
def predict_sentence():
  params = request.get_json()
  result = predict(params)
  return json.dumps(result)

# methods called from the APIs

def train(params):
  result = {}

  print('Training', params)

  annoy_vector_dimension = VECTOR_SIZE
  data_file_updated = False
  num_trees = default_num_trees
  stop_words = False

  try:
    if params:
      if params.get('vector_size'):
        annoy_vector_dimension = params.get('vector_size')
      if params.get('index_filename'):
        index_filename = params.get('index_filename')
      if params.get('data_file'):
        data_file = params.get('data_file')
      if params.get('data_file_updated'):
        data_file_updated = params.get('data_file_updated')
      if params.get('use_model'):
        use_model = params.get('use_model')
      if params.get('model_name'):
        model_name = params.get('model_name')
      if params.get('stop_words'):
        stop_words = params.get('stop_words')

    # required input params
    if len(index_filename) == 0 or len(data_file) == 0 or len(use_model) == 0 or len(model_name) == 0:
      result = {
          'error': 'Invalid Input'
      }
    else:
      index_params = {
        'annoy_vector_dimension': annoy_vector_dimension,
        'index_filename': index_filename,
        'data_file': data_file,
        'data_file_updated': data_file_updated,
        'use_model': use_model,
        'model_name': model_name,
        'stop_words': stop_words,
        'default_batch_size': default_batch_size,
        'num_trees': num_trees,
        'model_indexes_path': model_indexes_path
      }
      data_frame = read_data(data_file, data_file_updated)
      # this index is the content index
      create_index(index_params, data_frame, g_content_index)
      result = {
        'message': 'Training successful'
      }

  except Exception as e:
    print('Exception in train: {0}'.format(e))
    result = {
      'error': 'Exception in train: {0}'.format(e)
    }

  return result


def search(params):
  result = {}

  print('Search', params)

  annoy_vector_dimension = VECTOR_SIZE
  index_filename = default_index_file

  data_file = default_csv_file_path
  data_file_updated = False
  use_model = default_use_model
  use_updated_model = False
  k = default_k
  stop_words = False

  input_search_string = None
  filter_values = []

  try:
    if params:
      if params.get('search_string'):
        input_search_string = params.get('search_string')
      if params.get('vector_size'):
        annoy_vector_dimension = params.get('vector_size')
      if params.get('index_filename'):
        index_filename = params.get('index_filename')
      if params.get('data_file'):
        data_file = params.get('data_file')
      if params.get('data_file_updated'):
        data_file_updated = params.get('data_file_updated')
      if params.get('use_model'):
        use_model = params.get('use_model')
      if params.get('use_updated_model'):
        use_updated_model = params.get('use_updated_model')
      if params.get('k'):
        k = params.get('k')
      if params.get('stop_words'):
        stop_words = params.get('stop_words')
      if params.get('filter_values'):
        filter_values = params.get('filter_values')

    # required input params
    if len(input_search_string) <= 0 or len(index_filename) == 0 or len(data_file) == 0 or len(use_model) == 0:
      result = {
          'error': 'Invalid Input'
      }
    else:
      search_params = {
        'input_search_string': input_search_string,
        'annoy_vector_dimension': annoy_vector_dimension,
        'index_filename': index_filename,
        'data_file': data_file,
        'data_file_updated': data_file_updated,
        'use_model': use_model,
        'use_updated_model': use_updated_model,
        'k': k,
        'stop_words': stop_words,
        'filter_values': filter_values,
        'model_indexes_path': model_indexes_path
      }
      start_time = time.time()
      data_frame = read_data(data_file, data_file_updated)
      end_time = time.time()
      print_with_time(
          'Time to read data file: {}'.format(end_time-start_time))

      similarities = search_query(search_params, data_frame)
      similar_sentences = []
      for sentence in similarities:
        if len(filter_values) > 0:
          if sentence[2].lower() in filter_values:
            similar_sentences.append({
              'guid': sentence[g_id_index],
              'content': sentence[g_content_index]
            })
        else:
          similar_sentences.append({
            'guid': sentence[g_id_index],
            'content': sentence[g_content_index]
          })
        print(sentence[g_id_index])

      result = {
          'sourceGuid': '000',
          'sourceSentence': input_search_string,
          'similarDocs': similar_sentences
      }

  except Exception as e:
    print('Exception in search: {0}'.format(e))
    result = {
      'error': 'Exception in search: {0}'.format(e)
    }

  return result


def predict(params):
  result = {}

  print('Predict', params)

  annoy_vector_dimension = VECTOR_SIZE
  index_filename = default_index_file

  data_file = default_csv_file_path
  data_file_updated = False
  use_model = default_use_model
  use_updated_model = False
  k = default_k
  stop_words = False

  input_sentence_id = None
  filter_values = []

  try:
    if params:
      if params.get('guid'):
        input_sentence_id = params.get('guid')
      if params.get('vector_size'):
        annoy_vector_dimension = params.get('vector_size')
      if params.get('index_filename'):
        index_filename = params.get('index_filename')
      if params.get('data_file'):
        data_file = params.get('data_file')
      if params.get('data_file_updated'):
        data_file_updated = params.get('data_file_updated')
      if params.get('use_model'):
        use_model = params.get('use_model')
      if params.get('use_updated_model'):
        use_updated_model = params.get('use_updated_model')
      if params.get('k'):
        k = params.get('k')
      if params.get('stop_words'):
        stop_words = params.get('stop_words')
      if params.get('filter_values'):
        filter_values = params.get('filter_values')

    # required input params
    if len(input_sentence_id) <= 0 or len(index_filename) == 0 or len(data_file) == 0 or len(use_model) == 0:
      result = {
          'error': 'Invalid Input'
      }
    else:
      search_params = {
        'input_sentence_id': input_sentence_id,
        'annoy_vector_dimension': annoy_vector_dimension,
        'index_filename': index_filename,
        'data_file': data_file,
        'data_file_updated': data_file_updated,
        'use_model': use_model,
        'use_updated_model': use_updated_model,
        'k': k,
        'stop_words': stop_words,
        'filter_values': filter_values,
        'model_indexes_path': model_indexes_path
      }

      start_time = time.time()
      data_frame = read_data(data_file, data_file_updated)
      end_time = time.time()
      print_with_time(
          'Time to read data file: {}'.format(end_time-start_time))

      print_with_time('Input Sentence id: {}'.format(input_sentence_id))
      params_filter = 'GUID == "' + input_sentence_id + '"'
      input_data_object = data_frame.query(params_filter)
      input_sentence = input_data_object[g_content_key]

      similarities = find_similar_document(search_params, data_frame, input_sentence)

      similar_sentences = []
      for sentence in similarities[1:]:
        if len(filter_values) > 0:
          if sentence[2].lower() in filter_values:
            similar_sentences.append({
              'guid': sentence[g_id_index],
              'content': sentence[g_content_index]
            })
        else:
          similar_sentences.append({
            'guid': sentence[g_id_index],
            'content': sentence[g_content_index]
          })
        print(sentence[g_id_index])

      result = {
        'sourceGuid': input_sentence_id,
        'sourceSentence': input_sentence.values[0],
        'similarDocs': similar_sentences
      }

  except Exception as e:
    print('Exception in predict: {0}'.format(e))
    result = {
      'error': 'Exception in predict: {0}'.format(e)
    }

  return result

def read_data(path, force_download):
  if(get_source_type() == 'aws'):
    download_from_s3(path, force_download)

  global g_df_docs, g_data_file
  if g_df_docs is None or path != g_data_file:
    try:
      g_df_docs = read_from_datafile(path, g_columns)
      g_data_file = path
    except Exception as e:
      print('Exception in read_data: {0}'.format(e))
      raise

  return g_df_docs

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=1975, debug=True)
