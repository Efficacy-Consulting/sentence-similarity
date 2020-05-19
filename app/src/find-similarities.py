from flask import Flask, request
from flask_cors import CORS

import json
from json import JSONEncoder

import os
import time
import sys

from math import*
from decimal import Decimal

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import pandas as pd

from annoy import AnnoyIndex

from gensim.parsing.preprocessing import remove_stopwords

import boto3
from io import StringIO, BytesIO

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

app.config['DEBUG'] = True

# globals
VECTOR_SIZE = 512
g_df_docs = None
g_data_file = None
g_sentence_similarity = None
# 'local', 'aws'
g_source_type = 'aws'
g_s3_sentence_similarity_file = 'model-indexes/sentence-similarity.json'
g_s3_connection = None

default_use_model = 'https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed'
default_csv_file_path = 'short-wiki.csv'
model_indexes_path = 'model-indexes/'
model_index_reference_file = 'sentence-similarity.json'
default_index_file = 'wiki.annoy.index'
default_index_filepath = model_indexes_path + default_index_file
default_k = 10
default_batch_size = 32
default_num_trees = 10


class SimilarityResult:
    def __init__(self, sourceGuid, sourceSentence, similarDocs):
        self.sourceGuid = sourceGuid
        self.sourceSentence = sourceSentence
        self.similarDocs = similarDocs


class DocType:
    def __init__(self, guid, content):
        self.guid = guid
        self.content = content


class SimilarityResultEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


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
    return json.dumps(result, cls=SimilarityResultEncoder)


@app.route('/similarity', methods=['POST'])
def predict_sentence():
    params = request.get_json()
    result = predict(params)
    return json.dumps(result, cls=SimilarityResultEncoder)


@app.route('/similarity2', methods=['POST'])
def predictv2_sentence():
    params = request.get_json()
    result = predict2(params)
    return json.dumps(result, cls=SimilarityResultEncoder)

# methods called from the APIs


def get_index_files():
    result = None

    try:
        sentence_similarity_dict = get_sentence_similarity_dict()
        result = list(sentence_similarity_dict.values())

    except Exception as e:
        print('Exception in get_index_files: {0}'.format(e))
        result = {
            'error': 'Failure in get_index_files' + e.message
        }

    return result


def train(params):
    result = {}

    print('Training', params)

    annoy_vector_dimension = VECTOR_SIZE
    index_filename = default_index_file

    data_file = default_csv_file_path
    data_file_updated = False
    use_model = default_use_model
    num_trees = default_num_trees
    model_name = index_filename
    stop_words = False

    try:
        if params:
            if params.get('vector_size'):
                annoy_vector_dimension = params.get('vector_size')
            if params.get('index_filename'):
                index_filename = params.get('index_filename')
                model_name = index_filename
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

        start_time = time.time()
        embed_func = hub.Module(use_model)
        end_time = time.time()
        print_with_time('Load the module: {}'.format(end_time-start_time))

        start_time = time.time()
        sentences = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
        embedding = embed_func(sentences)
        end_time = time.time()
        print_with_time(
            'Init sentences embedding: {}'.format(end_time-start_time))

        start_time = time.time()
        data_frame = read_data(data_file, data_file_updated)
        content_array = data_frame.to_numpy()
        end_time = time.time()
        print('Read Data Time: {}'.format(end_time - start_time))

        start_time = time.time()
        ann = build_index(annoy_vector_dimension, embedding,
                          default_batch_size, sentences, content_array, stop_words)
        end_time = time.time()
        print('Build Index Time: {}'.format(end_time - start_time))

        ann.build(num_trees)

        # create model_indexes folder if it doesn't exist
        if not os.path.exists(model_indexes_path):
            os.makedirs(model_indexes_path)

        save_index(ann, model_indexes_path + index_filename)

        sentence_similarity_dict = get_sentence_similarity_dict()
        if(sentence_similarity_dict.get(index_filename) != None):
            sentence_similarity_dict.pop(index_filename, None)

        sentence_similarity_dict[index_filename] = {
            'model_name': model_name, 'data_file': data_file,
            'index_filename': index_filename, 'use_model': use_model,
            'vector_size': annoy_vector_dimension, 'stop_words': stop_words
        }
        put_sentence_similarity_dict(sentence_similarity_dict)

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

        if len(input_search_string) <= 0:
            print_with_time(
                'Search String: {}'.format(input_search_string))
            result = {
                'error': 'Invalid Input Search String'
            }
            return result

        start_time = time.time()
        annoy_index = AnnoyIndex(annoy_vector_dimension, metric='angular')
        load_index(annoy_index, model_indexes_path +
                   index_filename, use_updated_model)
        end_time = time.time()
        print_with_time(
            'Annoy Index load time: {}'.format(end_time-start_time))

        start_time = time.time()
        data_frame = read_data(data_file, data_file_updated)
        content_array = data_frame.to_numpy()
        end_time = time.time()
        print_with_time(
            'Time to read data file: {}'.format(end_time-start_time))

        start_time = time.time()
        embed_func = hub.Module(use_model)
        end_time = time.time()
        print_with_time('Load the module: {}'.format(end_time-start_time))

        start_time = time.time()
        sentences = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
        embedding = embed_func(sentences)
        end_time = time.time()
        print_with_time(
            'Init sentences embedding: {}'.format(end_time-start_time))

        start_time = time.time()
        sess = tf.compat.v1.Session()
        sess.run([tf.compat.v1.global_variables_initializer(),
                  tf.compat.v1.tables_initializer()])
        end_time = time.time()
        print_with_time(
            'Time to create session: {}'.format(end_time-start_time))

        if stop_words:
            input_search_string = remove_stopwords(input_search_string)

        start_time = time.time()
        sentence_vector = sess.run(embedding, feed_dict={
                                   sentences: [input_search_string]})
        nns = annoy_index.get_nns_by_vector(sentence_vector[0], k)
        end_time = time.time()
        print_with_time('nns done: Time: {}'.format(end_time-start_time))

        similar_sentences = []
        similarities = [content_array[nn] for nn in nns]
        for sentence in similarities:
            if len(filter_values) > 0:
                if sentence[2].lower() in filter_values:
                    similar_sentences.append({
                        'guid': sentence[1],
                        'content': sentence[3]
                    })
            else:
                similar_sentences.append({
                    'guid': sentence[1],
                    'content': sentence[3]
                })
            print(sentence[0])

        result = SimilarityResult(
            '000', input_search_string, similar_sentences)

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

        if len(input_sentence_id) <= 0:
            print_with_time('Input Sentence Id: {}'.format(input_sentence_id))
            result = {
                'error': 'Invalid Input id'
            }
            return result

        start_time = time.time()
        annoy_index = AnnoyIndex(annoy_vector_dimension, metric='angular')
        load_index(annoy_index, model_indexes_path +
                   index_filename, use_updated_model)
        end_time = time.time()
        print_with_time(
            'Annoy Index load time: {}'.format(end_time-start_time))

        start_time = time.time()
        data_frame = read_data(data_file, data_file_updated)
        content_array = data_frame.to_numpy()
        end_time = time.time()
        print_with_time(
            'Time to read data file: {}'.format(end_time-start_time))

        start_time = time.time()
        embed_func = hub.Module(use_model)
        end_time = time.time()
        print_with_time('Load the module: {}'.format(end_time-start_time))

        start_time = time.time()
        sentences = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])
        embedding = embed_func(sentences)
        end_time = time.time()
        print_with_time(
            'Init sentences embedding: {}'.format(end_time-start_time))

        start_time = time.time()
        sess = tf.compat.v1.Session()
        sess.run([tf.compat.v1.global_variables_initializer(),
                  tf.compat.v1.tables_initializer()])
        end_time = time.time()
        print_with_time(
            'Time to create session: {}'.format(end_time-start_time))

        print_with_time('Input Sentence id: {}'.format(input_sentence_id))
        params_filter = 'GUID == "' + input_sentence_id + '"'
        input_data_object = data_frame.query(params_filter)
        input_sentence = input_data_object['CONTENT']

        if stop_words:
            input_sentence = remove_stopwords(input_sentence)

        start_time = time.time()
        sentence_vector = sess.run(embedding, feed_dict={
                                   sentences: input_sentence})
        nns = annoy_index.get_nns_by_vector(sentence_vector[0], k)
        end_time = time.time()
        print_with_time('nns done: Time: {}'.format(end_time-start_time))

        similar_sentences = []
        similarities = [content_array[nn] for nn in nns]
        for sentence in similarities[1:]:
            if len(filter_values) > 0:
                if sentence[2].lower() in filter_values:
                    similar_sentences.append({
                        'guid': sentence[0],
                        'content': sentence[1]
                    })
            else:
                similar_sentences.append({
                    'guid': sentence[0],
                    'content': sentence[1]
                })
            print(sentence[0])

        result = SimilarityResult(
            input_sentence_id, input_sentence.values[0], similar_sentences)

    except Exception as e:
        print('Exception in predict: {0}'.format(e))
        result = {
            'error': 'Exception in predict: {0}'.format(e)
        }

    return result


def predict2(params):
    result = {}

    print('Predict2', params)

    annoy_vector_dimension = VECTOR_SIZE
    index_filename = default_index_file

    data_file = default_csv_file_path
    data_file_updated = False
    use_model = default_use_model
    k = default_k

    input_sentence_id = None

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
            if params.get('k'):
                k = params.get('k')

        if len(input_sentence_id) <= 0:
            print_with_time('Input Sentence Id: {}'.format(input_sentence_id))
            result = {
                'error': 'Invalid Input id'
            }
            return result

        start_time = time.time()
        data_frame = read_data(data_file, data_file_updated)
        # content_array = data_frame.to_numpy()
        end_time = time.time()
        print_with_time(
            'Time to read data file: {}'.format(end_time-start_time))

        start_time = time.time()
        # Reduce logging output
        tf.logging.set_verbosity(tf.logging.ERROR)
        embed_func = embed_useT('./use-large-3')
        end_time = time.time()
        print_with_time('Load the module: {}'.format(end_time-start_time))

        print_with_time('Input Sentence id: {}'.format(input_sentence_id))
        params_filter = 'GUID == "' + input_sentence_id + '"'
        input_data_object = data_frame.query(params_filter)
        input_sentence = input_data_object['CONTENT']

        start_time = time.time()
        similarities = recommendTopSentences(
            input_sentence_id, input_sentence.values[0], data_frame, embed_func)
        sentencesRecommended = sorted(
            similarities, key=lambda i: i['score'], reverse=True)
        end_time = time.time()
        print_with_time(
            'recommendTopSentences Time: {}'.format(end_time-start_time))

        similar_sentences = []
        for sentence in sentencesRecommended[:k]:
            params_filter = 'GUID == "' + sentence['guid'] + '"'
            result_data_object = data_frame.query(params_filter)
            result_sentence = result_data_object['CONTENT']

            similar_sentences.append({
                'guid': sentence['guid'],
                'content': result_sentence.values[0],
                'score': sentence['score']
            })

        result = SimilarityResult(
            input_sentence_id, input_sentence.values[0], similar_sentences)
    except Exception as e:
        print('Exception in predict2: {0}'.format(e))
        result = {
            'error': 'Exception in predict2: {0}'.format(e)
        }

    return result


def recommendTopSentences(input_sentence_id, input_sentence, data_frame, embed_func):
    similarities = []
    measures = Similarity()
    # filtered_input_sentence = remove_stopwords(input_sentence)
    input_encoding_matrix = embed_func([input_sentence])
    for index, row in data_frame.iterrows():
        sentence = row['CONTENT']
        guid = row['GUID']
        if(guid.lower() == input_sentence_id.lower()):
            continue
        # filtered_sentence = remove_stopwords(sentence)
        encoding_matrix = embed_func([sentence])
        similarities.append({'score': measures.cosine_similarity(
            input_encoding_matrix[0], encoding_matrix[0]), 'guid': guid})

    return similarities


def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


class Similarity():
    def euclidean_distance(self, x, y):
        """ return euclidean distance between two lists """
        return sqrt(sum(pow(a-b, 2) for a, b in zip(x, y)))

    def manhattan_distance(self, x, y):
        """ return manhattan distance between two lists """
        return sum(abs(a-b) for a, b in zip(x, y))

    def minkowski_distance(self, x, y, p_value):
        """ return minkowski distance between two lists """
        return self.nth_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)),
                             p_value)

    def nth_root(self, value, n_root):
        """ returns the n_root of an value """
        root_value = 1/float(n_root)
        return round(Decimal(value) ** Decimal(root_value), 3)

    def cosine_similarity(self, x, y):
        """ return cosine similarity between two lists """
        numerator = sum(a*b for a, b in zip(x, y))
        denominator = self.square_rooted(x)*self.square_rooted(y)
        return round(numerator/float(denominator), 3)

    def square_rooted(self, x):
        """ return 3 rounded square rooted value """
        return round(sqrt(sum([a*a for a in x])), 3)


# private methods
def print_with_time(msg):
    print('{}: {}'.format(time.ctime(), msg))
    sys.stdout.flush()


def read_data(path, force_download):
    if(g_source_type == 'aws'):
        download_from_s3(path, force_download)

    global g_df_docs, g_data_file
    if g_df_docs is None or path != g_data_file:
        try:
            g_df_docs = pd.read_csv(
                path, usecols=['id', 'title', 'publication', 'content'])
            g_data_file = path
        except Exception as e:
            print('Exception in read_data: {0}'.format(e))
            raise

    return g_df_docs


def build_index(annoy_vector_dimension, embedding_fun, batch_size, sentences, content_array, stop_words):
    ann = AnnoyIndex(annoy_vector_dimension, metric='angular')
    batch_sentences = []
    batch_indexes = []
    last_indexed = 0
    num_batches = 0
    content = ''

    with tf.compat.v1.Session() as sess:
        sess.run([tf.compat.v1.global_variables_initializer(),
                  tf.compat.v1.tables_initializer()])
        for sindex, sentence in enumerate(content_array):
            content = sentence[3]
            if stop_words:
                content = remove_stopwords(sentence[1])

            batch_sentences.append(content)
            batch_indexes.append(sindex)

            if len(batch_sentences) == batch_size:
                context_embed = sess.run(embedding_fun, feed_dict={
                                         sentences: batch_sentences})

                for index in batch_indexes:
                    ann.add_item(index, context_embed[index - last_indexed])
                    batch_sentences = []
                    batch_indexes = []

                last_indexed += batch_size
                if num_batches % 10000 == 0:
                    print_with_time('sindex: {} annoy_size: {}'.format(
                        sindex, ann.get_n_items()))

                num_batches += 1

        if batch_sentences:
            context_embed = sess.run(embedding_fun, feed_dict={
                                     sentences: batch_sentences})
            for index in batch_indexes:
                ann.add_item(index, context_embed[index - last_indexed])

    return ann


def save_index(ann, file_name):
    ann.save(file_name)
    if(g_source_type == 'aws'):
        upload_to_s3(file_name)


def load_index(annoy_index, file_name, force_download):
    if(g_source_type == 'aws'):
        download_from_s3(file_name, force_download)

    annoy_index.load(file_name)


def get_sentence_similarity_dict():
    global g_sentence_similarity
    if(g_sentence_similarity == None):
        if(g_source_type == 'aws'):
            g_sentence_similarity = read_sentence_similarity_from_aws_s3()
        else:
            g_sentence_similarity = read_sentence_similarity_from_file()

    return g_sentence_similarity


def put_sentence_similarity_dict(sentence_similarity):
    if(g_source_type == 'aws'):
        write_sentence_similarity_to_aws_s3(sentence_similarity)
    else:
        write_to_sentence_similarity_file(sentence_similarity)

    global g_sentence_similarity
    g_sentence_similarity = sentence_similarity


def upload_to_s3(file_name, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = get_s3_connection()
    try:
        response = s3_client.upload_file(
            file_name, get_s3_bucket(), object_name)
    except Exception as e:
        print('Exception in upload_to_s3: {0} for {1} file'.format(
            e, file_name))
        return False
    return True


def download_from_s3(file_name, force_download, object_name=None):
    """Download a file from an S3 bucket

    :param file_name: File to download
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was downloaded, else False
    """

    if os.path.exists(file_name) and force_download == False:
        return True

    # create model_indexes folder if it doesn't exist
    if not os.path.exists(model_indexes_path):
        os.makedirs(model_indexes_path)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = get_s3_connection()
    try:
        response = s3_client.download_file(
            get_s3_bucket(), object_name, file_name)
    except Exception as e:
        print('Exception in download_from_s3: {0} for {1} file'.format(
            e, file_name))
        return False
    return True


def get_s3_connection():
    global g_s3_connection
    s3_connection = g_s3_connection
    if(s3_connection == None):
        try:
            # connect to s3
            s3_connection = boto3.client('s3', aws_access_key_id='',
                                         aws_secret_access_key='')
            g_s3_connection = s3_connection
        except Exception as e:
            print('Exception in get_s3: {0}'.format(e))

    return s3_connection


def get_s3_bucket():
    return 'sentence-similarity-data.s3.us-east-1.amazonaws.com'


def read_sentence_similarity_from_aws_s3():
    sentence_similarity = {}
    try:
        s3 = get_s3_connection()
        obj = s3.get_object(
            Bucket=get_s3_bucket(), Key=g_s3_sentence_similarity_file)

        content = obj['Body'].read().decode('utf-8')
        print(content)

        sentence_similarity = json.loads(content)
    except Exception as e:
        print(
            'Exception in read_sentence_similarity_from_aws_s3: {0}'.format(e))
        sentence_similarity = {}

    return sentence_similarity


def write_sentence_similarity_to_aws_s3(sentence_similarity):
    try:
        s3 = get_s3_connection()
        s3.put_object(Bucket=get_s3_bucket(),
                      Key=g_s3_sentence_similarity_file,
                      Body=json.dumps(sentence_similarity).encode())
    except Exception as e:
        print(
            'Exception in write_sentence_similarity_to_aws_s3: {0}'.format(e))


def read_sentence_similarity_from_file():
    sentence_similarity = None
    try:
        global g_sentence_similarity
        if(g_sentence_similarity == None):
            with open(model_indexes_path + model_index_reference_file, 'r') as json_file:
                sentence_similarity = json.load(json_file)

    except Exception as e:
        print('Exception in read_sentence_similarity_from_file: {0}'.format(e))
        sentence_similarity = {}

    g_sentence_similarity = sentence_similarity
    return sentence_similarity


def write_to_sentence_similarity_file(sentence_similarity):
    try:
        if(sentence_similarity != None):
            with open(model_indexes_path + model_index_reference_file, 'w+') as json_file:
                json.dump(sentence_similarity, json_file, indent=2)

    except Exception as e:
        print('Exception in write_to_sentence_similarity_file: {0}'.format(e))
        raise


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1975, debug=True)
