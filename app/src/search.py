import time

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

from gensim.parsing.preprocessing import remove_stopwords

from app.src.utils import print_with_time
from app.src.doc_indexing import load_index

def search_query(params, data_frame):
  try:
    input_search_string = params.get('input_search_string')
    annoy_vector_dimension = params.get('annoy_vector_dimension')
    index_filename = params.get('index_filename')
    data_file = params.get('data_file')
    data_file_updated = params.get('data_file_updated')
    use_model = params.get('use_model')
    use_updated_model = params.get('use_updated_model')
    stop_words = params.get('stop_words')
    k = params.get('k')
    filter_values = params.get('filter_values')
    model_indexes_path = params.get('model_indexes_path')

    start_time = time.time()
    annoy_index = load_index(annoy_vector_dimension, model_indexes_path +
                                            index_filename, use_updated_model)
    end_time = time.time()
    print_with_time(
        'Annoy Index load time: {}'.format(end_time-start_time))

    content_array = data_frame.to_numpy()

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

    similarities = [content_array[nn] for nn in nns]

  except Exception as e:
    raise

  return similarities