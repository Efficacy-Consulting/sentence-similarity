import os
import time

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

from annoy import AnnoyIndex
from gensim.parsing.preprocessing import remove_stopwords

from app.src.utils import print_with_time, get_source_type, get_sentence_similarity_dict, put_sentence_similarity_dict
from app.src.aws_s3 import download_from_s3, upload_to_s3


def create_index(params, data_frame, content_index):
  try:
    annoy_vector_dimension = params.get('annoy_vector_dimension')
    index_filename = params.get('index_filename')
    data_file = params.get('data_file')
    data_file_updated = params.get('data_file_updated')
    use_model = params.get('use_model')
    model_name = params.get('model_name')
    stop_words = params.get('stop_words')
    default_batch_size = params.get('default_batch_size')
    num_trees = params.get('num_trees')
    model_indexes_path = params.get('model_indexes_path')

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
    # data_frame = read_data(data_file, data_file_updated)
    content_array = data_frame.to_numpy()
    end_time = time.time()
    print('Read Data Time: {}'.format(end_time - start_time))

    start_time = time.time()
    ann = build_index(annoy_vector_dimension, embedding,
                      default_batch_size, sentences,
                      content_array, stop_words, content_index)
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

  except Exception as e:
    raise


def build_index(annoy_vector_dimension, embedding_fun, batch_size, sentences, content_array, stop_words, content_index):
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
        content = sentence[content_index]
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
  if(get_source_type() == 'aws'):
    upload_to_s3(file_name)


def load_index(annoy_vector_dimension, file_name, force_download):
  annoy_index = AnnoyIndex(annoy_vector_dimension, metric='angular')
  if(get_source_type() == 'aws'):
    download_from_s3(file_name, force_download)

  annoy_index.load(file_name)

  return annoy_index
