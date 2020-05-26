import os
import time
import sys

import pandas as pd

from app.src.aws_s3 import read_sentence_similarity_from_aws_s3, write_sentence_similarity_to_aws_s3
from app.src.local_file import read_sentence_similarity_from_file, write_to_sentence_similarity_file

# 'local', 'aws'
g_source_type = 'local'
g_sentence_similarity = None


def get_source_type():
  return g_source_type


def get_sentence_similarity_dict():
  global g_sentence_similarity
  if(g_sentence_similarity == None):
    if(get_source_type() == 'aws'):
      g_sentence_similarity = read_sentence_similarity_from_aws_s3()
    else:
      g_sentence_similarity = read_sentence_similarity_from_file()

  return g_sentence_similarity


def put_sentence_similarity_dict(sentence_similarity):
  if(get_source_type() == 'aws'):
    write_sentence_similarity_to_aws_s3(sentence_similarity)
  else:
    write_to_sentence_similarity_file(sentence_similarity)

  global g_sentence_similarity
  g_sentence_similarity = sentence_similarity


def read_from_datafile(path, cols):
  try:
    df_docs = pd.read_csv(path, usecols=cols)
  except Exception as e:
    print('Exception in read_data: {0}'.format(e))
    raise

  return df_docs

def print_with_time(msg):
  print('{}: {}'.format(time.ctime(), msg))
  sys.stdout.flush()
