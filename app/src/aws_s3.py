import boto3
import os
import json

g_s3_sentence_similarity_file = 'model-indexes/sentence-similarity.json'
g_s3_connection = None

# TODO - this must be passed
model_indexes_path = 'model-indexes/'


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
