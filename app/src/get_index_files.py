from app.src.utils import get_sentence_similarity_dict
from app.src.aws_s3 import read_sentence_similarity_from_aws_s3
from app.src.local_file import read_sentence_similarity_from_file


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
