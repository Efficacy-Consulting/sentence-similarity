[![Sentence Similarity Video](SentenceSimilarity.gif)](https://youtu.be/hM_OoNOE698 "Sentence Similarity Video")

# sentence-similarity
Short wikipedia articles lookup using Google's USE (Universal Sentence Encoder) and Annoy (Approximate Nearest Neighbors Oh Yeah).

### Note: Not the entire wikipedia articles lookup ;). Checkout the disclaimer below

## Installation

### Clone the repo
```
git clone https://github.com/jaganlal/find-similar-sentences.git
cd find-similar-sentences/
```

### Install the required libs
```
pip install -r requirements.txt
```

## Usage

You can directly use the Angular Application (https://jaganlal.github.io/ui-sentence-similarity/) that i have put together to consume this service

This uses `flask` to expose the following endpoints at `port: 1975`

### Train the model
1. Train the model and build Annoy Index with defaults. (`method=POST`)

`localhost:1975/train`

`example`
```
  trainingParams = {
    model_name: 'wiki-annoy-use-2',
    use_model: 'https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed',
    index_filename: 'wiki.annoy.use.2.index',
    data_file: 'short-wiki.csv',
    stop_words: false
  }
```
#### Note: Checkout the code to see other default values

### Search
2.1. Search by string (checkout `short-wiki.csv` for more details).  (`method=POST`)

  `localhost:1975/search/{params}`

  `example`
  ```
    searchParams = {
      search_string: 'new york',
      index_filename: 'wiki.annoy.use.2.index',
      use_model: 'https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed',
      data_file: 'short-wiki.csv',
      vector_size: 'size-of-the-vector' (default: 512)
    }
  ```


#### Note: You have to use the same annoy index and model (that was trained) with for the best results



### Similar Documents
2.2. Find similar documents using `guid` (checkout `short-wiki.csv` for more details on `guid`).  (`method=POST`)

  `localhost:1975/similarity/{params}`

  `example`
  ```
    searchParams = {
      guid: 'music-wikipedia',
      index_filename: 'wiki.annoy.use.2.index',
      use_model: 'https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed',
      data_file: 'short-wiki.csv',
      vector_size: 'size-of-the-vector' (default: 512)
    }
  ```

#### Note: You have to use the same annoy index and model (that was trained) with for the best results

### Get list of trained indexes from model
3. All the indexes (trained with particular model) is stored in the local hard drive inside `model-indexes` folder, this endpoints returns a list object with information about the model and indexes. This information can be used for prediction. (`method=GET`)

    `localhost:1975/get-model-indexes`

## Disclaimer
I started to create (short-wiki.csv) a short intro on some of the articles (source: wikipedia) about places, people, culture etc. So this application will lookup from that articles. Checkout `short-wiki.csv` for more information on this. You can imagine this as a cleaned up data lookup. If you want to contribute (either code or data part), please feel free to fork it and create a PR.

## Note
As you have noticed, there are no error handlings

## Docker Compose
  `docker-compose build`: build the container
  `docker-compose up`: bring it up

## References
https://jaganlal.github.io/wiki-use-annoy

https://jaganlal.github.io/ui-sentence-similarity/
