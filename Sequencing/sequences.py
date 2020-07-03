# Author:Reddy prasade

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ["I love my dog",
                         "I love my cat",
                         "You Love my Dog!",
                         "Do you thing my dog is amazing"
                         ]

## oov_token="<OOV>" this Trick used for unreconized word in Sentence
tokenizer=Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index


## Sequencing - Turning sentences into data (NLP Zero to Hero - Part 2)
sequences = tokenizer.texts_to_sequences(sentences)


print(word_index)
print(sequences)


test_data = ["i really love my dog",
             "my dog love my manager"]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
