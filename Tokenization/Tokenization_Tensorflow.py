import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ["I love my dog",
            "I love my cat",
            "You Love my Dog!"]

tokenizer=Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

""" 
Tokenization in Tensorflow are classificed into 4 types

* hashing_trick(...): Converts a text to a sequence of indexes in a fixed-size hashing space.
* one_hot(...): One-hot encodes a text into a list of word indexes of size n.
* text_to_word_sequence(...): Converts a text to a sequence of words (or tokens).
* tokenizer_from_json(...): Parses a JSON tokenizer configuration file and returns a

"""