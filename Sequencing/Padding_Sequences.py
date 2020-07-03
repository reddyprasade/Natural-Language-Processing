# Author:Reddy prasade

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Padding a Sequenec
padded = pad_sequences(sequences, padding ='post')
"""
Padding = by Default padding will be there infront of Sentence, if padding is "Post" means We are Going to add a "zeros" after the Sentence
maxlen = "No of Words should Contain in the Sentence"
truncating = "post"  or "pre"

"""


print(word_index)
print(sequences)
print(padded)

