import re
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import tokenizer_from_json
# from keras_preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
import json
# from keras_preprocessing.sequence import pad_sequences

class Utils:
                  
    max_length = 120
    trunc_type='post'
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    def text_clean(self, text):    
        #remove links | Remove Unicode characters | \\n | only alphabets
        #patter = re.compile(r'https?:\/\/.*[\r\n]*|[^\x00-\x7F]*|\\n')
        patter_url = re.compile(r'https?://\S+|\\n')
        text = re.sub(patter_url, '', text)
        patter_hastag = re.compile('#([a-zA-Z0-9_]{1,50})')
        text = re.sub(patter_hastag, '', text)
        patter_unicode = re.compile('[^\x00-\x7F]*')
        text = re.sub(patter_unicode, '', text)
        #Filter to allow only alphabets
        text = re.sub(r'[^a-zA-Z\']', ' ', text)
        #Converting Text to Lowercase
        text = text.lower()
        #lemmatize text
        text = [self.lemmatizer.lemmatize(token) for token in text.split(" ")]
        text = [self.lemmatizer.lemmatize(token, "v") for token in text]
        #remove stop words
        text = [word for word in text if not word in self.stop_words]
        text = " ".join(text)
        
        return text
    
    
    def generate_padsequeces(self, text, tokenizer_config):
        # Generate and pad the test sequences
        json_file = open(tokenizer_config, 'r')
        loaded_tokenizer = json.loads(json_file.read())
        tokenizer = tokenizer_from_json(loaded_tokenizer)
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence,maxlen=self.max_length,  truncating=self.trunc_type)

        return padded
