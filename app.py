from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from model.utils import Utils
import logging
from flask.logging import create_logger
from model.attention import AttentionMecanism


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

URI_STRUCTURE = 'model/model_structure.json'
URI_WEIGHTS = 'model/model_weights.h5'
TOKENIZER_CONFIG = 'model/tokenizer_config.json'

def get_model(uri_structure, uri_weights):
     #load model estructure
     json_file = open(uri_structure, 'r')
     loaded_model_json = json_file.read()
     json_file.close()
     clf = model_from_json(loaded_model_json, custom_objects={'AttentionMecanism': AttentionMecanism})
     # load weights into new model
     clf.load_weights(uri_weights)

     return clf

@app.route("/")
def home():
    return "<h3>Sentiment Analysis</h3>"

@app.route("/predict", methods=['POST'])
def predict():
     #load model
     clf = get_model(URI_STRUCTURE, URI_WEIGHTS)
     clf.compile()

     #get text from request
     utils = Utils()
     inference_txt = str(request.json['text'])
     LOG.info(f"JSON payload: {inference_txt}")
     inference_txt = utils.text_clean(inference_txt)
     inference_txt = utils.generate_padsequeces(inference_txt, TOKENIZER_CONFIG)
     #Predict
     prediction = clf.predict(inference_txt)
     label = ''
     index = prediction[0].argmax()
     LOG.info(f"JSON index: {index}")

     if index == 0:
          label = 'Negative Text'
     elif index == 1:
          label = 'Neutral Text'
     else:
          label = 'Positive Text'
     
     LOG.info(f"JSON payload: {label}")
     LOG.info(f"JSON type: {type(label)}")

     return jsonify({'prediction': label})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


