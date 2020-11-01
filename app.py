import re
from flask import Flask, jsonify, render_template, request

from keras.models import load_model
# Run this cell to mount your Google Drive.
#from google.colab import drive
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
import tensorflow as tf
from keras import backend as K

app = Flask(__name__)
best_model =  load_model('BalanceNet_T20.h5')
graph = tf.get_default_graph()
with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
"""
@app.route('/init')
def init():
    global best_model
    # load the pre-trained Keras model
    best_model = load_model('BalanceNet1.h5')
    #graph = tf.get_default_graph()
"""  


@app.route('/process',methods= ['POST'])
def process():
    result = ""
    MAX_SEQUENCE_LENGTH = 30
    firstName = request.form['firstName']
    #lastName = request.form['lastName']
    #best_model =  load_model('BalanceNet1.h5')
    
    text = ["" for _ in range(5)]
    text[0] = str(firstName)
    
    
    sequences_test = tokenizer.texts_to_sequences(text)
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    #with graph.as_default():
    global graph
    with graph.as_default():
        y_prob = best_model.predict(data_test)
    lis = list(y_prob[0])
    emot = ["Neutral", "Happy", "Sad", "Hate","Anger"]
    maxi = lis.index(max(lis))
    result = result + '    Neutral: ' + str(y_prob[0][0]) + ' ......Happiness: ' + str(y_prob[0][1]) +' ......Sadness: ' + str(y_prob[0][2]) + ' ......Hatred: ' + str(y_prob[0][3]) + ' ......Anger: ' + str(y_prob[0][4]) + '..................... Prominent Emotion: ' + emot[maxi]
    
    #K.clear_session()
    
    
    #output = firstName + lastName
    if (firstName):
        return jsonify({'output': result})
    return jsonify({'error' : 'Missing data!'})

    
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

"""
@app.route('/', methods = ['POST'])
def index():
    MAX_SEQUENCE_LENGTH = 30
    
    best_model =  load_model('BalanceNet1.h5')
    #data2 = pd.read_csv('train.csv')
    text = request.form['firstName']
    x = text.split(' ')
    y = [int(k) for k in x]
    data_int_t = pad_sequences([y, [], [], [], []], padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    y_prob = best_model.predict(data_test)
    #processed_text = text.upper()
    return jsonify({'request' : str(y_prob[0][0])})
"""
