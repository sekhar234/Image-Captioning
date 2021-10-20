import string
from PIL import Image
import numpy as np
import os
from pickle import dump, load
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
#import jsonify
import requests
import pickle
import re


from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
# small library for seeing the progress of loops.
#from tqdm import tqdm_notebook as tqdm
#tqdm().pandas()


import matplotlib.pyplot as plt
import argparse

app = Flask(__name__)

count=0
max_caption_length = 32

tokenizer = pickle.load(open('tokenizer.p','rb'))
"""
#vocab_size = len(tokenizer) + 1

XceptionModel = Xception(include_top=False, pooling="avg")
#photo = extract_features(img_path, xception_model)
#img = Image.open(img_path)

model_weights_save_path = 'model_13.h5'
predictionModel = load_model(model_weights_save_path)"""

@app.route('/')
def home():
    return render_template('result.html')



def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
  return None


@app.route('/predictCaption',methods=['POST'])

def predictCaption():
    image = request.files['img']
    print(image)
    model = load_model('model_13.h5')
    tokenizer = load(open("tokenizer.p","rb"))
    XceptionModel = Xception(include_top=False, pooling="avg")
    photo = extract_features(image, XceptionModel)
    max_length = 32
    #url = request.form['imageSource']
    #XceptionModel = Xception(include_top=False, pooling="avg")

    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    predict = re.sub(r'\b(\w+)( \1\b)+', r'\1', final)
        
    return render_template('result.html', prediction_text=final)



    
if __name__ == '__main__':
    app.run()
