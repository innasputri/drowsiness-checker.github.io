from asyncio.windows_events import NULL
from cv2 import Mat
from flask import Flask, flash, render_template, request, redirect, url_for
import urllib.request
from werkzeug.utils import secure_filename

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from keras.preprocessing import image
from keras.models import load_model

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif', 'webp'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = 'static/images/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

reye = cv2.CascadeClassifier('haar_cascade/haarcascade_righteye_2splits.xml')
leye = cv2.CascadeClassifier('haar_cascade/haarcascade_lefteye_2splits.xml')
model = load_model('models/eyedet.h5')
rpred=NULL
lpred=NULL
resized = np.matrix(NULL)

@app.route("/", methods=['GET'])
def upload_form():
    return render_template( "predict_img.html")

@app.route("/", methods=['POST'])
def upload_image():
    #path = "./images/" + file.filename
    #file.save(path)

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        image_std = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = resize(image_std, filename)           
        gray_img = cv2.cvtColor(resize(image, filename), cv2.COLOR_BGR2GRAY)

        right_eye = reye.detectMultiScale(gray_img, 1.06, 30)
        left_eye = reye.detectMultiScale(gray_img, 1.06, 30)
    else:
        flash('Allowed image types are -> png, jpg, jpeg')
        return(redirect.url)

    for (x,y,w,h) in right_eye:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        #eye_output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.imshow(eye_output)

        r_eye=image[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(128, 128))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(128, 128, -1)
        r_eye = np.expand_dims(r_eye,axis=0)
        global rpred
        rpred = model.predict(r_eye)

    for (x,y,w,h) in left_eye:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

        l_eye=image[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(128, 128))
        l_eye = l_eye/255
        l_eye = l_eye.reshape(128, 128, -1)
        l_eye = np.expand_dims(l_eye,axis=0)
        global lpred
        lpred = model.predict(l_eye)

    if(rpred<0.5 and lpred<0.5):
        print(filename + ": Drowsiness Detected")
        flash('Drowsiness Detected')
        return render_template('predict_img.html', filename=filename)
    else:
        print(filename + ": NO Drowsiness Detected")
        flash('NO Drowsiness Detected')
        return render_template('predict_img.html', filename=filename)
    

@app.route("/display/<filename>")
def display_image(filename):
    #print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='images/' + filename), code=301)

def resize(reimage, name):
    size = os.stat(os.path.join(app.config['UPLOAD_FOLDER'], name)).st_size
    print('size is : ' + str(size))
    scale_size = 0.5 # percent of original size
    width = int(reimage.shape[1] * scale_size )
    height = int(reimage.shape[0] * scale_size)
    dimension = (width, height)
    if(app.config['MAX_CONTENT_LENGTH']> size * 700):
        reimage = cv2.resize(reimage, dimension)
        return reimage
    else:
        return reimage

if __name__ == '__main__':
    app.run(debug=True)