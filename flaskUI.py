# This file creates the UI of the page with flask and links file uploading to the neural network and displays page
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os
import network
import cv2

# configure page
app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
# set up network
script_dir = os.path.dirname(__file__)
network_path = os.path.join(script_dir, 'network.json')
net = network.load(network_path)

# displays the home page
@app.route("/")
def main():
    return render_template("home.html")

# gets the prediction from neural network when image is uploaded and sends user to prediction page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        prediction = network.predict(net, file_path) 

        return render_template('display.html', filename=filename, prediction=prediction)
    
    # return to home page if nothing is uploaded
    return render_template("home.html")

if __name__ == '__main__':
    app.run(debug=True)
