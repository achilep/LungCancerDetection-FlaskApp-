from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'my-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'

classifier = load_model("C:/Users/Mayank/Desktop/flask app/static/model/lung_model1.h5")
graph = tf.get_default_graph()


@app.route("/", methods=['GET', 'POST'])
def index():

    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():

    if request.method == "POST":
        file = request.files['image']
        file_name = secure_filename(file.filename)
        file_loc = "C:/Users/Mayank/Desktop/flask app/static/images/" + file_name
        file.save(file_loc)

        test_image = image.load_img(file_loc, target_size=(217, 217))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        global graph
        with graph.as_default():
            result = classifier.predict(test_image)
            if result[0][0] == 0:
              prediction = "Cancerous"
            else:
              prediction = "Not cancerous"

        return jsonify({"prediction": prediction, "im": "static/images/" + file_name})

    return render_template("index.html")
