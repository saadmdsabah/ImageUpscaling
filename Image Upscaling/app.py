from flask import Flask, render_template, request, jsonify, flash
from keras.models import load_model
from keras.preprocessing.image import load_img
import numpy as np
from PIL import Image

app = Flask(__name__)
value = None
app.secret_key = 'your_secret_key'

btc = load_model(r"ae_bw.h5",compile = False)
deblur = load_model(r"ae_bl.h5",compile = False)
denoise = load_model(r"ae_ny.h5",compile = False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/aboutproject')
def aboutproject():
    return render_template('aboutproject.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html')

@app.route('/process', methods=['POST'])
def process():
    global value
    selected_type = request.form['type']
    value = selected_type
    if(value == None or value == "None"):
        flash("Please select enhancement type :(",category="error")
    else:
        flash("Successfully selected " + value + " enhancement type ;)",category="success")
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    global value
    if value is None or value == "None":
        return jsonify({'error': ""})
    
    uploaded_image = request.files['image']
    uploaded_image.save(f"static/input/{uploaded_image.filename}")

    image = load_img(f"static/input/{uploaded_image.filename}", target_size=(224, 224, 3))
    image = np.array(image)/255.0
    image = np.expand_dims(image,axis = 0)

    img = None
    if(value == "Blur reduction"):
        print("blur")
        img = deblur.predict(image)
    elif(value == "Noise reduction"):
        print("noise")
        img = denoise.predict(image)
    else:
        print("color")
        img = btc.predict(image)

    img = (img[0] * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(f"static/output/{uploaded_image.filename}")

    processed_image_url = f"static/output/{uploaded_image.filename}"
    return jsonify({'result': processed_image_url})

if __name__ == '__main__':
    app.run(debug=True)