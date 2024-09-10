from flask import Flask, render_template, request, url_for, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your TensorFlow model
model_path = 'InceptionV3_75_1_0001.h5'
model = tf.keras.models.load_model(model_path)
dic = {0: 'Babi', 1: 'Sapi'}  # Define your classes accordingly

# Function to predict the class of the uploaded image
def predict_meat(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return dic.get(predicted_class, "Unknown")

# Route for home page
@app.route('/')
def home():
    return render_template('index.html', current_route='home')

# Route for handling image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'my_image' not in request.files:
        return "No file part"
    
    file = request.files['my_image']
    
    if file.filename == '':
        return render_template('index.html', current_route='index')
    
    if file:
        # Save the file to the server
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Predict the class of the uploaded image
        prediction = predict_meat(file_path, model)
        
        # Return the prediction result along with the image path for display
        return render_template('index.html', prediction=prediction, img_path=url_for('uploaded_file', filename=file.filename), current_route='home')

# Route to serve uploaded files (images)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route for How to Use page
@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html', current_route='how_to_use')

# Route for About Us page
@app.route('/about-us')
def about_us():
    return render_template('about_us.html', current_route='about_us')

if __name__ == '__main__':
    app.run(debug=True)
