from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import sqlite3

app = Flask(__name__)

# Load the Keras model
model = load_model("final.h5")
db_path = 'database.db'

tumor_types = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
# Function to preprocess the custom input image
def preprocess_custom_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (200, 200))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

# Route for home page
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'my_image' in request.files and request.form['action'] == 'Submit':
            # Handle image selection and prediction
            img = request.files['my_image']
            if img.filename == '':
                return render_template("home.html", error="No image selected")
            
            img_path = "static/" + img.filename
            img.save(img_path)

            # Perform image classification
            # Preprocess the custom input image
            custom_image = preprocess_custom_image(img_path)

            # Make predictions on the custom input image
            predictions = model.predict(np.array([custom_image]))
            predicted_class_index = np.argmax(predictions[0])
            
            predicted_tumor_type = tumor_types[predicted_class_index]

            return render_template("home.html", prediction=predicted_tumor_type, img_path=img_path, uploaded_img=img.filename)

        elif request.form['action'] == 'Enter':
            predicted_tumor_type = request.form['predicted_tumor_type']

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                if predicted_tumor_type == 'Glioma':
                    return render_template('type1.html')
                elif predicted_tumor_type == 'Meningioma':
                    return render_template('type2.html')
                elif predicted_tumor_type == 'Pituitary':
                    return render_template('type3.html')
                else:
                    raise ValueError("Invalid tumor type selected.")

            except ValueError as ve:
                error_message = "Invalid input: " + str(ve)
                return render_template('home.html', error=error_message)

            finally:
                cursor.close()
                conn.close()

    return render_template('home.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    tumor_type = request.form['tumor_type']
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        yo=['Glioma','Meningioma','Pituitary']
        if tumor_type == '1':
            hearing = int(request.form.get('hearing', 0))
            vision = int(request.form.get('vision', 0))
            headache = int(request.form.get('headache', 0))
            weakness = int(request.form.get('weakness', 0))

            select_query = "SELECT GRADE FROM data WHERE hearing=? AND vision=? AND headache_in_morning=? AND weakness_in_one_side_of_body_or_face=?"
            values = (hearing, vision, headache, weakness)
            yoyo=yo[0]

        elif tumor_type == '2':
            hearing = int(request.form.get('hearing', 0))
            vision = int(request.form.get('vision', 0))
            headache = int(request.form.get('headache', 0))
            weakness = int(request.form.get('weakness', 0))

            select_query = "SELECT GRADE FROM data WHERE hearing=? AND vision=? AND headache_in_morning=? AND weakness_in_one_side_of_body_or_face=?"
            values = (hearing, vision, headache, weakness)
            yoyo=yo[1]

        elif tumor_type == '3':
            headache = int(request.form.get('headache', 0))
            weakness = int(request.form.get('weakness', 0))
            blood_pressure = int(request.form.get('blood_pressure', 0))
            blood_sugar = int(request.form.get('blood_sugar', 0))

            select_query = "SELECT GRADE FROM data WHERE headache_in_morning=? AND weakness_in_one_side_of_body_or_face=? AND Blood_Pressure=? AND Blood_Sugar=?"
            values = (headache, weakness, blood_pressure, blood_sugar)
            yoyo=yo[2]

        else:
            raise ValueError("You don't have tumor")

        cursor.execute(select_query, values)
        result = cursor.fetchone()

        if result:
            value = int(result[0])
        else:
            value = None
        
        return render_template('result.html', result=value, type=yoyo)

    except ValueError as ve:
        error_message = "Invalid input: " + str(ve)
        return render_template('home.html', error=error_message)

    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
