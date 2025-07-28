# ✅ Import necessary libraries
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ✅ Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# ✅ Load both models
model_stage1 = load_model('plant_diagnosis_model.h5')
model_stage2 = load_model('plant_diagnosis_model_stage2.h5')

# ✅ Class labels for stage 2
class_labels = [
    'Nutrient Deficiency(iron deficiency) resized',
    'fungal infection (black spots) resized',
    'fungal infection(powdery mildew) resized',
    'sunburned(dried) resized'
]

# ✅ Mapping for clean display
label_map = {
    class_labels[0]: 'Iron Deficiency',
    class_labels[1]: 'Black Spots',
    class_labels[2]: 'Powdery Mildew',
    class_labels[3]: 'Sunburn Dried'
}

# ✅ Prediction function
def predict_plant(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred1 = model_stage1.predict(img_array)[0][0]
    print("Stage 1 score:", pred1)

    if pred1 <= 0.7:
        return f'Healthy (Confidence: {(1 - pred1) * 100:.2f}%)'
    else:
        pred2 = model_stage2.predict(img_array)
        print("Stage 2 prediction scores:", pred2[0])
        class_idx = np.argmax(pred2[0])
        diagnosis = label_map[class_labels[class_idx]]
        confidence = np.max(pred2[0]) * 100
        return f'Not Healthy – {diagnosis} (Confidence: {confidence:.2f}%)'

# ✅ Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    img_path = ''
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = predict_plant(filepath)
        img_path = f'uploads/{filename}'.replace('\\', '/')
    return render_template('index.html', prediction=prediction, image=img_path)

# ✅ Run the app
if __name__ == '__main__':
    app.run(debug=True)
