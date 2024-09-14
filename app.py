from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Define detailed reports for each disease

disease_reports = {
    "Cataract": "Cataract is a condition where the lens of the eye becomes cloudy, leading to a decrease in vision. It is often treated with surgery.",
    "Glaucoma": "Glaucoma is a group of eye conditions that damage the optic nerve, often due to high intraocular pressure. It can lead to vision loss if untreated.",
    "Diabetic Retinopathy": """Diabetic retinopathy is a complication of diabetes that affects the blood vessels in the retina, potentially leading to vision impairment.

Overview:
Diabetes mellitus refers to a group of diseases that affect how the body uses blood sugar (glucose). Glucose is an important source of energy for the cells that make up the muscles and tissues. It's also the brain's main source of fuel.

The main cause of diabetes varies by type. But no matter what type of diabetes you have, it can lead to excess sugar in the blood. Too much sugar in the blood can lead to serious health problems.

Chronic diabetes conditions include type 1 diabetes and type 2 diabetes. Potentially reversible diabetes conditions include prediabetes and gestational diabetes. Prediabetes happens when blood sugar levels are higher than normal. But the blood sugar levels aren't high enough to be called diabetes. And prediabetes can lead to diabetes unless steps are taken to prevent it. Gestational diabetes happens during pregnancy. But it may go away after the baby is born.
\n
Symptoms
Diabetes symptoms depend on how high your blood sugar is. Some people, especially if they have prediabetes, gestational diabetes or type 2 diabetes, may not have symptoms. In type 1 diabetes, symptoms tend to come on quickly and be more severe.\n \n

Some of the symptoms of type 1 diabetes and type 2 diabetes are:

Feeling more thirsty than usual.
Urinating often.
Losing weight without trying.
Presence of ketones in the urine. Ketones are a byproduct of the breakdown of muscle and fat that happens when there's not enough available insulin.
Feeling tired and weak.
Feeling irritable or having other mood changes.
Having blurry vision.
Having slow-healing sores.
Getting a lot of infections, such as gum, skin and vaginal infections.
Type 1 diabetes can start at any age. But it often starts during childhood or teen years. Type 2 diabetes, the more common type, can develop at any age. Type 2 diabetes is more common in people older than 40. But type 2 diabetes in children is increasing.
\n
When to see a doctor
If you think you or your child may have diabetes. If you notice any possible diabetes symptoms, contact your health care provider. The earlier the condition is diagnosed, the sooner treatment can begin.
If you've already been diagnosed with diabetes. After you receive your diagnosis, you'll need close medical follow-up until your blood sugar levels stabilize.
""",
    "Normal": "The eye appears to be healthy with no detectable conditions.",
    "Hypertension": "Hypertension can lead to retinal changes that might be indicative of underlying health issues. Regular monitoring is essential.",
    "Myopia": "Myopia, or nearsightedness, is a common vision condition where distant objects appear blurry. It can be managed with corrective lenses.",
    "Age Issues": "Age-related conditions such as macular degeneration can affect vision. Regular eye exams can help manage and detect such issues early.",
    "Other": "The condition is not categorized under known diseases. Further examination might be needed."
}


app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('ocular_disease_model.h5')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/services')
def services():
    return render_template('services.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/signin')
def signin():
    return render_template('signin.html')
@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        img = Image.open(file)
        img = img.resize((224, 224))  # Adjust this size to match your model's input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        labels = ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma", "Hypertension", "Myopia", "Age Issues", "Other"]
        predicted_label = labels[predicted_class]
        disease_report = disease_reports.get(predicted_label, "Report not available.")

        return render_template('index.html', prediction=predicted_label, disease_report=disease_report)
    
    
if __name__ == "__main__":
    app.run(debug=True)
