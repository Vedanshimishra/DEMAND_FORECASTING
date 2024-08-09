from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the form
        weeks = float(request.form['weeks'])
        center_id = float(request.form['center_id'])
        meal_id = float(request.form['meal_id'])
        checkout_price = float(request.form['checkout_price'])
        base_price = float(request.form['base_price'])
        emailer_for_promotion = float(request.form['emailer_for_promotion'])
        homepage_featured = float(request.form['homepage_featured'])
        
        features = np.array([[weeks, center_id, meal_id, checkout_price, base_price, emailer_for_promotion, homepage_featured]])
        prediction = model.predict(features)
        output = round(prediction[0], 2)
        return render_template('index.html.html', prediction_text=f'Predicted Number of Orders: {output}')
    except Exception as e:
        return render_template('index.html.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
