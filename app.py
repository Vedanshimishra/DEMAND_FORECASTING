from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Load the trained model
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create Flask app
app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the 'templates' folder

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get features from form
    try:
        # Retrieve features from the form, converting each to a float
        features = [
            float(request.form['id']),
            float(request.form['week']),
            float(request.form['center_id']),
            float(request.form['meal_id']),
            float(request.form['checkout_price']),
            float(request.form['base_price']),
            float(request.form['emailer_for_promotion']),
            float(request.form['homepage_featured'])
        ]

        # Convert the features into a numpy array (reshape as expected by the model)
        input_data = np.array(features).reshape(1, -1)

        # Make a prediction using the model
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction=prediction)
    
    except Exception as e:
        # Return an error message in case of failure
        return render_template('index.html', prediction=f"Error: {e}")

# Entry point for the application

    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))  # Use the PORT provided by Render or default to 5000
        app.run(host="0.0.0.0", port=port, debug=True)
