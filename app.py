from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create Flask app
app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML form to input features

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get features from form
    try:
        features = [float(request.form['id']),
                    float(request.form['week']),
                    float(request.form['center_id']),
                    float(request.form['meal_id']),
                    float(request.form['checkout_price']),
                    float(request.form['base_price']),
                    float(request.form['emailer_for_promotion']),
                    float(request.form['homepage_featured'])]

        # Convert the features into an array (same format as your model expects)
        input_data = np.array(features).reshape(1, -1)

        # Predict using the trained model
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
