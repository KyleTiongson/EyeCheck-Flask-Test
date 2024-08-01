from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('RFModel2.pkl')


@app.route('/')
def index():
    return render_template_string("""
           <!doctype html>
           <title>Flask App</title>
           <h1>Flask App is running</h1>
           <p>To use the predict endpoint, send a POST request to <code>/predict</code> with your raw data.</p>
           """)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data['age']
    eye_grade = data['eye_grade']
    eyeglasses = data['eyeglasses']  # Added eyeglasses category
    directions = data['directions']  # Sequence of directions

    # Prepare the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Eye Grade': [eye_grade],
        'Eyeglasses': [eyeglasses],
        'Directions': [' '.join(directions)]  # Join directions into a single string if needed
    })

    # Convert to the same format as the training data
    input_data = pd.get_dummies(input_data, columns=['Eye Grade', 'Eyeglasses'])

    # Ensure all expected columns are present
    expected_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Make prediction
    predictions = model.predict(input_data)

    # Ensure predictions is an array of strings
    predicted_directions = [str(direction).lower() for direction in predictions[0]]

    return jsonify({'predicted_directions': predicted_directions})


if __name__ == '__main__':
    app.run()
