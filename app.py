from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load the model
with open('./static/model/model.pickle', 'rb') as file:
    clf_loaded = pickle.load(file)

# Home route to render the HTML page
#@app.route('/')
#def index():
    #return render_template('index.html')

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        new_student = pd.DataFrame(data, index=[0])

        # Ensure the correct order of features for the model
        expected_features = clf_loaded.feature_names_in_
        new_student = new_student[expected_features]

        # Make prediction
        prediction = clf_loaded.predict(new_student)

        return jsonify({
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

