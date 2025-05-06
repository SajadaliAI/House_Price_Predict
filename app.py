from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and columns
model = pickle.load(open('price_predict_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['Location']
        area = float(request.form['Area'])
        bathrooms = int(request.form['Bathrooms'])
        bhk = int(request.form['BHK'])

        # Initialize zero array with correct length
        x = np.zeros(len(columns))
        if 'total_sqft' in columns: x[columns.index('total_sqft')] = area
        if 'bath' in columns: x[columns.index('bath')] = bathrooms
        if 'bhk' in columns: x[columns.index('bhk')] = bhk

        # One-hot encode location
        loc_feature = f"location_{location.lower()}"
        if loc_feature in columns:
            x[columns.index(loc_feature)] = 1

        prediction = model.predict([x])[0]
        output = round(prediction, 2)
        return render_template('index.html', prediction_text=f"Estimated Price: $  {output} Lakhs")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
