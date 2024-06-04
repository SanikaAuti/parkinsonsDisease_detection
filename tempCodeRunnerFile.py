import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

app = Flask(__name__)

# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

model = pickle.load(open('Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]

    prediction = model.predict(final_features)
    print("final features", final_features)
    print("prediction:", prediction)
    output = round(prediction[0], 2)
    print(output)
    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT DOES NOT HAVE PARKINSONS DISEASE')
    else:
        return render_template('index.html', prediction_text='THE PATIENT HAS PARKINSONS DISEASE')

@app.route('/predict_api', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)


