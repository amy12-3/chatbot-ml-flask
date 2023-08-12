import numpy as np
from flask import Flask, jsonify, request
import pickle

app = Flask(__name__)

@app.post('/predict')
def predict():
    model = pickle.load(open('model.pkl', 'rb'))
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction)
    if(prediction[0]==1):
        prediction = 'on-time'
    else:
        prediction = 'late'
    data = {
        'prediction': prediction,
        'status': 'success'
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port = 3233)
