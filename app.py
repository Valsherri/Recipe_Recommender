
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load model
model = load_model('model/hybrid_model.h5')

@app.route('/')
def home():
    return 'Hybrid Recommendation Model API is Running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    user_id = int(data['user_id'])
    item_id = int(data['item_id'])
    prep_feature = float(data['prep_input'])

    user_input = np.array([[user_id]])
    item_input = np.array([[item_id]])
    prep_input = np.array([[prep_feature]])

    prediction = model.predict([user_input, item_input, prep_input])
    return jsonify({'predicted_rating': float(prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
