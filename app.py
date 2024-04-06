from flask import Flask, request, jsonify
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS
# Load the trained model and CountVectorizer
model = pickle.load(open('fitted_model.pkl', 'rb'))
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    messages = data['messages']
    if not isinstance(messages, list):
        messages = [messages]
    # Vectorize the text data using the CountVectorizer
    x_new = cv.transform(messages)

    # Make predictions using the trained model
    predictions = model.predict(x_new)

    # Map predicted labels back to their original categories

    # Prepare the response JSON
    response = {'predictions': predictions.tolist()}

    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
