# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         features = [float(x) for x in request.form.values()]
#         final_features = [np.array(features)]
#         prediction = model.predict(final_features)
#         return render_template('index.html', prediction_text=f'Predicted Pumpkin Seed Type: {prediction[0]}')
#     except Exception as e:
#         return render_template('index.html', prediction_text=f'Error: {str(e)}')

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # Load pickled model components
# model = pickle.load(open('model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))
# label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract input features from the form
#         features = [float(x) for x in request.form.values()]
#         features_array = np.array([features])

#         # Scale features
#         scaled_features = scaler.transform(features_array)

#         # Make prediction
#         prediction = model.predict(scaled_features)

#         # Decode label
#         decoded_prediction = label_encoder.inverse_transform(prediction)

#         return render_template(
#             'index.html',
#             prediction_text=f'Predicted Pumpkin Seed Type: {decoded_prediction[0]}'
#         )

#     except Exception as e:
#         return render_template('index.html', prediction_text=f'Error: {str(e)}')

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("pumpkin_rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        features = [float(request.form[key]) for key in request.form]
        data = np.array([features])

        # Make prediction
        prediction = model.predict(data)
        class_name = label_encoder.inverse_transform(prediction)[0]

        return render_template("index.html", prediction_text=f'Predicted Pumpkin Seed Type: {class_name}')
    except Exception as e:
        return render_template("index.html", prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
