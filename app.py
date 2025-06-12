from flask import Flask, request, jsonify
from threading import Lock
import logging

from data_processing import load_and_preprocess_data
from model import train_prophet_model, predict_accident
from visualization import generate_visualization

app = Flask(__name__)
logger = logging.getLogger(__name__)
model = None
model_lock = Lock()

CSV_PATH = 'monatszahlen2505_verkehrsunfaelle_06_06_25.csv'

def initialize_model():
    global model
    try:
        logger.info(f"Loading data from {CSV_PATH}")
        df = load_and_preprocess_data(CSV_PATH)
        logger.info(f"Data loaded successfully. {len(df)} records found. Training model...")
        model = train_prophet_model(df)
        logger.info("Model trained successfully")
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
        raise

@app.before_request
def before_request():
    global model
    with model_lock:
        if model is None:
            try:
                initialize_model()
            except Exception as e:
                logger.error(f"Critical error during initialization: {str(e)}")
                model = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model
        if model is None:
            with model_lock:
                if model is None:
                    try:
                        initialize_model()
                    except:
                        return jsonify({'error': 'Model initialization failed'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        year = data.get('year')
        month = data.get('month')

        if year is None or month is None:
            return jsonify({'error': 'Missing year or month'}), 400

        try:
            year = int(year)
            month = int(month)
        except (TypeError, ValueError):
            return jsonify({'error': 'Year and month must be integers'}), 400

        prediction = predict_accident(model, year, month)
        if prediction is None:
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify({'prediction': prediction})
    except Exception as e:
        logger.error(f"API error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/')
def home():
    return """
    <h1>Traffic Accident Prediction API</h1>
    <p>Send a POST request to /predict with JSON format:</p>
    <pre>
    {
        "year": 2021,
        "month": 1
    }
    </pre>
    <p>You will receive prediction in format:</p>
    <pre>
    {
        "prediction": 31
    }
    </pre>
    """

if __name__ == '__main__':
    import logging
    try:
        logging.basicConfig(level=logging.INFO)
        logger.info("Loading data for visualization...")
        df = load_and_preprocess_data(CSV_PATH)
        logger.info("Generating visualization...")
        viz_image = generate_visualization(df)

        with open('historical_accidents.png', 'wb') as f:
            f.write(viz_image)
            logger.info("Saved historical visualization to historical_accidents.png")

        logger.info("Training model for prediction...")
        model = train_prophet_model(df)
        prediction = predict_accident(model, 2021, 1)
        logger.info(f'Prediction for Jan 2021: {prediction}')

        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5050, debug=True)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)