import pandas as pd
import re
import logging
from prophet import Prophet

logger = logging.getLogger(__name__)

def train_prophet_model(df):
    try:
        alkohol_pattern = re.compile(r'alkoholunf[\w]*lle', re.IGNORECASE)
        alcohol_df = df[
            df['MONATSZAHL'].apply(lambda x: bool(alkohol_pattern.search(x)) if isinstance(x, str) else False) &
            (df['AUSPRAEGUNG'].str.strip().str.lower() == 'insgesamt')
        ].copy()

        if alcohol_df.empty:
            logger.error("No data found for Alkoholunfälle insgesamt")
            similar_categories = df[df['MONATSZAHL'].str.contains('alkohol', case=False, na=False)]
            logger.info(f"Similar categories found: {similar_categories['MONATSZAHL'].unique()}")
            logger.info(f"Available types: {df['AUSPRAEGUNG'].unique()}")
            raise ValueError("No data available for specified category and type")

        logger.info(f"Found {len(alcohol_df)} records for Alkoholunfälle insgesamt")

        alcohol_df['ds'] = pd.to_datetime(
            alcohol_df['JAHR'].astype(str) + '-' +
            alcohol_df['MONAT'].astype(str).str.zfill(2) + '-01'
        )
        alcohol_df = alcohol_df.rename(columns={'WERT': 'y'})
        alcohol_df = alcohol_df.sort_values('ds')

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.fit(alcohol_df[['ds', 'y']])

        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        raise

def predict_accident(model, year, month):
    try:
        future_date = pd.to_datetime(f'{year}-{str(month).zfill(2)}-01')
        future_df = pd.DataFrame({'ds': [future_date]})
        forecast = model.predict(future_df)
        return int(round(forecast['yhat'].values[0]))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return None