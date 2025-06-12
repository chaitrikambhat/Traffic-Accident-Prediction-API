import pandas as pd
import csv
import chardet
import logging

logger = logging.getLogger(__name__)

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
    result = chardet.detect(rawdata)
    return result['encoding']

def load_and_preprocess_data(file_path):
    try:
        encoding = detect_encoding(file_path)
        logger.info(f"Detected encoding: {encoding}")

        rows = []
        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                cleaned_row = [field.strip() for field in row]
                rows.append(cleaned_row)

        headers = rows[0]
        data = rows[1:]
        df = pd.DataFrame(data, columns=headers)

        logger.info(f"Original columns: {df.columns.tolist()}")
        logger.info(f"Original shape: {df.shape}")

        original_cols = ['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT', 'WERT']
        df = df[original_cols].copy()

        for col in original_cols:
            df[col] = df[col].astype(str).str.strip()

        for col in ['MONATSZAHL', 'AUSPRAEGUNG']:
            df[col] = df[col].apply(lambda x: x.encode('latin1').decode('utf-8', 'ignore') if isinstance(x, str) else x)

        df['JAHR'] = pd.to_numeric(df['JAHR'], errors='coerce')
        df['WERT'] = pd.to_numeric(df['WERT'], errors='coerce')
        df = df.dropna(subset=['WERT', 'JAHR'])
        df['JAHR'] = df['JAHR'].astype('Int64')

        df['MONAT'] = df['MONAT'].astype(str).str.strip()
        df = df[df['MONAT'] != 'Summe']

        def extract_month(m):
            try:
                if len(m) == 6 and m.isdigit():
                    return int(m[4:6])
                if m.isdigit() and len(m) > 2:
                    return int(m[-2:])
                return int(m)
            except:
                return None

        df['MONAT'] = df['MONAT'].apply(extract_month)
        df = df.dropna(subset=['MONAT'])
        df['MONAT'] = df['MONAT'].astype(int)
        df = df[(df['MONAT'] >= 1) & (df['MONAT'] <= 12)]
        df = df[df['JAHR'] <= 2020]

        logger.info(f"Processed data shape: {df.shape}")
        logger.info(f"Unique categories: {df['MONATSZAHL'].unique()}")
        logger.info(f"Unique types: {df['AUSPRAEGUNG'].unique()}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        raise