# Traffic Accident Prediction API - Munich Verkehrsunfälle

This repository contains a step-by-step implementation of a Traffic Accident Prediction API based on historical monthly traffic accident data in Munich. The project covers data loading, preprocessing, visualization, and forecasting future accident counts using Facebook Prophet. The API is built using Flask.

---

## Project Overview

- **Data Source:** Munich Open Data "Monatszahlen Verkehrsunfälle" dataset (CSV)
- **Goal:** Predict the number of alcohol-related traffic accidents in Munich for a given month and year.
- **Approach:** 
  1. Load and preprocess raw CSV data with careful encoding detection.
  2. Visualize historical accident trends.
  3. Train a time-series forecasting model using Prophet.
  4. Expose prediction functionality via a Flask REST API.
  
---

## Repository Structure

The repository contains the original dataset CSV file and generated visualization images.
   
  Contains Python scripts:
  - `data_processing.py` - Functions to load and clean the data.
  - `visualization.py` - Code to generate historical accident trend plots.
  - `model.py` - Model training and prediction logic with Prophet.
  - `app.py` - Flask API exposing `/predict` endpoint and serving predictions.

- `historical_accidents.png`  
  Visualization of historical traffic accident trends (generated by the code).

- `README.md`  
  This documentation.

---

## Step-by-step Development and Commits

1. **Initial data loading and exploration**  
   - Detect file encoding reliably.  
   - Read CSV and clean column data (commit: `Initial data load and preprocessing`).

2. **Data filtering and cleaning**  
   - Remove invalid rows, convert types, handle month parsing.  
   - Filter for relevant years and categories (commit: `Data cleaning and filtering`).

3. **Visualization**  
   - Plot yearly aggregated accidents by category.  
   - Save visualization image (commit: `Add historical accidents visualization`).

4. **Model training**  
   - Extract alcohol-related accidents data.  
   - Train Prophet model with yearly seasonality.  
   - Validate training results (commit: `Train Prophet forecasting model`).

5. **Prediction API**  
   - Develop Flask app with `/predict` POST endpoint.  
   - Implement thread-safe model loading and error handling (commit: `Implement Flask API for predictions`).

6. **Final testing and packaging**  
   - Run full app with visualization, training, and server start.  
   - Document usage and dependencies (commit: `Finalize app and add README`).

---

## How to Run

### Prerequisites

* Python 3.8+
* Install dependencies:

```bash
pip install flask prophet pandas matplotlib chardet
```

## 🙋‍♀️ About Me

Chaitrika Mohan Bhat<br>
Feel free to connect or reach out:

* [LinkedIn](https://www.linkedin.com/in/chaitrika-m-bhat/)
* [Email](mailto:chaitrikambhat@gmail.com)

