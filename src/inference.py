import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_pipeline.joblib"

pipeline = joblib.load(MODEL_PATH)




def predict(input_data: dict):
    """
    Parameters
    ----------
    input_data: dict
        Dictionary containing patient variables

    Returns
    -------
    dict
        {
            “prediction”: string,
            “probability”: string
        }
    """
    

    X = pd.DataFrame([input_data])

    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0][1]
    
    prdction = ""
    if int(prediction) == 1:
        prdction = "Alive"
    else:
        prdction = "Dead"

    
    return {
        "prediction": str(prdction),
        "probability": str(int (float(probability)*100)) + " %"
    }
