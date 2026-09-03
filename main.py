from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib
import json


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_model (3).pkl"
SCALER_PATH = BASE_DIR / "scaler (2).pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder (2).pkl"
CONFIG_PATH = BASE_DIR / "feature_config (2).json"


# =========================================================
# LOAD MODEL ARTIFACTS
# =========================================================

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    with open(CONFIG_PATH, "r") as f:
        feature_config = json.load(f)

except Exception as e:
    raise RuntimeError(f"Failed to load ML artifacts: {e}")


# =========================================================
# CONFIG VALUES
# =========================================================

RAW_FEATURES = feature_config["raw_features"]
ENGINEERED_FEATURES = feature_config["engineered_features"]
ALL_FEATURES = feature_config["all_features"]

BEST_MODEL_NAME = feature_config["best_model_name"]
MODEL_NEEDS_SCALING = feature_config["model_needs_scaling"]

MODEL_LABELS = {
    0: "NORMAL",
    1: "ANOMALY"
}


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Burn-in Anomaly Detection API",
    description="ML API for detecting component burn-in anomalies",
    version="1.0"
)


# =========================================================
# INPUT MODEL
# =========================================================

# IMPORTANT:
# These fields must match the values inside RAW_FEATURES
# from your feature_config.json.

class ComponentData(BaseModel):
    initial_measurement: float
    burn_in_24h: float
    burn_in_96h: float
    final_burn_in_reading: float


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def apply_feature_engineering(df_input):

    df_engineered = df_input.copy()

    if "drift_24h" in ENGINEERED_FEATURES:
        df_engineered["drift_24h"] = (
            df_engineered["burn_in_24h"]
            - df_engineered["initial_measurement"]
        )

    if "drift_96h" in ENGINEERED_FEATURES:
        df_engineered["drift_96h"] = (
            df_engineered["burn_in_96h"]
            - df_engineered["initial_measurement"]
        )

    if "total_drift" in ENGINEERED_FEATURES:
        df_engineered["total_drift"] = (
            df_engineered["final_burn_in_reading"]
            - df_engineered["initial_measurement"]
        )

    if "drift_rate" in ENGINEERED_FEATURES:

        if "total_drift" in df_engineered.columns:
            df_engineered["drift_rate"] = (
                df_engineered["total_drift"] / 3
            )

        else:
            df_engineered["drift_rate"] = (
                df_engineered["final_burn_in_reading"]
                - df_engineered["initial_measurement"]
            ) / 3

    if "acceleration" in ENGINEERED_FEATURES:

        if (
            "drift_96h" in df_engineered.columns
            and "drift_24h" in df_engineered.columns
        ):

            df_engineered["acceleration"] = (
                df_engineered["drift_96h"]
                - df_engineered["drift_24h"]
            )

        else:

            drift_96h = (
                df_engineered["burn_in_96h"]
                - df_engineered["initial_measurement"]
            )

            drift_24h = (
                df_engineered["burn_in_24h"]
                - df_engineered["initial_measurement"]
            )

            df_engineered["acceleration"] = (
                drift_96h - drift_24h
            )

    return df_engineered


# =========================================================
# HOME / HEALTH CHECK
# =========================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Burn-in Anomaly Detection API is running",
        "model": BEST_MODEL_NAME
    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():

    return {
        "model": BEST_MODEL_NAME,
        "raw_features": RAW_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "all_features": ALL_FEATURES,
        "scaling_required": MODEL_NEEDS_SCALING
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(data: ComponentData):

    try:

        # ---------------------------------------------
        # 1. CREATE INPUT DATAFRAME
        # ---------------------------------------------

        input_data = {
            feature: getattr(data, feature)
            for feature in RAW_FEATURES
        }

        input_df = pd.DataFrame([input_data])


        # ---------------------------------------------
        # 2. FEATURE ENGINEERING
        # ---------------------------------------------

        processed_df = apply_feature_engineering(input_df)


        # ---------------------------------------------
        # 3. ENSURE CORRECT FEATURE ORDER
        # ---------------------------------------------

        final_features_df = processed_df[ALL_FEATURES]


        # ---------------------------------------------
        # 4. SCALE IF REQUIRED
        # ---------------------------------------------

        if MODEL_NEEDS_SCALING:

            X_final = scaler.transform(final_features_df)

        else:

            X_final = final_features_df.values


        # ---------------------------------------------
        # 5. PREDICTION
        # ---------------------------------------------

        prediction_encoded = model.predict(X_final)


        # ---------------------------------------------
        # 6. ANOMALY PROBABILITY
        # ---------------------------------------------

        anomaly_probability = None

        if hasattr(model, "predict_proba"):

            anomaly_class_index = list(model.classes_).index(1)

            anomaly_probability = float(
                model.predict_proba(X_final)[0][anomaly_class_index]
            )


        # ---------------------------------------------
        # 7. DECODE RESULT
        # ---------------------------------------------

        prediction_value = int(prediction_encoded[0])

        predicted_label = MODEL_LABELS.get(
            prediction_value,
            str(prediction_value)
        )


        # ---------------------------------------------
        # 8. RETURN RESULT
        # ---------------------------------------------

        return {
            "prediction": predicted_label,
            "prediction_code": prediction_value,
            "anomaly_probability": anomaly_probability,
            "model": BEST_MODEL_NAME
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )