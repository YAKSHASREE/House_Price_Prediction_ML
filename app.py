
import os
import pickle
import pandas as pd

from pathlib import Path
from flask import Flask, render_template, request
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Profile details
PROFILE_NAME = os.environ.get("PROFILE_NAME", "Your Name")
PROFILE_EMAIL = os.environ.get("PROFILE_EMAIL", "your@email.com")
PROFILE_PHONE = os.environ.get("PROFILE_PHONE", "Your Phone")
PROFILE_IMAGE = os.environ.get("PROFILE_IMAGE", "")


# Load trained model
try:

    model_path = (
        Path(__file__).resolve().parent /
        "house_price_model.pkl"
    )

    with open(model_path, "rb") as file:

        saved_data = pickle.load(file)

    model = saved_data["model"]

    city_mapping = saved_data["city_mapping"]

    statezip_mapping = saved_data["statezip_mapping"]

    country_mapping = saved_data["country_mapping"]

    numeric_features = saved_data["numeric_features"]

    categorical_features = saved_data["categorical_features"]

    best_model_name = saved_data["best_model_name"]

    print("Model loaded successfully!")
    print("Selected Model:", best_model_name)

except Exception as e:

    print("Model Loading Error:", e)
    raise


# Convert location code into original value
def get_location_name(mapping, code):

    for name, value in mapping.items():

        if value == code:

            return name

    return None


# Home page and prediction
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None
    property_details = None
    entered_details = {}

    if request.method == "POST":

        entered_details = request.form.to_dict()

        try:

            # Get location codes
            city_code = int(request.form["city_code"])

            statezip_code = int(
                request.form["statezip_code"]
            )

            country_code = int(
                request.form["country_code"]
            )

            # Get original location names
            city = get_location_name(
                city_mapping,
                city_code
            )

            statezip = get_location_name(
                statezip_mapping,
                statezip_code
            )

            country = get_location_name(
                country_mapping,
                country_code
            )

            # Validate location codes
            if city is None or statezip is None or country is None:

                raise ValueError(
                    "Invalid location code. Please enter valid codes."
                )

            # Collect property details
            house_details = {

                "bedrooms": float(
                    request.form["bedrooms"]
                ),

                "bathrooms": float(
                    request.form["bathrooms"]
                ),

                "sqft_living": float(
                    request.form["sqft_living"]
                ),

                "sqft_lot": float(
                    request.form["sqft_lot"]
                ),

                "floors": float(
                    request.form["floors"]
                ),

                "waterfront": float(
                    request.form["waterfront"]
                ),

                "view": float(
                    request.form["view"]
                ),

                "condition": float(
                    request.form["condition"]
                ),

                "sqft_above": float(
                    request.form["sqft_above"]
                ),

                "sqft_basement": float(
                    request.form["sqft_basement"]
                ),

                "yr_built": float(
                    request.form["yr_built"]
                ),

                "yr_renovated": float(
                    request.form["yr_renovated"]
                ),

                "city_code": city_code,

                "statezip_code": statezip_code,

                "country_code": country_code
            }

            # Arrange features in the same order
            # used during model training
            features = (
                numeric_features +
                categorical_features
            )

            input_data = pd.DataFrame(
                [house_details],
                columns=features
            )

            # Predict house price
            prediction = model.predict(input_data)[0]

            # Validate prediction
            if prediction <= 0:

                raise ValueError(
                    "Invalid predicted house price."
                )

            # Format prediction result
            result = f"${prediction:,.2f}"

            # Location details for webpage
            property_details = {

                "city": city,

                "statezip": statezip,

                "country": country
            }

            print("\nPrediction Successful!")

            print("Predicted Price:", result)

            print("Model Used:", best_model_name)

        except Exception as e:

            error = str(e)

            print("\nPrediction Error:", error)

    # Display webpage
    return render_template(

        "index.html",

        result=result,

        error=error,

        property_details=property_details,

        entered_details=entered_details,

        profile_name=PROFILE_NAME,

        profile_email=PROFILE_EMAIL,

        profile_phone=PROFILE_PHONE,

        profile_photo=PROFILE_IMAGE,

        best_model_name=best_model_name
    )


# Run Flask application
if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=int(os.environ.get("PORT", 5000)),

        debug=False
    )
