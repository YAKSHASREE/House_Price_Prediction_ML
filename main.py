
import pandas as pd
import pickle
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


class HousePriceModel:

    def __init__(self):

        self.base_path = Path(__file__).resolve().parent
        self.df = None

        self.city_mapping = {}
        self.statezip_mapping = {}
        self.country_mapping = {}

        self.models = {}
        self.best_model = None
        self.best_model_name = None

        self.numeric_features = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated"
        ]

        self.categorical_features = [
            "city_code",
            "statezip_code",
            "country_code"
        ]

    # Load dataset
    def load_dataset(self):

        try:

            file_path = self.base_path / "House_price_prediction_data.csv"

            self.df = pd.read_csv(file_path)

            print("\nDataset loaded successfully!")
            print("Dataset Shape:", self.df.shape)

        except Exception as e:

            print("Dataset Loading Error:", e)
            raise

    # Preprocessing
    def preprocess_data(self):

        try:

            self.df.drop(
                ["date", "street"],
                axis=1,
                inplace=True
            )

            # City mapping
            cities = sorted(self.df["city"].dropna().unique())

            for i, city in enumerate(cities, start=1):
                self.city_mapping[city] = i

            # State ZIP mapping
            statezips = sorted(
                self.df["statezip"].dropna().unique()
            )

            for i, zipcode in enumerate(statezips, start=1):
                self.statezip_mapping[zipcode] = i

            # Country mapping
            countries = sorted(
                self.df["country"].dropna().unique()
            )

            for i, country in enumerate(countries, start=1):
                self.country_mapping[country] = i

            # Apply mappings
            self.df["city_code"] = self.df["city"].map(
                self.city_mapping
            )

            self.df["statezip_code"] = self.df["statezip"].map(
                self.statezip_mapping
            )

            self.df["country_code"] = self.df["country"].map(
                self.country_mapping
            )

            # Remove missing values
            self.df.dropna(inplace=True)

            # Remove invalid prices
            self.df = self.df[self.df["price"] > 0]

            print("\nPreprocessing completed!")
            print("Final Dataset Shape:", self.df.shape)

            print("\nCITY MAPPING")
            print(self.city_mapping)

            print("\nSTATE ZIP MAPPING")
            print(self.statezip_mapping)

            print("\nCOUNTRY MAPPING")
            print(self.country_mapping)

        except Exception as e:

            print("Preprocessing Error:", e)
            raise

    # Model training
    def train_models(self):

        try:

            features = (
                self.numeric_features +
                self.categorical_features
            )

            X = self.df[features]
            y = self.df["price"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            # Decision Tree Regression
            dt = DecisionTreeRegressor(
                max_depth=10,
                random_state=42
            )

            dt.fit(X_train, y_train)

            # Random Forest Regression
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)

            # Store models
            self.models = {
                "Decision Tree Regression": dt,
                "Random Forest Regression": rf
            }

            model_scores = {}

            print("\nMODEL PERFORMANCE")
            print("--------------------------------")

            for name, model in self.models.items():

                predictions = model.predict(X_test)

                r2 = r2_score(y_test, predictions)

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                model_scores[name] = r2

                print("\nModel:", name)
                print("R2 Score:", round(r2, 4))
                print("Mean Absolute Error:", round(mae, 2))

            # Select best model
            self.best_model_name = max(
                model_scores,
                key=model_scores.get
            )

            self.best_model = self.models[
                self.best_model_name
            ]

            print("\nBest Model:", self.best_model_name)

        except Exception as e:

            print("Model Training Error:", e)
            raise

    # Save best model
    def save_model(self):

        try:

            saved_data = {

                "model": self.best_model,

                "city_mapping": self.city_mapping,

                "statezip_mapping": self.statezip_mapping,

                "country_mapping": self.country_mapping,

                "numeric_features": self.numeric_features,

                "categorical_features": self.categorical_features,

                "best_model_name": self.best_model_name
            }

            file_path = self.base_path / "house_price_model.pkl"

            with open(file_path, "wb") as file:

                pickle.dump(saved_data, file)

            print("\nModel saved successfully!")
            print("File:", file_path)

        except Exception as e:

            print("Model Saving Error:", e)
            raise


# Main execution
if __name__ == "__main__":

    try:

        house_model = HousePriceModel()

        house_model.load_dataset()

        house_model.preprocess_data()

        house_model.train_models()

        house_model.save_model()

        print("\nProject executed successfully!")

    except Exception as e:

        error_type, error_message, error_traceback = sys.exc_info()

        line_number = (
            error_traceback.tb_lineno
            if error_traceback else "Unknown"
        )

        print("\nProject Execution Failed!")

        print("Error Type:", type(e).__name__)

        print("Error Message:", error_message)

        print("Error Line:", line_number)
