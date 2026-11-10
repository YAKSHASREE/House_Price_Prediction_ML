# 🏡 House Price Prediction Using Machine Learning

A machine learning web application that predicts house prices based on property features and location details using **Decision Tree Regression** and **Random Forest Regression**.

The application is developed using Python, Scikit-learn, Flask and Pandas. It automatically compares the performance of both regression algorithms using the R² score and selects the best-performing model for prediction.

## 🚀 Project Overview

House prices depend on several factors, including the number of bedrooms, bathrooms, living area, lot size, property condition, year built and location.

This project uses historical housing data to train regression models and estimate house prices based on user-provided property information.

The trained model is integrated into a Flask web application with an interactive interface.

## ✨ Features

- House price prediction based on property details.
- Data preprocessing and categorical value encoding.
- Implementation of Decision Tree Regression.
- Implementation of Random Forest Regression.
- Model evaluation using R² Score and Mean Absolute Error (MAE).
- Automatic selection of the best-performing regression model.
- Model serialization using Pickle.
- Flask-based web application.
- User-friendly dark-themed interface.
- Profile details and image configured through environment variables.
- Deployment support using Render.

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning algorithms |
| Flask | Web application development |
| HTML | Webpage structure |
| CSS | Webpage styling |
| Pickle | Saving and loading trained models |
| Git & GitHub | Version control |
| Render | Application deployment |

## 🤖 Machine Learning Algorithms

This project implements two regression algorithms:

### 1. Decision Tree Regression

Decision Tree Regression predicts house prices by learning decision rules from the training dataset. It splits the data into smaller groups based on feature values to make predictions.

### 2. Random Forest Regression

Random Forest Regression combines multiple decision trees and uses their predictions to generate the final result. It generally provides more stable predictions than an individual decision tree.

### Model Selection

Both models are trained and evaluated using the same training and testing datasets.

Evaluation metrics:

- **R² Score:** Measures how well the model explains the variation in house prices.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted prices.

The model with the higher R² score is automatically selected and saved as `house_price_model.pkl`.

## 📊 Dataset Information

The project uses `House_price_prediction_data.csv`.

**Dataset dimensions:** 4,600 rows × 18 columns.

Important dataset features include:

| Feature | Description |
|---|---|
| price | Target variable |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| sqft_living | Living area in square feet |
| sqft_lot | Lot area in square feet |
| floors | Number of floors |
| waterfront | Waterfront property indicator |
| view | Property view rating |
| condition | Property condition |
| sqft_above | Above-ground area |
| sqft_basement | Basement area |
| yr_built | Year of construction |
| yr_renovated | Year of renovation |
| city | Property city |
| statezip | State and ZIP code |
| country | Country |

The `date` and `street` columns are removed during preprocessing. The `price` column is used as the prediction target.

Location columns are converted into numerical codes using mappings generated during preprocessing.

## 📁 Project Structure

```text
ML_MiniProject/
│
├── templates/
│   └── index.html
│
├── main.py
├── app.py
├── House_price_prediction_data.csv
├── house_price_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `.env` file and `.venv` folder are maintained locally and are not uploaded to GitHub.

## ⚙️ Installation and Setup

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the project directory

```bash
cd ML_MiniProject
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Train the machine learning models

```bash
python main.py
```

This executes the following operations:

1. Loads the housing dataset.
2. Preprocesses the data.
3. Encodes categorical location values.
4. Splits the dataset into training and testing sets.
5. Trains Decision Tree and Random Forest regression models.
6. Evaluates both models.
7. Selects the best-performing model.
8. Saves the selected model as `house_price_model.pkl`.

### Step 6: Run the Flask application

```bash
python app.py
```

### Step 7: Open the application

Visit the following URL in your browser:

```text
http://127.0.0.1:5000
```

## 🔐 Environment Variables

The application uses environment variables for profile configuration.

Create a `.env` file locally:

```env
FLASK_SECRET_KEY=YOUR_SECRET_KEY
PROFILE_NAME=Your Name
PROFILE_EMAIL=your_email
PROFILE_PHONE=your_phone
PROFILE_IMAGE=YOUR_DIRECT_IMAGE_URL
```

**Important:** Never upload `.env` to GitHub. Add the same variables separately in Render's Environment settings when deploying.

The profile image is loaded through its external URL. No local static image folder is required.

## ☁️ Deployment on Render

The Flask application can be deployed using Render.

**Build Command:**

```bash
pip install -r requirements.txt
```

**Start Command:**

```bash
gunicorn app:app
```

Add the required environment variables in the Render dashboard before deployment.

The trained `house_price_model.pkl` must be included in the deployed repository.

## 📈 Model Evaluation

The application prints the following performance metrics for both algorithms:

- R² Score
- Mean Absolute Error (MAE)

The best model is selected automatically based on the higher R² score.

Actual performance values depend on the training and testing data. Run `python main.py` to obtain the results for your dataset.

## 🔮 Future Enhancements

- Improve categorical feature encoding.
- Add more regression algorithms for comparison.
- Improve model accuracy through hyperparameter tuning.
- Add visualizations for model performance.
- Provide more detailed prediction reports.
- Improve the user interface and input validation.

## 👩‍💻 Author

**YAKSHASREE DANABOINA**

B.Tech – Artificial Intelligence and Machine Learning

Project: House Price Prediction Using Machine Learning

---

⭐ If you find this project useful, consider giving the repository a star!
