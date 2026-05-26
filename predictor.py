import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def load_and_preprocess_data(filepath):
    """
    Loads the Car dataset and preprocesses categorical features.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {filepath}.")
        print("Please double-check the path and filename.")
        return None, None, None, None
    
    # 1. Drop the 'Car_Name' column as it has high cardinality and isn't useful for simple regression
    df = df.drop('Car_Name', axis=1)
    
    # 2. Convert 'Year' into 'Age_of_Car' (Assuming current year is 2026)
    df['Age_of_Car'] = 2026 - df['Year']
    df = df.drop('Year', axis=1)
    
    # 3. One-Hot Encode categorical variables (Fuel_Type, Selling_type, Transmission)
    # drop_first=True helps avoid the dummy variable trap (multicollinearity)
    df = pd.get_dummies(df, drop_first=True)
    
    # 4. Separate features (X) and target (Y)
    X = df.drop('Selling_Price', axis=1)
    Y = df['Selling_Price']
    
    # 5. Split into training and testing sets (90% train, 10% test is common for this specific dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)
    
    return X_train, X_test, Y_train, Y_test

def plot_and_save_predictions(y_true, y_pred, title, filename):
    """Generates and saves a scatter plot of Actual vs Predicted prices."""
    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(title)
    
    # Save the plot instead of just showing it so you can use it in your README
    plt.savefig(f"assets/{filename}")
    plt.close()
    print(f"Plot saved to assets/{filename}")

def main():
    # Updated to your exact absolute file path
    filepath = "/Users/krishna/Desktop/Car Prize Prediction/data/Cars_data.csv" 
    
    print(f"Loading and preprocessing data from {filepath}...")
    X_train, X_test, Y_train, Y_test = load_and_preprocess_data(filepath)
    
    if X_train is None:
        return
    
    # ---------------------------------------------------------
    # Model Training: Linear Regression
    # ---------------------------------------------------------
    print("\n--- Training Linear Regression Model ---")
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, Y_train)
    
    # --- Training Data Evaluation ---
    training_data_prediction = lin_reg_model.predict(X_train)
    
    error_score_train = metrics.r2_score(Y_train, training_data_prediction)
    mae_train = metrics.mean_absolute_error(Y_train, training_data_prediction)
    
    print(f"\n[Training Data Metrics]")
    print(f"R Squared Error : {error_score_train:.4f}")
    print(f"Mean Absolute Error : {mae_train:.4f}")
    
    plot_and_save_predictions(Y_train, training_data_prediction, 
                              "Actual Prices vs Predicted Prices (Training Data)", 
                              "train_predictions.png")

    # --- Testing Data Evaluation ---
    test_data_prediction = lin_reg_model.predict(X_test)
    
    error_score_test = metrics.r2_score(Y_test, test_data_prediction)
    mae_test = metrics.mean_absolute_error(Y_test, test_data_prediction)
    
    print(f"\n[Test Data Metrics]")
    print(f"R Squared Error : {error_score_test:.4f}")
    print(f"Mean Absolute Error : {mae_test:.4f}")
    
    plot_and_save_predictions(Y_test, test_data_prediction, 
                              "Actual Prices vs Predicted Prices (Test Data)", 
                              "test_predictions.png")

if __name__ == "__main__":
    main()