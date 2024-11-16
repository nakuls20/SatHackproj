import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI 

# Generate a time series for x
start = 0
stop = 288
x = np.arange(start, stop)  # Sequential time steps
y = 3 * x + 20 + np.random.normal(0, 10, len(x))  # Linear trend with noise

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Prepare data for LSTM
def create_sequences(data, sequence_length):
    x_seq, y_seq = [], []
    for i in range(len(data) - sequence_length):
        x_seq.append(data[i:i + sequence_length])
        y_seq.append(data[i + sequence_length])
    return np.array(x_seq), np.array(y_seq)

sequence_length = 10
X, Y = create_sequences(y_scaled, sequence_length)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=16, verbose=1)

# Predict on the test set
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)

# Visualization of test predictions
y_test_actual = scaler.inverse_transform(Y_test)
plt.plot(y_test_actual, color="blue", label="Actual values")
plt.plot(y_pred, color="red", label="Predicted values")
plt.legend()
plt.title("LSTM Model - Test Predictions")
plt.show()

# Prompt user for future predictions
num_days = int(input("Enter the number of future days to predict: "))
threshold = float(input("Enter the threshold value: "))

# Future Predictions
last_sequence = y_scaled[-sequence_length:]  # Start from the last known sequence
future_predictions = []

for i in range(num_days):
    # Reshape last sequence for prediction
    input_sequence = last_sequence.reshape((1, sequence_length, 1))
    next_value_scaled = model.predict(input_sequence, verbose=0)[0][0]
    next_value = scaler.inverse_transform([[next_value_scaled]])[0][0]
    future_predictions.append(next_value)

    if next_value > threshold:
        print(f"Threshold exceeded on Day {stop + i + 1}: Predicted value = {next_value:.2f}")
        break

    # Update the last sequence for the next prediction
    last_sequence = np.append(last_sequence[1:], next_value_scaled).reshape(-1, 1)

# Visualization of future predictions
future_x = np.arange(stop, stop + len(future_predictions))
plt.scatter(x, y, color="blue", label="Original data (with noise)")
plt.plot(future_x, future_predictions, color="green", linestyle="--", label="Future predictions")
plt.xlabel("Time steps (x)")
plt.ylabel("Values (y)")
plt.legend()
plt.title("LSTM Model with Future Predictions and Threshold")
plt.show()

# Output predictions
for i, value in enumerate(future_predictions, start=1):
    print(f"Day {stop + i}: Predicted value = {value:.2f}")


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    days = int(request.form.get("days"))
    threshold = float(request.form.get("threshold"))

    # Mock predictions
    predictions = [threshold * (i + 1) for i in range(days)]
    dates = [f"Day {i + 1}" for i in range(days)]

    # Render results page with the predictions
    return render_template("results.html", predictions=predictions, dates=dates)

if __name__ == "__main__":
    app.run(debug=True)