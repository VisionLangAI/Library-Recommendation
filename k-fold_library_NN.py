import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam

# Load the preprocessed data for the library dataset
df_library = pd.read_excel("resampled_normalized_data.xlsx")
print("\nStep 1: Library Data Loaded Successfully")

# Split the dataset into features and target
X_library = df_library.drop(["current_weather"], axis=1)  # Features
y_library = df_library["current_weather"]  # Target variable
print("Step 2: Library Dataset Split into Features and Target Successfully")

# Convert target labels to numerical format
label_encoder_library = LabelEncoder()
y_library = label_encoder_library.fit_transform(y_library.astype(str))  # Convert to string before label encoding
print("Step 3: Library Target Labels Converted to Numerical Format Successfully")

# Convert to NumPy arrays and change data type to float32
X_library = X_library.astype(np.float32)
y_library = y_library.astype(np.int32)  # Change to int32 for compatibility

# Define numerical columns for scaling (adjust as needed)
numerical_columns = X_library.columns

# Create a scaler
scaler = StandardScaler()

# Fit the scaler on the data and transform it
X_library_scaled = scaler.fit_transform(X_library[numerical_columns])

# Update the original DataFrame with scaled values
X_library[numerical_columns] = X_library_scaled

# One-hot encode target labels
y_library_encoded = tf.keras.utils.to_categorical(y_library, num_classes=4)

# Create a Sequential model
model_library = Sequential()

# Add a dense layer with ReLU activation function
model_library.add(Dense(256, input_dim=X_library.shape[1], activation='relu'))
# Add a dropout layer to prevent overfitting
#model_library.add(Dropout(0.1))

# Add another dense layer with ReLU activation function
model_library.add(Dense(128, activation='relu'))
# Add another dropout layer
#model_library.add(Dropout(0.1))

# Add two more hidden layers
model_library.add(Dense(64, activation='relu'))
#model_library.add(Dropout(0.1))

model_library.add(Dense(32, activation='relu'))
#model_library.add(Dropout(0.1))

model_library.add(Dense(16, activation='relu'))
#model_library.add(Dropout(0.1))

model_library.add(Dense(8, activation='relu'))
#model_library.add(Dropout(0.1))

# Output layer with softmax activation for multi-class classification
# Adjust the number of units to match the number of classes
model_library.add(Dense(4, activation='softmax'))

# Compile the model
model_library.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Step 6: Library Model Compilation Successful")

# Define k-fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize variables to store results
accuracy_values = []

# Train and evaluate the model using k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_library), 1):
    X_train, X_test = X_library.iloc[train_index], X_library.iloc[test_index]
    y_train, y_test = y_library_encoded[train_index], y_library_encoded[test_index]

    print(f"\nFold {fold}")

    # Train the model
    history = model_library.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    y_pred_prob = model_library.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    accuracy_values.append(accuracy)

    # Print accuracy for the current fold
    print(f"Accuracy (Fold {fold}): {accuracy}")

# Print average accuracy across folds
print(f"\nAverage Accuracy: {np.mean(accuracy_values)}")
