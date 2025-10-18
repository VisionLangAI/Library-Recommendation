import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.model_selection import train_test_split

# Assuming X_library is your feature matrix and y_library is your target variable

X_train_library, X_test_library, y_train_library, y_test_library = train_test_split(
    X_library, y_library, test_size=0.2, random_state=42, stratify=y_library
)
print("Step 4: Library Dataset Split into Training and Testing Sets Successfully")

# Convert to NumPy arrays and change data type to float32
X_train_library = X_train_library.astype(np.float32)
y_train_library = y_train_library.astype(np.int32)  # Change to int32 for compatibility

# Define numerical columns for scaling (adjust as needed)
numerical_columns = X_train_library.columns

# Create a scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_library_scaled = scaler.fit_transform(X_train_library[numerical_columns])

# Update the original DataFrame with scaled values
X_train_library[numerical_columns] = X_train_library_scaled

# One-hot encode target labels
y_train_library_encoded = tf.keras.utils.to_categorical(y_train_library, num_classes=4)



# Create a Sequential model
model_library = Sequential()

# Add a dense layer with ReLU activation function
model_library.add(Dense(256, input_dim=X_library.shape[1], activation='relu'))
# Add a dropout layer to prevent overfitting
#model_library.add(Dropout(0.1))

# Add another dense layer with ReLU activation function
model_library.add(Dense(256, activation='relu'))
# Add another dropout layer
model_library.add(Dropout(0.4))


# Add two more hidden layers
model_library.add(Dense(64, activation='relu'))
model_library.add(Dropout(0.4))

model_library.add(Dense(64, activation='relu'))
model_library.add(Dropout(0.4))

model_library.add(Dense(64, activation='relu'))
model_library.add(Dropout(0.4))

model_library.add(Dense(64, activation='relu'))
model_library.add(Dropout(0.4))

model_library.add(Dense(64, activation='relu'))
model_library.add(Dropout(0.4))





# Output layer with softmax activation for multi-class classification
# Adjust the number of units to match the number of classes
model_library.add(Dense(4, activation='sigmoid'))

# Compile the model
model_library.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])

print("Step 6: Library Model Compilation Successful")

# Define early stopping callback
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model for the library dataset with early stopping
history = model_library.fit(X_train_library, y_train_library_encoded, epochs=100, batch_size=32, validation_split=0.2)
print("Step 7: Library Model Training Successful")

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model on the library test set
y_pred_library_prob = model_library.predict(X_test_library)
y_pred_library = np.argmax(y_pred_library_prob, axis=1)
print("Step 8: Library Model Evaluation Successful")

# Convert predicted labels back to the original format
y_pred_original_library = label_encoder_library.inverse_transform(y_pred_library)

# Print Library accuracy
accuracy_library = accuracy_score(label_encoder_library.inverse_transform(y_test_library), y_pred_original_library)
print(f"Library Accuracy: {accuracy_library}")
