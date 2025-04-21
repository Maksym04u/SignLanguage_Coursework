# %%

# Import necessary libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from keras._tf_keras.keras import Sequential, Model
from keras._tf_keras.keras.layers import GRU, Dense, Dropout, BatchNormalization, Input, Attention, Concatenate
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.optimizers import Adam

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of actions (signs) labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Define the number of sequences and frames
sequences = 30  # Changed to match your data collection
frames = 20

# Create a label map to map each action label to a numeric value
label_map = {label:num for num, label in enumerate(actions)}

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over actions and sequences to load landmarks and corresponding labels
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

# Convert landmarks and labels to numpy arrays
X, Y = np.array(landmarks), to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=34, stratify=Y)

# Define the enhanced model architecture with GRU layers and attention
def create_model(input_shape, num_classes):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First GRU layer
    x = GRU(64, return_sequences=True, activation='tanh',
            kernel_regularizer='l2')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second GRU layer
    x = GRU(128, return_sequences=True, activation='tanh',
            kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Attention mechanism
    attention = Attention()([x, x])
    x = Concatenate()([x, attention])
    
    # Third GRU layer
    x = GRU(64, return_sequences=False, activation='tanh',
            kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Dense layers
    x = Dense(64, activation='tanh', kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)

# Create the model
model = create_model(input_shape=(frames, 126), num_classes=len(actions))

# Compile the model with Adam optimizer and categorical cross-entropy loss
optimizer = Adam(learning_rate=0.0005)  # Reduced learning rate
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])

# Define callbacks with more patience
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,  # Increased patience
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # More gradual learning rate reduction
    patience=7,
    min_lr=0.00001
)

# Train the model with callbacks
history = model.fit(
    X_train, Y_train,
    epochs=150,  # Increased epochs
    batch_size=16,  # Reduced batch size
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save the trained model
model.save('my_model.h5')

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)
# Get the true labels from the test set
test_labels = np.argmax(Y_test, axis=1)

# Calculate and print the accuracy of the predictions
accuracy = metrics.accuracy_score(test_labels, predictions)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(metrics.classification_report(test_labels, predictions, target_names=actions))