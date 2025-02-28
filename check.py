new_df = df

X = new_df.iloc[:, :-1].values  
y = new_df['label'].values      


# x= independent , y output
new_df_scaler = StandardScaler()     
X = new_df_scaler.fit_transform(X)


X = X.reshape(X.shape[0], 33, 4) 


num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)



New_X_train, New_X_test, New_y_train, New_y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


with open("../Models/scaler.pkl", "wb") as f:
    pickle.dump(new_df_scaler, f)

def create_CNN_LSTM_model(input_shape, num_classes):
    model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),

            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),

            Bidirectional(LSTM(128, return_sequences=False)),

            Dense(64, activation='relu'),
            Dropout(0.4),  
            Dense(num_classes, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_CNN_GRU_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Bidirectional(GRU(128, return_sequences=False)),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_RNN_model(input_shape, num_classes):
    model = Sequential([
        SimpleRNN(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense, Dropout, LSTM, Input, Bidirectional, Reshape

def create_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = Bidirectional(LSTM(128, return_sequences=False))(x)
    
    x = Dense(128, activation='relu')(encoded)
    x = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
    decoded = Reshape((input_shape[0], input_shape[1]))(x)  # Fix applied here
    
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

def create_classification_model(encoder, num_classes):
    inputs = encoder.input
    x = encoder.output
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

autoencoder, encoder = create_autoencoder((33, 4))
classification_model = create_classification_model(encoder, num_classes)
models = {
    'CNN_GRU': create_CNN_GRU_model((33, 4), num_classes),
    'RNN': create_RNN_model((33, 4), num_classes),
    'CNN_LSTM': create_CNN_LSTM_model((33, 4), num_classes),
    'Autoencoder_Classifier': classification_model
}

import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

results = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")

    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(f'../Models/{model_name}.h5', save_best_only=True)
    ]

    history = model.fit(New_X_train, New_y_train, epochs=15, batch_size=64, 
                        validation_data=(New_X_test, New_y_test), verbose=1, callbacks=callbacks)

    test_loss, test_acc = model.evaluate(New_X_test, New_y_test, verbose=1)
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")

    # Save training history
    with open(f'../Models/{model_name}_history.json', 'w') as f:
        json.dump(history.history, f)

    results[model_name] = {
        'model': model,
        'history': history.history,  # Convert history to a dictionary
        'test_accuracy': test_acc
    }

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing keypoints
pose = mp_pose.Pose()




# List to store extracted keypoints for saving
keypoints_data = []

def visualize_pose(video_path, max_frames=200, save_csv=True):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Video file not found or could not be opened.")
        return

    print("✅ Video opened successfully!")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        # Convert to RGB and process with MediaPipe Pose
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            print(f"🟢 Frame {frame_count}: Keypoints detected!")

            # Extract and store keypoints
            row = [frame_count]  # Start with frame number
            h, w, _ = frame.shape  # Get frame size

            for i, landmark in enumerate(results.pose_landmarks.landmark):
                cx, cy = int(landmark.x * w), int(landmark.y * h)  # Convert to pixel coordinates
                print(f"  Keypoint {i}: x={landmark.x:.4f}, y={landmark.y:.4f}, z={landmark.z:.4f}, visibility={landmark.visibility:.4f}")
                
                # Store for CSV
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Overlay keypoint indices on the frame
                cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save keypoints data
            if save_csv:
                keypoints_data.append(row)

        else:
            print(f"🔴 Frame {frame_count}: No keypoints detected.")

        cv2.imshow("Pose Visualization", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video processing completed!")

    # Save keypoints to CSV
    if save_csv and keypoints_data:
        columns =["Frame"] + [f"K{i}_{c}" for i in range(33) for c in ("x", "y", "z", "visibility")]
        LR_test_keypoints = pd.DataFrame(keypoints_data, columns=columns)
        LR_test_keypoints.to_csv("../Datasets/CSV_DATASET/pose_keypoints.csv", index=False)
        print("📂 Keypoints saved to pose_keypoints.csv")

visualize_pose("../Datasets/Test_videos/Normal Gait (1) copy.mp4")
new_keypoints = keypoints.copy()
new_keypoints.drop("Frame", axis=1, inplace=True)

# Scale keypoints
new_keypoints = new_df_scaler.transform(new_keypoints)

# Reshape for CNN input
new_keypoints = new_keypoints.reshape(new_keypoints.shape[0], 33, 4)  # Ensure correct shape
print(f"✅ Extracted keypoints shape: {new_keypoints.shape}")
print(f"🔹 Sample keypoints (first frame):\n{new_keypoints[0]}")
models = {
    'CNN_GRU': '../Models/CNN_GRU.h5',
    'RNN': '../Models/RNN.h5',
    'CNN_LSTM': '../Models/CNN_LSTM.h5',
    'Autoencoder_Classifier': '../Models/Autoencoder_Classifier.h5'
}
# Class labels
class_labels = {
    0: "Normal",
    1: "Limping",
    2: "Slouch",
    3: "No Arm Swing",
    4: "Concriduction"
}
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model

# Iterate through models and show individual outputs
for model_name, model_path in models.items():
    print("\n" + "=" * 50)
    print(f"🔍 Evaluating Model: {model_name}")
    print("=" * 50)

    # Load model
    model = load_model(model_path)  # Use Keras load_model for .h5 files

    # Predict using the model
    predictions = model.predict(new_keypoints)
    predicted_classes = np.argmax(predictions, axis=1)

    # Store frame-wise predictions
    frame_predictions = [class_labels.get(pred, "Unknown") for pred in predicted_classes]

    # Count occurrences of each class
    counter = Counter(predicted_classes)

    # Find the most common class safely
    most_common_class = counter.most_common(1)
    overall_prediction = class_labels.get(most_common_class[0][0], "Unknown") if most_common_class else "No valid predictions"

    # Display class distribution
    print("\n📊 Class Distribution:")
    for key, value in counter.items():
        print(f"{class_labels.get(key, 'Unknown')}: {value} frames")

    # Display overall gait condition
    print(f"\n🚶 Overall Gait Condition (Most Common Prediction): {overall_prediction}")

print("\n✅ Multi-Model Gait Classification Completed.")