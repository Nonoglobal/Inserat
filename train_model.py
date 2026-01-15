import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os

def create_complex_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Erstellt ein tiefes Convolutional Neural Network (CNN).
    Dieses Modell ist komplexer als ein einfaches MLP und nutzt
    moderne Techniken wie BatchNormalization und Dropout für bessere Erkennung.
    """
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=input_shape),
        
        # Block 1: Feature Extraction (Low Level Features)
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2: Feature Extraction (Mid Level Features)
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 3: Feature Extraction (High Level Features)
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Classification Head (Entscheidungsebene)
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    print("Initialisiere Training für Ziffernerkennung...")
    
    # 1. Daten laden (MNIST Datensatz)
    # Enthält 60.000 Trainingsbilder und 10.000 Testbilder
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 2. Preprocessing
    # Normalisierung der Pixelwerte auf 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Dimensionen anpassen für CNN (Batch, Höhe, Breite, Farbkanäle)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    # Labels zu One-Hot Vektoren konvertieren (z.B. 5 -> [0,0,0,0,0,1,0,0,0,0])
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # 3. Modell erstellen
    model = create_complex_model()
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    # 4. Training
    # Early Stopping verhindert Overfitting, wenn das Modell nicht mehr lernt
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Starte Training...")
    history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping])
    
    # 5. Modell speichern
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model.save('models/digit_recognizer.keras')
    print("Modell erfolgreich gespeichert unter models/digit_recognizer.keras")

if __name__ == "__main__":
    main()

