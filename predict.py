import tensorflow as tf
import numpy as np
import os
import random

def predict_digit():
    # Pfad zum gespeicherten Modell
    model_path = 'models/digit_recognizer.keras'
    
    # Prüfen ob Modell existiert
    if not os.path.exists(model_path):
        print(f"FEHLER: Modell nicht gefunden unter '{model_path}'")
        print("Bitte führe zuerst 'train_model.py' aus, um das Netzwerk zu trainieren.")
        return

    print("Lade neuronales Netzwerk...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        return

    print("Lade Testdaten (MNIST)...")
    # Wir laden nur die Testdaten, um eine Vorhersage zu simulieren
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Ein zufälliges Bild aus den 10.000 Testbildern auswählen
    random_index = random.randint(0, len(x_test) - 1)
    test_image = x_test[random_index]
    true_label = y_test[random_index]

    # ASCII-Art Darstellung des Bildes für die Konsole
    print(f"\n--- EINGABE BILD (Index: {random_index}) ---")
    for row in test_image:
        line = ""
        for pixel in row:
            # Helligkeitswerte in Zeichen umwandeln
            if pixel > 200: line += "██"
            elif pixel > 50: line += "░░"
            else: line += "  "
        print(line)
    print("---------------------------------------")

    # Bild für das Netzwerk vorbereiten (Normalisieren & Dimensionen anpassen)
    # Shape muss (1, 28, 28, 1) sein -> (BatchSize, Höhe, Breite, Kanäle)
    img_processed = test_image.astype('float32') / 255.0
    img_processed = np.expand_dims(img_processed, axis=-1)
    img_processed = np.expand_dims(img_processed, axis=0)

    # Vorhersage durchführen
    predictions = model.predict(img_processed, verbose=0)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    print(f"\n>>> ERGEBNIS")
    print(f"Netzwerk erkennt:  [{predicted_digit}]")
    print(f"Wahre Zahl:        [{true_label}]")
    print(f"Sicherheit:        {confidence:.2f}%")
    
    if predicted_digit == true_label:
        print("✅ KORREKT")
    else:
        print("❌ FALSCH")

if __name__ == "__main__":
    predict_digit()