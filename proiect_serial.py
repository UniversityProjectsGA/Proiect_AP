import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from sklearn.metrics import confusion_matrix

# Folosim keras.datasets.cifar10.load_data pentru datele de invatare si testare:
# imagini_invatare: 50000 de imagini de 32x32 pixeli cu valori de la 0-255
# imagini_invatare: 10000 de etichete de la 0-9
# imagini_testare: 10000 de imagini de 32x32 pixeli cu valori de la 0-255
# etichete_testare: 10000 de etichete de la 0-9
(imagini_invatare, etichete_invatare), (imagini_testare, etichete_testare) = keras.datasets.cifar10.load_data()

# Reducem valoarea pixelilor de la 0-255 la 0-1
imagini_invatare = imagini_invatare.astype('float32') / 255.0
imagini_testare = imagini_testare.astype('float32') / 255.0

# Modele cu 'Convolution Layer', 'Pooling Layer', 'Flatten' si 'Dropout' 
model_v1 = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_v2 = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model_v3 = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.35),

    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.5),

    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

# Compilam modelele
model_v1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_v2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_v3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def model1_precizie():
    timp_invatare_start_model1 = time.time()
    # Invatam modelul pe datele de test cu 10 epoci
    model_v1.fit(imagini_invatare, etichete_invatare, epochs=10, batch_size=64)

    timp_invatare_stop_model1 = time.time()

    timp_testare_start_model1 = time.time()
    # Testam modelul cu datele de test
    test_precizie_mv1 = model_v1.evaluate(imagini_testare, etichete_testare)

    timp_testare_stop_model1 = time.time()

    predictie1 = model_v1.predict(imagini_testare)
    etichete_prezise1 = np.argmax(predictie1, axis=1)
    mat_conf1 = confusion_matrix(etichete_testare, etichete_prezise1)

    print("Precizie model 1:", test_precizie_mv1[1])
    print("Timp invatare model 1:", timp_invatare_stop_model1 - timp_invatare_start_model1)
    print("Timp testare model 1:", timp_testare_stop_model1 - timp_testare_start_model1)
    print("Matrice de confuzie model 1:")
    print(mat_conf1)

def model2_precizie():
    timp_invatare_start_model2 = time.time()
    # Invatam modelul pe datele de test cu 10 epoci
    model_v2.fit(imagini_invatare, etichete_invatare, epochs=10, batch_size=64)

    timp_invatare_stop_model2 = time.time()

    timp_testare_start_model2 = time.time()
    # Testam modelul cu datele de test
    test_precizie_mv2 = model_v2.evaluate(imagini_testare, etichete_testare)

    timp_testare_stop_model2 = time.time()

    predictie2 = model_v2.predict(imagini_testare)
    etichete_prezise2 = np.argmax(predictie2, axis=1)
    mat_conf2 = confusion_matrix(etichete_testare, etichete_prezise2)

    print("Precizie model 2:", test_precizie_mv2[1])
    print("Timp invatare model 2:", timp_invatare_stop_model2 - timp_invatare_start_model2)
    print("Timp testare model 2:", timp_testare_stop_model2 - timp_testare_start_model2)
    print("Matrice de confuzie model 2:")
    print(mat_conf2)

def model3_precizie():
    timp_invatare_start_model3 = time.time()
    # Invatam modelul pe datele de test cu 10 epoci
    model_v3.fit(imagini_invatare, etichete_invatare, epochs=10, batch_size=64)

    timp_invatare_stop_model3 = time.time()

    timp_testare_start_model3 = time.time()
    # Testam modelul cu datele de test
    test_precizie_mv3 = model_v3.evaluate(imagini_testare, etichete_testare)

    timp_testare_stop_model3 = time.time()

    predictie3 = model_v3.predict(imagini_testare)
    etichete_prezise3 = np.argmax(predictie3, axis=1)
    mat_conf3 = confusion_matrix(etichete_testare, etichete_prezise3)

    print("Precizie model 3:", test_precizie_mv3[1])
    print("Timp invatare model 3:", timp_invatare_stop_model3 - timp_invatare_start_model3)
    print("Timp testare model 3:", timp_testare_stop_model3 - timp_testare_start_model3)
    print("Matrice de confuzie model 3:")
    print(mat_conf3)

def main():
    print("Proiect Algoritmi Paraleli")
    print("Neural Nets")
    while True:
        print("1. Precizie folosind modelul 1")
        print("2. Precizie folosind modelul 2")
        print("3. Precizie folosind modelul 3")
        print("0. Inchide programul")
        alegere = input("Raspuns: ")

        if alegere == "1":
            model1_precizie()
        elif alegere == "2":
            model2_precizie()
        elif alegere == "3":
            model3_precizie()
        elif alegere == "0":
            print("Programul se inchide.")
            break
        else:
            print("Raspuns incorect")

if __name__ == "__main__":
    main()