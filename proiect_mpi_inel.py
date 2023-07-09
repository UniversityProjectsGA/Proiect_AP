from mpi4py import MPI
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from sklearn.metrics import confusion_matrix

# Impartim datele de testare la procesele MPI
def imparte_date(date, numar_procese, rank):
    numar_date = len(date)
    marime_portie_date = numar_date // numar_procese
    inceput = rank * marime_portie_date
    sfarsit = (rank + 1) * marime_portie_date if rank < numar_procese - 1 else numar_date
    return date[inceput:sfarsit]

# Procesam datele prin reducerea pixelilor de la 0-255 la 0-1
def procesare_date(date):
    return date.astype('float32') / 255.0

# Modele cu 'Convolution Layer', 'Pooling Layer', 'Flatten' si 'Dropout'
def fun_model_v1():
    model_v1 = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model_v1

def fun_model_v2():
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
    return model_v2

def fun_model_v3():
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
    return model_v3
    
# Invatam modelele
def invatare_model_v1(model_v1, imagini_invatare, etichete_invatare):
    model_v1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_v1.fit(imagini_invatare, etichete_invatare, epochs=5, batch_size=64)

def invatare_model_v2(model_v2, imagini_invatare, etichete_invatare):
    model_v2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_v2.fit(imagini_invatare, etichete_invatare, epochs=5, batch_size=64)

def invatare_model_v3(model_v3, imagini_invatare, etichete_invatare):
    model_v3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_v3.fit(imagini_invatare, etichete_invatare, epochs=5, batch_size=64)
    
# Reducem gradientii
def reducere_gradienti(model):
    gradienti = [gradient.numpy() for gradient in model.trainable_weights]
    gradienti_redusi = [np.zeros_like(gradient) for gradient in gradienti]
    for i in range(len(gradienti)):
        comm.Allreduce(gradienti[i], gradienti_redusi[i], op=MPI.SUM)
    gradienti = [tf.convert_to_tensor(gradient) for gradient in gradienti_redusi]
    model.optimizer.apply_gradients(zip(gradienti, model.trainable_weights))

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    numar_procese = comm.Get_size()
    rank = comm.Get_rank()

    # Stocam preciziile intr-o singura variabila pentru afisare
    rezultate_precizii = []

    # Determine neighboring ranks in the ring
    rank_stanga = (rank - 1) % numar_procese
    rank_dreapta = (rank + 1) % numar_procese

    # Bariera pentru sincronizare in printare
    print("Procesul %d are in stanga procesul: %d si in dreapta procesul: %d" % (rank, rank_stanga, rank_dreapta))
    comm.Barrier()

    if rank == 0:
        print("Proiect Algoritmi Paraleli")
        print("Neural Nets")
        print("Topologie Inel:")
        print("1. Precizie folosind modelul 1")
        print("2. Precizie folosind modelul 2")
        print("3. Precizie folosind modelul 3")
        print("0. Inchide programul")
        alegere = input("Raspuns: ")
        alegere = comm.bcast(alegere, root=0)
    else:
        alegere = comm.bcast(None, root=0)

    if alegere == "1":   
        # Folosim keras.datasets.cifar10.load_data pentru datele de invatare si testare:
        # imagini_invatare: 50000 de imagini de 32x32 pixeli cu valori de la 0-255
        # imagini_invatare: 10000 de etichete de la 0-9
        # imagini_testare: 10000 de imagini de 32x32 pixeli cu valori de la 0-255
        # etichete_testare: 10000 de etichete de la 0-9
        (imagini_invatare, etichete_invatare), (imagini_testare, etichete_testare) = keras.datasets.cifar10.load_data()

        # Impartim datele de invatare
        imagini_invatare_locale = imparte_date(imagini_invatare, numar_procese, rank)
        etichete_invatare_locale = imparte_date(etichete_invatare, numar_procese, rank)

        # Procesam datele de invatare
        imagini_invatare_locale = procesare_date(imagini_invatare_locale)

        # Modelul 1
        model_v1 = fun_model_v1()

        # Invatam modelul 1
        timp_invatare_start_v1 = time.time()

        invatare_model_v1(model_v1, imagini_invatare_locale, etichete_invatare_locale)

        timp_invatare_stop_v1 = time.time()

        timp_invatare_v1 = timp_invatare_stop_v1 - timp_invatare_start_v1

        # Reducem gradientii pentru modelul 1
        reducere_gradienti(model_v1)

        # Procesam datele de testare
        imagini_testare_local = procesare_date(imagini_testare)

        # Testam modelul 1
        timp_testare_start_v1 = time.time()

        test_precizie_mv1 = model_v1.evaluate(imagini_testare_local, etichete_testare)

        timp_testare_stop_v1 = time.time()

        timp_testare_v1 = timp_testare_stop_v1 - timp_testare_start_v1

        rezultate_precizii.append((rank, test_precizie_mv1[1], timp_invatare_v1, timp_testare_v1))
    if alegere == "2":   
        (imagini_invatare, etichete_invatare), (imagini_testare, etichete_testare) = keras.datasets.cifar10.load_data()

        # Impartim datele de invatare
        imagini_invatare_locale = imparte_date(imagini_invatare, numar_procese, rank)
        etichete_invatare_locale = imparte_date(etichete_invatare, numar_procese, rank)

        # Procesam datele de invatare
        imagini_invatare_locale = procesare_date(imagini_invatare_locale)

        # Modelul 2
        model_v2 = fun_model_v2()

        # Invatam modelul 2
        timp_invatare_start_v2 = time.time()

        invatare_model_v2(model_v2, imagini_invatare_locale, etichete_invatare_locale)

        timp_invatare_stop_v2 = time.time()

        timp_invatare_v2 = timp_invatare_stop_v2 - timp_invatare_start_v2

        # Reducem gradientii pentru modelul 2
        reducere_gradienti(model_v2)

        # Procesam datele de testare
        imagini_testare_local = procesare_date(imagini_testare)

        # Testam modelul 2
        timp_testare_start_v2 = time.time()

        test_precizie_mv2 = model_v2.evaluate(imagini_testare_local, etichete_testare)

        timp_testare_stop_v2 = time.time()

        timp_testare_v2 = timp_testare_stop_v2 - timp_testare_start_v2

        rezultate_precizii.append((rank, test_precizie_mv2[1], timp_invatare_v2, timp_testare_v2))
    if alegere == "3":   
        (imagini_invatare, etichete_invatare), (imagini_testare, etichete_testare) = keras.datasets.cifar10.load_data()

        # Impartim datele de invatare
        imagini_invatare_locale = imparte_date(imagini_invatare, numar_procese, rank)
        etichete_invatare_locale = imparte_date(etichete_invatare, numar_procese, rank)

        # Procesam datele de invatare
        imagini_invatare_locale = procesare_date(imagini_invatare_locale)

        # Modelul 3
        model_v3 = fun_model_v3()

        # Invatam modelul 3
        timp_invatare_start_v3 = time.time()

        invatare_model_v3(model_v3, imagini_invatare_locale, etichete_invatare_locale)

        timp_invatare_stop_v3 = time.time()

        timp_invatare_v3 = timp_invatare_stop_v3 - timp_invatare_start_v3

        # Reducem gradientii pentru modelul 3
        reducere_gradienti(model_v3)

        # Procesam datele de testare
        imagini_testare_local = procesare_date(imagini_testare)

        # Testam modelul 3
        timp_testare_start_v3 = time.time()

        test_precizie_mv3 = model_v3.evaluate(imagini_testare_local, etichete_testare)

        timp_testare_stop_v3 = time.time()

        timp_testare_v3 = timp_testare_stop_v3 - timp_testare_start_v3

        rezultate_precizii.append((rank, test_precizie_mv3[1], timp_invatare_v3, timp_testare_v3))
    if alegere == "0":
        print("Procesul se inchide.")

    # Folosim Gather pentru a aduna toate rezultatele
    rezultate_precizii_adunat = comm.gather(rezultate_precizii, root=0)

    if rank == 0:
        # Afisam rezultatele
        if alegere != "0":
            print("Model", alegere)
            for rezultate in rezultate_precizii_adunat:
                for rank, precizie, timp_invatare, timp_testare in rezultate:
                    print("Procesul %d - Precizie: %.4f - Durata invatare: %.4f s - Durata testare: %.4f s" % (rank, precizie, timp_invatare, timp_testare))
        elif alegere == "0":
            print("Program inchis.")