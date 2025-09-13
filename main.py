import numpy as np
from klase import load_data
from nnfs.datasets import spiral_data  # samo ako zatreba za test
from klase import Layer_Dense
from klase import Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from klase import Optimizer_SGD

import pickle

def sacuvaj_model(model, ime_fajla):
    with open(ime_fajla, 'wb') as f:
        pickle.dump(model, f)

def ucitaj_model(ime_fajla):
    with open(ime_fajla, 'rb') as f:
        return pickle.load(f)

# Učitaj podatke iz MNIST dataset-a
(training_data, validation_data, test_data) = load_data()

X, y = training_data
X_val, y_val = validation_data
X_test, y_test = test_data

# Inicijalizuj slojeve i aktivacije
dense1 = Layer_Dense(784, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Optimizator
optimizer = Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9)

# Trening petlja
for epoch in range(101):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Preciznost
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y)

    # Štampaj info
    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Sačuvaj trenirane slojeve i optimizer
sacuvaj_model(
    {
        'dense1': dense1,
        'dense2': dense2,
        'optimizer': optimizer,
        'activation1': activation1,
        'loss_activation': loss_activation
    },
    'model.pkl'
)

print("Model je uspešno sačuvan u 'model.pkl'")

model = ucitaj_model('model.pkl')
dense1 = model['dense1']
dense2 = model['dense2']
optimizer = model['optimizer']

# Provera da li je sve uspešno učitano
if not all([dense1, dense2, optimizer]):
    raise ValueError("Greška prilikom učitavanja modela iz fajla.")


# Funkcija za testiranje modela
def test_model(dense1, activation1, dense2, loss_activation, X_test, y_test):
    # Forward pass
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    # Izračunaj gubitak
    loss = loss_activation.forward(dense2.output, y_test)

    # Predikcija
    predictions = np.argmax(loss_activation.output, axis=1)
    accuracy = np.mean(predictions == y_test)

    return accuracy, loss


# Inicijalizuj slojeve i aktivacije za testiranje (isto kao pre)
activation1 = model['activation1']
loss_activation = model['loss_activation']

# Testiranje modela
accuracy, loss = test_model(dense1, activation1, dense2, loss_activation, X_test, y_test)

# Ispis rezultata
print(f'Test Accuracy: {accuracy:.3f}, Test Loss: {loss:.3f}')

