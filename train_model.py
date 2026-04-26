import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Carregamento e pré-processamento do dataset

print("\nCarregando dataset MNIST...")

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
x_test  = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

model = models.Sequential([                             # Modo Sequencial, "stackando" camadas
    
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # Convolução de 32 filtros, Kernel 3 x 3
    layers.MaxPooling2D((2, 2)),                                           # Pooling - reduzir custo computacional
    
    layers.Conv2D(64, (3, 3), activation='relu'),       # Segunda Convolução, 64 filtros, para padrões mais complexos
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),                                   # Matriz em Vetor
    
    layers.Dense(64, activation='relu'),                # 64 neurôneos, combinações de padrões
    layers.Dense(10, activation='softmax')              # 10 neurôneos, softmax → probabilidade
])

model.compile(optimizer='adam',                         # Algoritmo de otimização
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nComeçando treinamento...")
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1) # Parâmetros para o processo de treinamento

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nAcurácia final no conjunto de teste: {test_acc:.4f}")

model.save('model.h5')
print("\nModelo salvo com sucesso como: 'model.h5'.")