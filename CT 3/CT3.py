'''
CSC 510 - Critical Thinking 3 Option 2 (Tensorflow ANN Model)
Karl Estes
Due: october 3rd, 2021

Assignment
----------
Using Tensorflow and your own research, write a basic Tensorflow ANN model to perform a basic function of your choosing. 
Your submission should be inference-ready upon execution, and include all model checkpoints necessary for inference. 
Your submission should include a self-executable Python script, which model inference can be confirmed. 
The executable script should visually display results. Accuracy will not be graded but must run without error and display classification results on-screen.
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
import os
from os import system, name
from sklearn.model_selection import train_test_split


def equation(x):
    return (abs(math.sin(x**x) / 2 ** (((x ** x) - math.pi/2) / math.pi)))

def gen_rand_x(n):
    x = []
    for i in range(n):
        x.append(random.random() * 3)
    return x

def calc_y(x):
    y = []
    for i in x:
        y.append(equation(i))
    return y

def plot_results(x, y, preds):
    fig,(ax1, ax2, ax3) = plt.subplots(3, 1)
    
    ax1.plot(x,y, 'bo',label="Actual")
    ax1.title.set_text("Actual Values")
    ax2.plot(x,preds, 'r+',label="Predicted")
    ax2.title.set_text("Predicted Values")
    ax3.plot(x,y,'bo',label="Actual")
    ax3.plot(x,preds,'r+',label="Predicted")
    ax3.title.set_text("Actual-Predicted Overlay")
    ax3.legend()
    fig.tight_layout(pad=1.0) # padding between subplots
    plt.show()

# Retrieved from tensorflow.org tutorial on regression
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_new_model(n, batchSize, epochs):

    checkpoint_path = "training_ct3/regression_model.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_weights_only=False, verbose=0, save_freq='epoch', save_best_only=False)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, activation='linear', input_shape=[1]),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
    ])

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01, decay=1e-3), loss=tf.keras.losses.MeanSquaredError())
    model.summary()

    x = gen_rand_x(n)
    y = calc_y(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batchSize, verbose=1, callbacks=[cp_callback])
    plot_loss(history)

def clearTerminal():
    """Clears the terminal of all text on Windows/MacOs/Linux"""
    
    # For windows
    if name == 'nt':
        _ = system('cls')
    # Mac and linux 
    else:
        _ = system('clear')


if __name__ == "__main__":

    clearTerminal()
    print("The ANN for the following script is trained to learn the output of the equation |sin(x^x) / 2^(((x^x) - π/2) / π)| with x values between 0 and 3")
    print("An inference ready model should be available upon execution. Choosing to train a new model will overwrite the previously saved model.\n\n")

    while True:
        print("\n(1) Load and Evaluate Model")
        print("(2) Train New Model")
        print("(3) Quit Program\n")
        print(">> ",end='')
        userInput = input()

        if userInput == "1":
            # Load Model
            print("\nLoading TF Model\n")
            model = tf.keras.models.load_model("training_ct3/regression_model.hdf5")

            print("How many values would you like to test? >> ", end='')
            userInput = input()

            n = int(userInput)

            x = gen_rand_x(n)
            y = calc_y(x)

            print("\nEvaluating Model\n")

            results = model.evaluate(x,y)
            preds = model.predict(x)

            plot_results(x,y,preds)

        elif userInput == "2":
            print("\nPlease enter number of datapoints for generated training set: ",end='')
            userInput = input()
            n = int(userInput)
    
            print("\nPlease enter batch size for training: ", end='')
            userInput = input()
            batchSize = int(userInput)

            print("\nPlease enter number of training epochs: ", end='')
            userInput = input()
            epochs = int(userInput)
            
            train_new_model(n, batchSize, epochs)

        elif userInput == "3":
            quit(0)
    
    

    