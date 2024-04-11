import os
import pandas as pd
import numpy as np
from perceptron import Perceptron

chars = "abcdefghijklmnopqrstuvwxyz"


def train_network(training_set, layer, epoch=100, print_epoch = True):
    for epoch in range(epoch):
        if print_epoch:
            print(f'Epoch {epoch}')
        for index, row in training_set.iterrows():
            target = df.loc[row['class']].tolist()
            for perceptron, target_value in zip(layer, target):
                prediction = perceptron.predict(row[:-1].values)
                if print_epoch:
                    print(f'Prediction: {prediction} - Target: {target_value}')
                perceptron.correction(row[:-1].values, prediction, target_value)
            if print_epoch:
                print('-----------------')


def process_data():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'lang'

    num_inputs = os.listdir(dir_path).__len__()

    classes = os.listdir(dir_path)
    class_matrix = np.eye(num_inputs)

    df = pd.DataFrame(class_matrix, columns=classes).transpose()

    vectors = {lang: [] for lang in classes}
    for lang_dir in os.listdir(dir_path):
        path = dir_path + os.sep + lang_dir
        for file in os.listdir(path):
            with open(path + os.sep + file, 'rb') as f:
                lang_vector = np.zeros(26)
                count = 0
                while (byte := f.read(1)):
                    try:
                        decoded_byte = byte.decode('utf-8').lower()
                        if decoded_byte in chars:
                            lang_vector[chars.index(decoded_byte)] += 1
                            count += 1
                    except UnicodeDecodeError:
                        pass
                lang_vector = lang_vector / count
                vectors[lang_dir].append(lang_vector)

    return classes, df, vectors

def test_network(text, layer, df):
    lang_vector = np.zeros(26)
    count = 0
    for char in text:
        if char in chars:
            lang_vector[chars.index(char)] += 1
            count += 1
    lang_vector = lang_vector / count

    prediction = [perceptron.predict(lang_vector) for perceptron in layer]
    max_similarity = 0
    lang = ''
    for index, row in df.iterrows():
        evaluation = np.dot(prediction, row)
        if evaluation > max_similarity:
            max_similarity = evaluation
            lang = index
    print(f'Text: {text} - Language: {lang} , Prediction: {max_similarity}')


def generate_train_set():
    training_set = pd.DataFrame()
    for lclass in classes:
        testing_slice = pd.DataFrame(vectors[lclass])
        testing_slice['class'] = lclass
        training_set = pd.concat([training_set, testing_slice])
    return training_set

def print_menu():
    print('1. Predict text',
          '2. Train network',
          '3. Print language vectors',
          '4. Print every perceptron',
          '5. Exit', sep='\n')


if __name__ == '__main__':

    classes, df, vectors = process_data()

    # print(df)

    layer = [Perceptron(26) for i in range(classes.__len__())]

    training_set = generate_train_set()
    # train_network(training_set, layer, epoch=1000, print_epoch=False)
    #
    # text_to_predict = 'Während die anderen Landesteile des Vereinigten Königreichs (Nordirland, Schottland, Wales) aufgrund der Devolution eigene Parlamente besitzen, die in den entsprechenden Landesteilen Sonderrechte gegenüber der britischen Zentralregierung in London haben, ist dies für England nicht der Fall.'
    # test_network(text_to_predict, layer, df)
    #
    print_menu()
    while (user_input := input('Input: ').lower()) != '5':

        if user_input == '1':
            text_to_predict = input('Enter text to predict: ')
            test_network(text_to_predict, layer, df)
        elif user_input == '2':
            train_network(training_set, layer, epoch=1000)
        elif user_input == '3':
            print(df)
        elif user_input == '4':
            for perceptron in layer:
                print(perceptron)
        else:
            print('Invalid option')
            print_menu()
        print_menu()