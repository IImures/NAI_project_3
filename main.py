import os
import pandas as pd
import numpy as np
from perceptron import Perceptron

chars = "abcdefghijklmnopqrstuvwxyz"


def train_network(epoch=100, print_epoch = True):
    global perceptron
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


if __name__ == '__main__':

    classes, df, vectors = process_data()

    layer = [Perceptron(26) for i in range(classes.__len__())]

    training_set = pd.DataFrame()
    for lclass in classes:
        testing_slice = pd.DataFrame(vectors[lclass])
        testing_slice['class'] = lclass
        training_set = pd.concat([training_set, testing_slice])

    train_network(epoch=1000)
