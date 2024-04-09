import os
import pandas as pd
import numpy as np
from perceptron import Perceptron

chars = "abcdefghijklmnopqrstuvwxyz"

if __name__ == '__main__':

    dir_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'lang'

    num_inputs = os.listdir(dir_path).__len__()
    classes = os.listdir(dir_path)
    class_matrix = np.eye(num_inputs)
    df = pd.DataFrame(class_matrix, columns=classes).transpose()
    print(df)

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

    layer = [Perceptron(26, classes[i]) for i in range(classes.__len__())]
