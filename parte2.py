import pandas
import numpy as np

def load_data(file):
    data=pandas.read_csv(file)
    array1 = np.transpose(np.array([data['x1'].values, data['x2'].values]))
    array2 = np.transpose(np.array([data['y1'].values, data['y2'].values]))
    return array1,array2


def main():
    load_data("part2_train_data.csv")




if __name__ == '__main__':
    main()