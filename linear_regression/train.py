from linear_regression.features import process
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pickle


def train(feature,
          model,
          test_ratio,
          dataset_path,
          dataset_reader):

    # Step 1: Get dataset.
    mol_strings, properties = dataset_reader.read(dataset_path)
    if len(mol_strings) == 0:
        return

    Xs, Ys = process(feature, mol_strings, properties)
    Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(Xs, Ys, test_size=test_ratio, random_state=0)
    # with open(dataset_pkl_path, 'wb') as f:
    #     data = {'Xs': Xs, 'Ys': Ys}
    #     Xs, Ys = data['Xs'], data['Ys']
    #
    # dataset_pkl_path = dataset_path.replace('csv', 'pkl')
    # if os.path.exists(dataset_pkl_path):
    #     # Read Cache.
    #     with open(dataset_pkl_path, 'rb') as f:
    #         data = pickle.load(f)
    #         Xs, Ys = data['Xs'], data['Ys']
    # else:

    # Learn.
    model.fit(Xs_train, Ys_train)

    # Test
    Ys_test_pred = model.predict(Xs_test)
    error = mean_squared_error(Ys_test, Ys_test_pred)
    print('Mean Squared Error = {}'.format(error))
    return error

if __name__ == '__main__':

    from linear_regression.features import AtomFeature
    from linear_regression.dataset_reader import DelaneyReader
    feature = AtomFeature(list(range(1, 61)))
    model = LinearRegression()
    reader = DelaneyReader()
    train(feature, model, 0.1, "../dataset/delaney.csv", reader)
    print(model.coef_)