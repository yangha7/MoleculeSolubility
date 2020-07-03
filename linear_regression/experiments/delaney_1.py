from sklearn.linear_model import LinearRegression
from linear_regression.features import AtomFeature
from linear_regression.dataset_reader import DelaneyReader
from linear_regression.train import train
from linear_regression.visualization import weight_histogram

"""
Model: Linear Regression Model
Features: Atom Number Feature
"""

feature = AtomFeature(list(range(1, 61)))
model = LinearRegression()
reader = DelaneyReader()

train(feature=feature,
      model=model,
      test_ratio=0.1,
      dataset_path="./dataset/delaney.csv",
      dataset_reader=reader)

weight_histogram(model.coef_,
                 feature.description(),
                 mode='cleanup',
                 title='AtomFeature')
