from sklearn.linear_model import LinearRegression
from linear_regression.features import AtomFeature, FunctionalGroupFeature, IntegrateFeature
from linear_regression.dataset_reader import DelaneyReader
from linear_regression.train import train
from linear_regression.visualization import weight_histogram

"""
Model: Linear Regression Model
Features: Atom Number Feature, Functional Group Feature
"""

atom_feature = AtomFeature(list(range(1, 61)))
function_group_feature = FunctionalGroupFeature()
feature = IntegrateFeature([atom_feature, function_group_feature])

model = LinearRegression()
reader = DelaneyReader()

train(feature=feature,
      model=model,
      test_ratio=0.1,
      dataset_path="./dataset/delaney.csv",
      dataset_reader=reader)

# print(model.coef_)
coef, name = weight_histogram(model.coef_, feature.description(),
                              mode='cleanup',
                              title='Atom & Functional Group Feature')

print(coef, name)
