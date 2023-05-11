import dataset
import regression
import importlib

 
X, Y = dataset.load_linear_example()
#datasetのテスト
print(f'{X=}')
print(f'{Y=}')
#regressionのver.1テスト
model = regression.LinearRegression()
print(model.x)

#regressionのver.2テスト
model.fit(X,Y)
print(model.theta)


