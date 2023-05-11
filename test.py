import dataset
import regression
import importlib
importlib.reload(regression)

 
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

#regressionのver.3テスト
print(model.predict(X))

#regressionのver.4テスト
print(model.score(X,Y))



