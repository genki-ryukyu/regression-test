import dataset
import regression
 
X, Y = dataset.load_linear_example()

print(f'{X=}')
print(f'{Y=}')

model = regression.LinearRegression()
print(model.x)


