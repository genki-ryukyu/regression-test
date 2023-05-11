import numpy as np

def load_linear_example():
    X = np.array([[1,4],[1,8],[1,13],[1,17]])
    Y = np.array([7,10,11,14])
    return X,Y

if __name__ == "__main__":
    X, Y = load_linear_example()
    print(f'{X=}')
    print(f'{Y=}')