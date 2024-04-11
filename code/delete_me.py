import numpy as np

full_list = ['a','b', 'c','d','e','f','g']
partial_list = ['a','c','g']

partial_data = np.array([[0,0,1],[0,0,1],[1,0,1],[1,0,0],[0,0,0],[0,1,0]])
print(partial_data)
rows,cols = partial_data.shape

for index, key in enumerate(full_list):
    if key not in partial_list:
        partial_data = np.hstack((partial_data[:,:index],np.zeros((rows,1)),partial_data[:,index:]))

