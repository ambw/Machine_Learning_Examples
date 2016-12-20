from mnist import MNIST
import numpy as np
import pandas as pd

mndata = MNIST('./data')
imgs, labels = mndata.load_training()

# img = np.asarray(imgs)
label = np.asarray(labels)

y_matrix = pd.get_dummies(label.ravel()).as_matrix()
print(label.ravel())
print(label)