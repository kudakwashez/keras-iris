import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.cross_validation import train_test_split


from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils


# In[4]:

iris = sns.load_dataset("iris")
sns.pairplot(iris, hue='species')

X=iris.values[:,:4]
y=iris.values[:,4]


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.5,random_state=1)

# keras needs inputs to be vectors , so one hot encoding values
def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))
y_train_ohe=one_hot_encode_object_array(y_train)
y_test_ohe=one_hot_encode_object_array(y_test)


model=Sequential()

model.add(Dense(16,input_shape=(4,)))
model.add(Activation("sigmoid"))

model.add(Dense(3))
model.add(Activation("softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(X_train,y_train_ohe,nb_epoch=100,batch_size=1,verbose=1)

loss, accuracy = model.evaluate(X_test, y_test_ohe, verbose=1)
print("Accuracy = {:.2f}".format(accuracy))


