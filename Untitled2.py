
# coding: utf-8

# In[1]:

from numpy import genfromtxt, zeros
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
# read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)
print (data.shape)
print (target.shape)
print (set (target))


# In[2]:

from pylab import plot, show
plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
show()


# In[3]:

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3


# In[4]:

from sklearn.neighbors.nearest_centroid import NearestCentroid
classifier = NearestCentroid()
classifier.fit(data,t)


# In[5]:

print (classifier.predict(data[0]))


# In[6]:

print (t[0])


# In[7]:

from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t
, test_size=0.4, random_state=0)


# In[8]:

classifier.fit(train,t_train) 
print (classifier.score(test,t_test))


# In[9]:

from sklearn.metrics import confusion_matrix
print (confusion_matrix(classifier.predict(test),t_test))


# In[ ]:



