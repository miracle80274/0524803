
# coding: utf-8

# In[2]:

from numpy import genfromtxt, zeros
data = genfromtxt('iris.csv',delimiter=',',usecols=(0,1,2,3))
# read the fifth column
target = genfromtxt('iris.csv',delimiter=',',usecols=(4),dtype=str)
print (data.shape)
print (target.shape)
print (set (target))


# In[3]:

from pylab import plot, show
plot(data[target=='setosa',0],data[target=='setosa',2],'bo')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'ro')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
show()


# In[4]:

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3


# In[14]:

from sklearn.neighbors.nearest_centroid import NearestCentroid
classifier = NearestCentroid()
classifier.fit(data,t)


# In[6]:

print (classifier.predict(data[0]))


# In[7]:

print (t[0])


# In[15]:

from sklearn import cross_validation
train, test, t_train, t_test = cross_validation.train_test_split(data, t
, test_size=0.4, random_state=0)


# In[9]:

classifier.fit(train,t_train) 
print (classifier.score(test,t_test))


# In[10]:

from sklearn.metrics import confusion_matrix
print (confusion_matrix(classifier.predict(test),t_test))


# In[11]:

from pylab import plot, show
plot(data[target=='setosa',0],data[target=='setosa',2],'b+')
plot(data[target=='versicolor',0],data[target=='versicolor',2],'r*')
plot(data[target=='virginica',0],data[target=='virginica',2],'go')
show()


# In[12]:

from pylab import plot, show
plot(data[target=='setosa',1],data[target=='setosa',3],'b+')
plot(data[target=='versicolor',1],data[target=='versicolor',3],'r*')
plot(data[target=='virginica',1],data[target=='virginica',3],'go')
show()


# In[13]:

from pylab import figure, subplot, hist, xlim, show
xmin = min(data[:,0])
xmax = max(data[:,0])
figure()
subplot(411) # distribution of the setosa class (1st, on the top)
hist(data[target=='setosa',0],color='b',alpha=.7)
xlim(xmin,xmax)
subplot(412) # distribution of the versicolor class (2nd)
hist(data[target=='versicolor',0],color='r',alpha=.7)
xlim(xmin,xmax)
subplot(413) # distribution of the virginica class (3rd)
hist(data[target=='virginica',0],color='g',alpha=.7)
xlim(xmin,xmax)
subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)
show()


# In[17]:

from sklearn.metrics import classification_report
print (classification_report(classifier.predict(test), t_test,
target_names=['setosa', 'versicolor', 'virginica']))


# In[ ]:



