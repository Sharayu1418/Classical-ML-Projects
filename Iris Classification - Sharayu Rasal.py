#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib scikit-learn seaborn


# In[2]:


from sklearn import datasets
import pandas as pd


# In[3]:


iris = datasets.load_iris()


# In[4]:


X = pd.DataFrame(iris.data, columns=iris.feature_names) 
y = pd.DataFrame(iris.target, columns=['Species'])


# In[5]:


iris_data = pd.concat([X, y], axis=1)


# In[6]:


iris_binary = iris_data[iris_data['Species'] != 2] #Removing Iris-virginica


# In[7]:


X_binary = iris_binary[['sepal length (cm)', 'petal length (cm)']] 
y_binary = iris_binary['Species']


# In[8]:


print(iris_data.head())


# In[9]:


import matplotlib.pyplot as plt

#scatter plot  
plt.scatter(X_binary['sepal length (cm)'], X_binary['petal length (cm)'], c=y_binary, cmap='bwr', edgecolor='k')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Iris Setosa vs Versicolor (Sepal Length vs Petal Length)')
plt.show()
#shows linearly seprable features


# In[10]:


from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.4, random_state=45)


# In[17]:


perceptron = Perceptron(max_iter=1800, tol=1e-3)
perceptron.fit(X_train, y_train)


# In[18]:


y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Perceptron Accuracy: {accuracy}')


# In[19]:


from sklearn.linear_model import SGDClassifier


# In[23]:


adaline = SGDClassifier(loss='squared_error', learning_rate='constant', eta0=0.01, max_iter=1850)


# In[24]:


adaline.fit(X_train, y_train)


# In[25]:


y_pred_adaline = adaline.predict(X_test)
accuracy_adaline = accuracy_score(y_test, y_pred_adaline)
print(f'Adaline Accuracy: {accuracy_adaline}')


# In[27]:


X_binary_three = iris_binary[['sepal length (cm)', 'petal length (cm)', 'sepal width (cm)']]
y_binary = iris_binary['Species'].values 
#to find accuracy with 3 features


# In[28]:


from sklearn.model_selection import train_test_split

#split the dataset into training and testing sets (80% training, 20% testing)
X_train_3, X_test_3, y_train, y_test = train_test_split(X_binary_three, y_binary, test_size=0.2, random_state=1)


# In[29]:


from sklearn.linear_model import Perceptron

# Initialize the Perceptron model
perceptron = Perceptron(max_iter=1000, eta0=0.1, random_state=1)

# Train the Perceptron model on the training data with three features
perceptron.fit(X_train_3, y_train)

# Make predictions on the test set
y_pred_perceptron = perceptron.predict(X_test_3)

# Evaluate the accuracy
accuracy_perceptron = perceptron.score(X_test_3, y_test)
print(f'Perceptron Accuracy with three features: {accuracy_perceptron}')


# In[30]:


from sklearn.linear_model import SGDClassifier
adaline = SGDClassifier(loss='squared_error', learning_rate='constant', eta0=0.01, max_iter=1000)
adaline.fit(X_train_3, y_train)
y_pred_adaline = adaline.predict(X_test_3)
accuracy_adaline = adaline.score(X_test_3, y_test)
print(f'Adaline Accuracy with three features: {accuracy_adaline}')


# In[31]:


from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_binary_four = iris_binary.drop('Species', axis=1)
y_binary = iris_binary['Species'] 
X_train_4, X_test_4, y_train, y_test = train_test_split(X_binary_four, y_binary, test_size=0.3, random_state=42, stratify=y_binary)


# In[33]:


perceptron = Perceptron(max_iter=40, eta0=0.1, random_state=45)
perceptron.fit(X_train_4, y_train)
y_pred_perceptron = perceptron.predict(X_test_4)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)
print(f'Perceptron Accuracy with 4 Features: {accuracy_perceptron * 100:f}%')


# In[34]:


adaline = SGDClassifier(loss='squared_error', max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
adaline.fit(X_train_4, y_train)
y_pred_adaline = adaline.predict(X_test_4)
accuracy_adaline = accuracy_score(y_test, y_pred_adaline)
print(f'Adaline Accuracy with 4 Features: {accuracy_adaline * 100:f}%')


# In[ ]:


#Since the data is truly linearly separable both models can create a perfect decision boundary.
#Linearly Separable Data i.e Setosa and Versicolor used


# In[ ]:


'The Perceptron and Adaline models for binary linearly separable data in the Iris dataset (using two, three, or four features) 
'is due to the fact that these models are well-suited for linearly separable problems. 
'In this case, if you’ve chosen species like Iris setosa and Iris versicolor with features 
'such as petal length and petal width, the data is perfectly linearly separable. 
'The Perceptron algorithm excels in finding a linear decision boundary for such data, ensuring 100% classification accuracy. Similarly, the Adaline model, which minimizes the squared error over the entire dataset, also finds an optimal linear decision boundary that perfectly separates the classes. Adding more features (moving from 2 to 3 or 4) only reinforces this linear separability, allowing both models to continue achieving perfect classification with no misclassifications, which explains the consistent 100% accuracy across different feature sets.


# In[35]:


iris_non_linear = iris_data[iris_data['Species'] != 0]  # Removing Iris Setosa
#using two features
X_non_linear = iris_non_linear[['sepal length (cm)', 'petal length (cm)']]
y_non_linear = iris_non_linear['Species']


# In[36]:


X_train_non_linear, X_test_non_linear, y_train_non_linear, y_test_non_linear = train_test_split(
    X_non_linear, y_non_linear, test_size=0.3, random_state=42, stratify=y_non_linear)
scaler_non_linear = StandardScaler() #for uniform scaling
X_train_non_linear = scaler_non_linear.fit_transform(X_train_non_linear)
X_test_non_linear = scaler_non_linear.transform(X_test_non_linear)


# In[38]:


perceptron_non_linear = Perceptron(max_iter=60, eta0=0.2, random_state=45)
perceptron_non_linear.fit(X_train_non_linear, y_train_non_linear)


# In[39]:


y_pred_perceptron_non_linear = perceptron_non_linear.predict(X_test_non_linear)
accuracy_perceptron_non_linear = accuracy_score(y_test_non_linear, y_pred_perceptron_non_linear)


# In[63]:


print(f'Perceptron Accuracy with Non-Linearly Separable Data (2 Features): {accuracy_perceptron_non_linear * 100:}%')


# In[50]:


adaline_non_linear = SGDClassifier(loss='squared_error', max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
adaline_non_linear.fit(X_train_non_linear, y_train_non_linear)


# In[62]:


y_pred_adaline_non_linear = adaline_non_linear.predict(X_test_non_linear)
accuracy_adaline_non_linear = accuracy_score(y_test_non_linear, y_pred_adaline_non_linear)
print(f'Adaline Accuracy with Non-Linearly Separable Data (2 Features): {accuracy_adaline_non_linear * 100:}%')


# In[54]:


X_train_non_linear_three, X_test_non_linear_three, y_train_non_linear_three, y_test_non_linear_three = train_test_split
(X_non_linear_three, y_non_linear_three, test_size=0.3, random_state=42, stratify=y_non_linear_three)


# In[56]:


perceptron_non_linear_three = Perceptron(max_iter=40, eta0=0.1, random_state=42)
perceptron_non_linear_three.fit(X_train_non_linear_three, y_train_non_linear_three)


# In[67]:


y_pred_perceptron_non_linear_three = perceptron_non_linear_three.predict(X_test_non_linear_three)
accuracy_perceptron_non_linear_three = accuracy_score(y_test_non_linear_three, y_pred_perceptron_non_linear_three)
print(f'Perceptron Accuracy with Non-Linearly Separable Data: {accuracy_perceptron_non_linear_three * 100:f}%')


# In[60]:


adaline_non_linear_three = SGDClassifier(loss='squared_error', max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
adaline_non_linear_three.fit(X_train_non_linear_three, y_train_non_linear_three)


# In[61]:


y_pred_adaline_non_linear_three = adaline_non_linear_three.predict(X_test_non_linear_three)
accuracy_adaline_non_linear_three = accuracy_score(y_test_non_linear_three, y_pred_adaline_non_linear_three)


# In[68]:


print(f'Adaline Accuracy with Non-Linearly Separable Data: {accuracy_adaline_non_linear_three * 100:f}%')


# In[69]:


X_non_linear_four = iris_non_linear[['sepal length (cm)', 'petal length (cm)', 'sepal width (cm)', 'petal width (cm)']]
y_non_linear_four = iris_non_linear['Species']


# In[70]:


X_train_non_linear_four, X_test_non_linear_four, y_train_non_linear_four, y_test_non_linear_four = train_test_split
(X_non_linear_four, y_non_linear_four, test_size=0.3, random_state=42, stratify=y_non_linear_four)


# In[75]:


perceptron_non_linear_four = Perceptron(max_iter=40, eta0=0.1, random_state=42)
perceptron_non_linear_four.fit(X_train_non_linear_four, y_train_non_linear_four)


# In[72]:


y_pred_perceptron_four = perceptron_non_linear_four.predict(X_test_non_linear_four)
accuracy_perceptron_four = accuracy_score(y_test_non_linear_four, y_pred_perceptron_four)


# In[74]:


print(f"Perceptron Accuracy with 4 Features: {accuracy_perceptron_four:f}")


# In[76]:


adaline_non_linear_four = SGDClassifier(loss='squared_error', max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
adaline_non_linear_four.fit(X_train_non_linear_four, y_train_non_linear_four)


# In[77]:


y_pred_adaline_four = adaline_non_linear_four.predict(X_test_non_linear_four)
accuracy_adaline_four = accuracy_score(y_test_non_linear_four, y_pred_adaline_four)


# In[78]:


print(f"Adaline Accuracy with 4 Features: {accuracy_adaline_four:f}")


# In[ ]:


'''''For non-linearly separable data, the differences in accuracy between the Perceptron and Adaline models are due to how they handle classification when a linear decision boundary isn't sufficient.

The Perceptron model relies strictly on linear decision boundaries. Since the data isn't perfectly separable, it struggles to create a boundary that accurately classifies all points, leading to lower accuracy (90%) with fewer features. As more features are added (3 or 4), the model has more information to work with, and the boundary improves, reaching 96% accuracy with 4 features.

Adaline, which minimizes a continuous error function over all samples (not just misclassified ones like Perceptron), generally performs better on non-linear data because it tries to find an optimal solution in a least-squares sense. That's why it achieves 96% accuracy with two features—better than the Perceptron. However, when you add more features (3 or 4), the additional information may introduce noise or confusion in non-linearly separable data, leading to a slight drop in performance for Adaline compared to Perceptron, as seen in the 90% accuracy with four features.

In summary, Adaline outperforms Perceptron with fewer features due to its ability to optimize globally, but as more features are added, the Perceptron catches up, even surpassing Adaline with four features, because the additional dimensions allow it to create better boundaries in higher-dimensional space.

