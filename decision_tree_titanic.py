#%% md
# ### Objective
# To establish the likeleyhood of surviving the Titanic incedent or not surviving the Titanic incedent.
# 
# Create a Decision Tree that can predict the survival of passengers on the Titanic. Make sure not to impose any restrictions on the depth of the tree.
#%% md
# ### Import needed Libraries
#%%
# Import libraries
import pandas as pd
import numpy as np
from scipy.ndimage import histogram
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# for visualisation
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from IPython.display import Image  
import seaborn as sns
from subprocess import call
#%%

#%% md
# ### Import the data set
#%%
# Import data and display columns headings (axis = 1)

titanic_df = pd.read_csv("titanic.csv")
titanic_df.head()
#%% md
# ### Clean data up 
#%%
# Define current data values, objects and null versus non-null
 
titanic_df.info()
#%% md
# #### Looking at the data. (891)
# 
# 
# 
# Passenger ID is fine. ( Leave out due to relevance )
# 
# Survived is fine. ( Dependant variable )
# 
# Pclass is fine 
# 
# Name is object so leave out or convert. ( Leave out )
# 
# Sex is object so leave out or convert. ( Convert )
# 
# Age is missing values so substitute with mean value for age. 
# 
# Sibsp is fine 
# 
# Parch is fine 
# 
# Ticket is object, so leave out or convert ( Leave out )
# 
# Fare is fine
# 
# Cabin is object and missing values so leave out or convert ( Leave out )
# 
# Embarked is missing values and is object so leave out or convert ( Convert )
# 
#%%
# Drop, name, fare, cabin, parch, ticket on axis 1

titanic_df.drop("Name", axis=1, inplace=True)
titanic_df.drop("Ticket", axis=1, inplace=True)
titanic_df.drop("Fare", axis=1, inplace=True)
titanic_df.drop("Parch", axis=1, inplace=True)
titanic_df.drop("Cabin", axis=1, inplace=True)
#%%
# Fill in the average age for the ages that are missing 
# Average Age for missing values (29)

df2 = titanic_df["Age"].mean()
titanic_df.fillna(df2, inplace=True)

print(f"The average age of the Titanic passanger was {int(df2)} years old.")
#%% md
# ### One-Hot Encoding
# One-hot encoding is a technique used to ensure that categorical variables are better represented in the machine. Let's take a look at the "Sex" column
#%%
titanic_df["Sex"].unique()
#%% md
# Machine Learning classifiers don't know how to handle strings. As a result, you need to convert it into a categorical representation. There are two main ways to go about this:
# 
# Label Encoding: Assigning, for example, 0 for "male" and 1 for "female". The problem here is it intrinsically makes one category "larger than" the other category.
# 
# One-hot encoding: Assigning, for example, [1, 0] for "male" and [0, 1] for female. In this case, you have an array of size (n_categories,) and you represent a 1 in the correct index, and 0 elsewhere. In Pandas, this would show as extra columns. For example, rather than having a "Sex" column, it would be a "Sex_male" and "Sex_female" column. Then, if the person is male, it would simply show as a 1 in the "Sex_male" column and a 0 in the "Sex_female" column.
# 
# There is a nice and easy method that does this in pandas: get_dummies()
#%%
titanic_df = pd.get_dummies(titanic_df, prefix="Sex", columns=["Sex"])
titanic_df.head()
#%% md
# Now, we do the same to the "Embarked" column.
#%%
titanic_df = pd.get_dummies(titanic_df, prefix="Embarked", columns=["Embarked"])
titanic_df.head()
#%%
# Define new data values, objects and null versus non-null
 
titanic_df.info()
#%% md
# ### Select relevant variables from the data and split the data into a training, development, and test set.
#%% md
# #### Independant Variables
#%%
# Independant Variables

X = titanic_df[["Pclass","Age","Sex_male","Sex_female","Embarked_S","Embarked_C","Embarked_Q"]]
X.info()
#%% md
# #### Dependant Variable
#%%
# Dependant Variable

y = titanic_df["Survived"]
y.info()
#%%
# What the graphic representation of the Survived looks like  

f, ax = plt.subplots(1, 2, figsize=(12, 4)) 
titanic_df['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=False) 
ax[0].set_title('Survived (1) and Not-Survived (0)') 
ax[0].set_ylabel('') 
sns.countplot(x='Survived', data=titanic_df, ax=ax[1])
ax[1].set_ylabel('Quantity') 
ax[1].set_title('Survived (1) and Not-Survived (0)') 
plt.show()
#%% md
# #### Split into Training Development and test set. 
# Assign a random state of 42
# 
# Assign a 20 % Test and 80 % Training set
#%%
# Random state.

r = 42 # Random state set at 42

# X TrainFull, Y TrainFull.

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=r) # 80% training and 20% test
#%% md
# #### Split into Training Development and test set. 
# Assign a random state
# 
# Assign a 20 % Test and 80 % Training set
#%%
# X Development and Y Development.

X_train, X_dev, y_train, y_dev = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=r) # 80% training and 20% development
#%%
model = DecisionTreeClassifier()

model.fit(X_train, y_train)
#%% md
# 
# ### Train a decision tree and make a plot of it.
#%%
# Make a plot of the decision tree

plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=list(X.columns), class_names=['Not Survived', 'Survived'], filled=True)
plt.show()
#%% md
# ### Compute your model’s accuracy on the development set.
#%%
# Compute your model’s accuracy on the development set.

score = model.score(X_dev, y_dev)
print(f"Accuracy on the development set: {model.score(X_dev, y_dev)}")
#%% md
# ### Try building your model with different values of the max_depth [2-10]. At each step, create a plot of your tree and store the accuracies on both the training and development data.
#%%
# Develop for loop to define 2 - 11 iterations of max-depths outcomes. 

max_depths = list(range(2, 11))
train_accuracy = []
dev_accuracy = []

for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train, y_train)

    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=list(X.columns), class_names=['Not Survived', 'Survived'], filled=True)
    plt.title(f"Decision Tree Max depth {depth}")
    plt.show()

    train_score = model.score(X_train, y_train)
    dev_score = model.score(X_dev, y_dev)

    train_accuracy.append(train_score)
    dev_accuracy.append(dev_score)
#%% md
# ### Plot a line of your training accuracies and another of your development accuracies in the same graph. Write down what shape the lines have and what this shape means.
#%%
# Accuracy versus depth training and development plot.

plt.plot(max_depths, train_accuracy, label='Training Accuracy')
plt.plot(max_depths, dev_accuracy, label='Development Accuracy')
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title("Accuracy vs Max Depth")
plt.legend()
plt.show()

#%% md
# #### The development accuracy increases up until depth 4 where it starts decreasing. 
# #### The training accuracy keeps on steadily increasing.
# #### After depth 4 the training accuracy and development accuracy diverge making the model less accurate after depth 4
#%%
# Best depth for model.

best_depth = max_depths[np.argmax(dev_accuracy)]
print(f"Best depth for model is: {best_depth}")

#%% md
# ### Report the accuracy of your final model on the test data.
#%%
# Making predictions
model.fit(X_train, y_train)
model.score(X_test, y_test)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate the accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print('The training accuracy is', train_accuracy)
print('The test accuracy is', test_accuracy)
print('The best depth for the model', best_depth)
#%% md
# We have our Decision Tree. It achieves an accuracy of 83.13% across the dataset.
# At a Max-Depth which is set at 4
#%%
# decision_tree = tree.DecisionTreeClassifier at max_depth of 4

decision_tree = tree.DecisionTreeClassifier(max_depth = 4)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)

decision_tree.fit(X_train, y_train)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
print('The model accuracy is', acc_decision_tree)
