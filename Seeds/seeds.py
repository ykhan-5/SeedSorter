import pandas 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load in dataset
names = ['area A', 'perimeter', 'compactness', 'length of kernel', 'width of kernel', 'asymmetry coefficient', 'length of kernel groove', 'class']
dataset = pandas.read_csv("/Users/temp/Desktop/CS/Summer-23/ML/Seeds/Data/seeds_dataset.csv", names=names, delimiter=' ')

print("\n", dataset.dtypes)
print('\n')
print(dataset.columns)
print('\n')

# Define the mapping of old values to new values
class_mapping = {1: "Kama", 2: "Rosa", 3: "Canadian"}

# Use the replace method to rename the elements in the 'class' column
dataset['class'] = dataset['class'].replace(class_mapping)

# Convert the 'class' column to string type
dataset['class'] = dataset['class'].astype(str)

print(dataset['class'].unique(),"\n\n")

# Find Shape 
print("\033[1mThe dataset size dimensions [(row,col)] are {shape}\033[0m" .format(shape=dataset.shape))

print("\n\033[1mThe first 30 entries:\033[0m")
print(dataset.head(10))
print('\n')

#Class-distribution 
print("\n\033[1mThe dataset by size is:\033[0m")
print(dataset.groupby('class').size(),'\n\n'); # group the instances based on a specified column and get the size.

print("\n\033[1mThe dataset by mean is:\033[0m")
print(dataset.groupby('class').mean(),'\n\n')

# Describe
print(dataset.describe())
print('\n')

#Boxplot
dataset.plot(kind="box", subplots=True, layout=(1,7), sharex=False,sharey=False)
plt.show()

#Histogram 
dataset.hist()
plt.show()

#Scatter plot matrix
scatter_matrix(dataset)
plt.show()

#Split out dataset (validation)
array = dataset.values
X = array[:,0:7]
Y = array[:,7]

validation_size = .20
seed = 9

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) 

#Test
seed = 9
scoring = 'accuracy'

#Build model
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#Evaluate each for dataset
results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(cv_results)
    print(msg)

#Compare Algorithms 
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#Make predictuions using LDA (Had best performance)
print("Predicting on unseen data.")
lda = LinearDiscriminantAnalysis()                                   # sklearn lib. stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point
lda.fit(X_train, Y_train)                                      # train the model with the train dataset
predictions = lda.predict(X_validation)                        # get the predictions using the validation test with this model knn
print(accuracy_score(Y_validation, predictions))               # compares the validation known answer with the predicted to determine accuracy
print(confusion_matrix(Y_validation, predictions))             # matrix of accuracy classification where C(0,0) is true negatives, C(1,0) is false negatives, C(1,1) true posivtes, C(0,1) false positvies.
print(classification_report(Y_validation, predictions))        # text report
print("X_validation predict ===")
print(predictions);                                            # array of predicted values

for row_index, (input, predictions, Y_validation) in enumerate(zip (X_validation, predictions, Y_validation)):
  if predictions != Y_validation:
    print('Row', row_index, 'has been classified as ', predictions, 'and should be ', Y_validation)
    print(X_validation[row_index])
  else:
    print("Correct!\n")