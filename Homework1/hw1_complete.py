import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import person
from sklearn.datasets import load_iris
import tensorflow as tf
rng = np.random.default_rng(2022)


## Here's the information needed to do the first few tasks,
# which will give you some practice with basic Python methods
list_of_names = ['Roger', 'Mary', 'Luisa', 'Elvis']
list_of_ages  = [23, 24, 19, 86]
list_of_heights_cm = [175, 162, 178, 182]

for name in list_of_names:
  print("The name {:} is {:} letters long".format(name, len(name)))

# List comprehension to make a list of the lengths of the names
lengths_of_names = [len(name) for name in list_of_names]
print(lengths_of_names)

# Create a dictionary named "people"
people = {}

# Iterate through the names and create Person objects
for name, age, height in zip(list_of_names, list_of_ages, list_of_heights_cm):
    created_person = person.person(name, age, height)
    people[name] = created_person
    
# Convert the lists to NumPy arrays
ages_array = np.array(list_of_ages)
heights_array = np.array(list_of_heights_cm)

# Calculate the average age
average_age = np.mean(ages_array)

# Create a scatter plot
plt.scatter(ages_array, heights_array, label='People')

# Add gridlines, labels, and title
plt.grid(True)
plt.xlabel('Ages')
plt.ylabel('Heights (cm)')
plt.title('Scatter Plot of Ages vs Heights')

# Add legend (optional, in this case)
plt.legend()

# Save the plot as a PNG file
plt.savefig('scatter_plot.png')

# Show the plot (optional, depending on your needs)
plt.show()

########################################
# Here's the information for the second part, involving the linear
# classifier

# import the iris dataset as a pandas dataframe
iris_db = load_iris(as_frame=True) 
x_data = iris_db['data'] 
y_labels = iris_db['target'] # correct numeric labels
target_names = iris_db['target_names'] # string namesy


# Here's a starter example of plotting the data
fig=plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
l_colors = ['maroon', 'darkgreen', 'blue']
for n, species in enumerate(target_names):    
  plt.scatter(x_data[y_labels==n].iloc[:,0], 
              x_data[y_labels==n].iloc[:,1], 
              c=l_colors[n], label=target_names[n])
plt.xlabel(iris_db['feature_names'][0])
plt.ylabel(iris_db['feature_names'][1])
plt.grid(True)
plt.legend() # uses the 'label' argument passed to scatter()
plt.tight_layout()
# uncomment this line to show the figure, or use
# interactive mode -- plt.ion() --  in iPython
# plt.show()
plt.savefig('iris_data.png')


## A trivial example classifier.  You'll copy and modify this to 
# perform a linear classification function.
def classify_rand(x):    
  return rng.integers(0,2, endpoint=True)

# Define the linear classifier function
def classify_iris(features):
    # Given weights and biases
    weights = np.array([[0.7230, -0.5053, -0.8363,  0.1594],
                        [0.4765, -0.9116,  0.4933,  -0.7420],
                        [0.9793, -2.5114,  0.3961,  0.5976]])
    biases = np.array([0.1, 0.2, 0.3])

    # Perform matrix-vector multiplication
    scores = np.matmul(weights, features) + biases

    # Classify based on the index of the maximum score
    label = np.argmax(scores)

    return label


# A function to measure the accuracy of a classifier and
# create a confusion matrix.  Keras and Scikit-learn have more sophisticated
# functions that do this, but this simple version will work for
# this assignment.
def evaluate_classifier(cls_func, x_data, labels, print_confusion_matrix=True):
  n_correct = 0
  n_total = x_data.shape[0]
  cm = np.zeros((3,3))
  for i in range(n_total):
    x = x_data[i,:]
    y = cls_func(x)
    y_true = labels[i]
    cm[y_true, y] += 1
    if y == y_true:
      n_correct += 1    
    acc = n_correct / n_total
  print(f"Accuracy = {n_correct} correct / {n_total} total = {100.0*acc:3.2f}%")
  if print_confusion_matrix:
    print(f"{12*' '}Estimated Labels")
    print(f"              {0:3.0f}  {1.0:3.0f}  {2.0:3.0f}")
    print(f"{12*' '} {15*'-'}")
    print(f"True    0 |   {cm[0,0]:3.0f}  {cm[0,1]:3.0f}  {cm[0,2]:3.0f} ")
    print(f"Labels: 1 |   {cm[1,0]:3.0f}  {cm[1,1]:3.0f}  {cm[1,2]:3.0f} ")
    print(f"        2 |   {cm[2,0]:3.0f}  {cm[2,1]:3.0f}  {cm[2,2]:3.0f} ")
    print(f"{40*'-'}")
  ## done printing confusion matrix  

  return acc, cm

## Now evaluate the classifier we've built.  This will evaluate the
# random classifier, which should have accuracy around 33%.
acc, cm = evaluate_classifier(classify_rand, x_data.to_numpy(), y_labels.to_numpy())
acc, cm = evaluate_classifier(classify_iris, x_data.to_numpy(), y_labels.to_numpy())

tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = tf_model.fit(x_data.to_numpy(),  y_labels.to_numpy(), epochs=50, batch_size=32, validation_split=0.2)

train_loss, train_accuracy = tf_model.evaluate(x_data.to_numpy(), y_labels.to_numpy())
print("Training Loss:", train_loss)
print("Training Accuracy:", train_accuracy)