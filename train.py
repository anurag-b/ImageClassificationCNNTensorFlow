#!/usr/bin/env python

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

#Check tensorflow version
#tf.__version__

#Convolution Layer 1
##When dealing with high-dimensional inputs such as images, 
#it is impractical to connect neurons to all neurons in the previous volume. 
#Instead, we will connect each neuron to only a local region of the input volume. 
#The spatial extent of this connectivity is a hyperparameter called the receptive field 
#of the neuron (equivalently this is the filter size). 
#smaller size than input
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

#more filters, featuer map will b
# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

#Get data here
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
#channels mean number of primary colors
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

#print(data.shape)
# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)


###Functions for creating new TensorFlow variables in the given shape and initializing them with random values###
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev = 0.05))

def new_biases(length):
    #equivalent to y intercept
    #constant value carried over across matrix math
    return tf.Variable(tf.constant(0.05,shape=[length]))

#has pooling and ReLU built in. SO think of it as one block
#usually CNNs have 3 blocks repeated

def new_conv_layer(input,             #The previous layer
                   num_input_channels,#Number of channels in previous layer
                   filter_size,       #Width and height of wach filter
                   num_filters,       #Number of filters
                   use_pooling=True): #Use 2x2 max-pooling
    #Shape of the filter-weights for the convolution
    #This format is determined by the Tensorflow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    #Create new weights aka. filters with the given shaoe
    weights = new_weights(shape = shape)

    #Create new biases, one for each filter
    biases = new_biases(length = num_filters)

    #Create the tensorFlow operation for convolution
    #Note the strides are set to 1 in all dimensions
    # The first an dlast stride must always be 1
    # because the first is for the image-numebr and
    # the last is for the input-channel
    # But e.g. strides=[1,2,2,1] would mean that the filter
    # is moved 2 pixels across the x- and y- axis of the image
    # The padding is set to same which means the inout image
    # is padded with zeros so the size of the output is the same
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1],
                         padding='SAME')

    #Add the biases to the result of the convolution
    # A bias-value is added to each filter-channel
    layer += biases
    # Use pooling to down-sample the image resoltion
    if use_pooling:
        #This is a 2x2 max-pooling, which means that we
        #consider 2x2 windows and select the largest value
        #in each window. Then w move 2 pixels to the next window
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               padding='SAME')

    #Rectified Linear Unit(ReLU)
    #It calculates max(x,0) for each input pixel x.
    #This adds some non-linearity to the formula and allows us
    # to learn more complicated functions
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later

    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer
    layer_shape = layer.get_shape()

    # The shapeof the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]
    
    # The number o features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this
    num_features = layer_shape[1:4].num_elements()

    #Reshape the layer to [num_images, num_features]
    # Note that we just set the size of the second dimension
    # to num_features and the size in that dimension is acalculated
    # so the total size of the tensor is unchanged form reshaping
    layer_flat = tf.reshape(layer, [-1, num_features])

    #The sahpe of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer
                 num_inputs,     # Num inputs from previous layer
                 num_outputs,    # Num outputs
                 use_relu=True): # Use ReLU?
    # Create new weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    #Calculate the layer as the matris multiplication of
    # the input and weights, and then add the bias-value
    layer = tf.matmul(input, weights) + biases

    # Use Relu
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

### Create placeholder variables

x =  tf.placeholder(tf.float32, shape=[None, img_size_flat], name = 'x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

### Convolutional layer 1

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)

# print(layer_conv1) # check here

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)

# print(layer_conv2) #check here

layer_flat, num_features = flatten_layer(layer_conv2)

# Uncomment below to check here if everything going good
# print(layer_flat)
# print(num_features)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
# print(layer_fc1)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
# print(layer_fc2)

y_pred = tf.nn.softmax(layer_fc2)

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
# take the average of the cross-entropy for all the image classifications
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# vector of booleans whether the predicted class equals the rue class of each image
correct_prediction = tf.equal(y_pred_cls,y_true_cls)

#Typecast the vector of boolens to floats to make false = 0 and true = 1
# and then claculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#######TENSORFLOW SESSION HERE#############
session = tf.Session()

# initialize the weights and biases
session.run(tf.global_variables_initializer())

train_batch_size = 64

total_iterations = 0

def optimize(num_iterations):
    # Update the global variable rather than local
    global total_iterations

    # Used to print time_usage below
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        # Get a batch of training examples
        # x_batch holds the batch of images
        # y_true_batch are the true labels for those images
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with proper names
        # for placeholder variables in the tensorflow graph and then run the optimizer
        feed_dict_train = {x:x_batch, y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # Tensorflow assigns the variables feed_dict_train
        # to the placeholder variables and then runs the optimizer
        session.run(optimizer, feed_dict = feed_dict_train)

        # Prints status after every 100 iterations
        if i%100 == 0:
            # Calculate yje accuracy on the training-set
            acc = session.run(accuracy, feed_dict = feed_dict_train)

            msg = "Optimization Iterations: {0:>6}, Training Accuracy:{1:6.1%}"

            print(msg.format(i+1,acc))
        # Update the total number of iterations performed
        total_iterations += num_iterations

        end_time = time.time()

        time_diff = end_time - start_time

        # print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))


# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy():

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

# print_test_accuracy()

optimize(num_iterations=10000)
print_test_accuracy()
