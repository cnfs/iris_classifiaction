from __future__ import absolute_import, division, print_function ## to remove incompatibility problems in future from  future is used

import os
import matplotlib.pyplot as plt

import tensorflow as tf
##this part is for enabling eager execution so that it does not build a
##computational graph
import tensorflow.contrib.eager as tfe
import pandas as pd

tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

##now importing data set classify based on length,width of petals,sepals
##will use tf.keras.utils.get_file
##C:\Users\chirag dawra\.keras\datasets\iris_training.csv 

train_url="http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp=tf.keras.utils.get_file(fname=os.path.basename(
                     train_url),origin=train_url)
print("local copy of data set: {}".format(train_dataset_fp))

##check or inspect the data
print("pd __version __ {}".format(pd.__version__))
##iris_dataframe=pd.read_csv("http://download.tensorflow.org/data/iris_training.csv",sep=",")
##iris_dataframe
#parsing the database for our use

def parse_csv(line): #each line or row will be passed to this function line to line
    example_defaults=[[0.] , [0.] ,[0.] , [0.] ,[0] ] #setting field type first four are float andfeatures and last label
    parsed_line=tf.decode_csv(line,example_defaults) ## earlier we used pandas but this csv file is in unicode so we decode it using tf.decode_csv
    features=tf.reshape(parsed_line[:-1] ,shape=(4,) ) ##we took first four features and clubbed them into a tensort of shape (4,) the no. of rows of data we dont know
    label=tf.reshape(parsed_line[-1],shape=() ) ## we took the last column as label i.e 0,1,2 for different species of iris we took its a 0 shape tensor
    return features,label

##now we will use tensor flow dataset api for reading or transforming it into a form used for training
train_dataset=tf.data.TextLineDataset(train_dataset_fp) ##this will be used for passing the data from original fp dataset line by line to parse csv
train_dataset=train_dataset.skip(1)  ## skip the header part of the data
train_dataset=train_dataset.map(parse_csv)  ##it combines scalar features and label to combined (features,label) pairs
train_dataset=train_dataset.shuffle(buffer_size=1000) ##randomizing the data give the max buffer size more than the data
train_dataset=train_dataset.batch(32)

features,label=iter(train_dataset).next()
print("example features : " , features[0] )
print("example label : " , label[0] )

##now prepare the model it requires the decision of how many layers to include will do this uding keras api
## we have used 2 hidden layers with 10 neurons each and are fully connected
model =tf.keras.Sequential( [ tf.keras.layers.Dense(10,activation="relu", input_shape=(4,) ),  ## the first hidden layer which receives the input features
                             tf.keras.layers.Dense(10,activation="relu" ),  ## the second hidden layer with 10 neurons
                             tf.keras.layers.Dense(3) ] )  ## this the output layer with 3 neurons outputs the probabilities for the occurence of each type of species

## now calculating the loss using tf.losses.sparse_softmax_cross_entropy
def loss(model ,x ,y): ## this is a function which receives over model as input,input features x ,label y
    y_=model(x)   ## here y_ will have the value what features return after passing through our model
    return tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_ ) ## return the loss with the labels =y,our model prediction y_

## now calculating the gradients which will be used for optimizing the loss
def grad(model ,inputs ,targets ):
    with tf.GradientTape() as tape:
        loss_value=loss(model,inputs,targets) ##
    return tape.gradient(loss_value,model.variables)
##now building an optimizer we will use stochastic gradient descent for claculating the gradient for the batch of 32 which we have defined
##in our dataset we took randomly
## we run the batches for a fixed no. of times over the whole data which is called num_epochs, epoch is one iteration over the batch
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train_loss_results = [] ##will store the loss for each epoch
train_accuracy_results=[] ##will store the accuracy
num_epochs=201

for epoch in range(num_epochs):
    epoch_loss_avg=tfe.metrics.Mean() ## i think it is random initialization
    epoch_accuracy=tfe.metrics.Accuracy() ##same as above

    for x,y in train_dataset: ## x=labels y=features over batch of 32
        grads=grad(model,x,y) ##will return the gradient defined above
        optimizer.apply_gradients(zip(grads,model.variables),
                                 global_step=tf.train.get_or_create_global_step() )
        epoch_loss_avg(loss(model,x,y))
        epoch_accuracy( tf.argmax(model(x),axis=1,output_type=tf.int32) ,y)
    train_loss_results.append(epoch_loss_avg.result)
    train_accuracy_results.append(epoch_accuracy.result)
    if epoch %50 ==0:
        print( "Epoch {:03d}: Loss : {:.3f} ,Accuracy: {: .3%} ".format(epoch,epoch_loss_avg.result(),epoch_accuracy.result() ) )

##now ploting the curve using matplotlib
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()

        
      
