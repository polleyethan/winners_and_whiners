import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_points_data


class TeamModel(tf.keras.Model):
    def __init__(self):
        super(TeamModel, self).__init__()

        self.batch_size = 100 #TODO 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        self.dense1 = tf.keras.layers.Dense(25, activation="relu")

        self.dense2 = tf.keras.layers.Dense(10, activation="relu")

        #self.dense3 = tf.keras.layers.Dense(8, activation="relu")

        self.dense4 = tf.keras.layers.Dense(2)


    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs, 
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU 
        """
        
        dense_1 = self.dense1(inputs)

        dense_2 = self.dense2(dense_1)
        #dense_3 = self.dense3(dense_2)

        dense_4 = self.dense4(dense_2)
        #softmax = tf.nn.softmax(dense_4)
        print(dense_4)
        return dense_4

    def loss(self, logits, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        
        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """
        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

        for x in range(len(logits)):
            print("******************************")
            print("Score Predicted"+str(logits[x]))
            print("Score Actual: "+str(labels[x]))

        return tf.reduce_mean(self.mse(labels, logits))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    #TODO: Fill in

    #train_inputs = np.reshape(train_inputs, (-1, model.window_size))
    #train_labels = np.reshape(train_labels, (-1, model.window_size))
    

    inputs = train_inputs#tf.convert_to_tensor(train_inputs)
    labels = train_labels#tf.convert_to_tensor(train_labels)
    print(labels)
    for i in range(0, len(train_inputs), model.batch_size):
        batch_ins = inputs[i:i+model.batch_size]
        batch_outs = labels[i:i+model.batch_size]
        with tf.GradientTape() as tape:

            probs = model.call(batch_ins)

            loss = model.loss(probs, batch_outs)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """
    
    #TODO: Fill in
    #NOTE: Ensure a correct perplexity formula (different from raw loss)

    inputs = tf.convert_to_tensor(test_inputs)
    outs = tf.convert_to_tensor(test_labels)

    avg_loss = 0
    steps = 0
    ste = 0
    correct = 0

    for i in range(0, len(inputs), model.batch_size):
        batch_ins = inputs[i:i+model.batch_size]
        batch_outs = outs[i:i+model.batch_size]
        probs = model.call(batch_ins)
        loss = model.loss(probs, batch_outs)
        avg_loss += loss
        steps += 1
        for x in range(len(batch_ins)):
            ste+=1
            if batch_outs[x][0] == 1 and probs[0][0] > probs[0][1]:
                correct +=1
            if batch_outs[x][1] == 1 and probs[0][0] < probs[0][1]:
                correct+=1

    return np.exp(avg_loss/steps), correct/ste

def main():
    (train_inputs, train_labels, test_inputs, test_labels) = get_points_data()

    # TODO: initialize model
    model = TeamModel()

    # TODO: Set-up the training step
    for epoch in range(10):
        print("********** EPOCH "+str(epoch)+" *****************")
        train(model, train_inputs, train_labels)

    # TODO: Set up the testing steps
    loss, acc = test(model, test_inputs, test_labels)

    # Print out perplexity 
    print("Test loss: "+ str(loss))
    print("Test Acc: "+ str(acc))
    # BONUS: Try printing out various sentences with different start words and sample_n parameters 

    pass
    

if __name__ == '__main__':
    main()
