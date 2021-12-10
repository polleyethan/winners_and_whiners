import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.python.framework.ops import convert_to_tensor
from preprocess import get_win_data, get_points_data, get_win_data_with_players
from espn_page_parsing import getPassableStatArray


class TeamModel(tf.keras.Model):
    def __init__(self):
        super(TeamModel, self).__init__()

        self.batch_size = 100 #TODO 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.dense1 = tf.keras.layers.Dense(750, activation="relu")

        self.dense2 = tf.keras.layers.Dense(500, activation="relu")

        self.dense3 = tf.keras.layers.Dense(300, activation="relu")

        self.dense4 = tf.keras.layers.Dense(100, activation="relu")

        self.dense5 = tf.keras.layers.Dense(75, activation="relu")

        self.dense6 = tf.keras.layers.Dense(30, activation="relu")

        self.dense7 = tf.keras.layers.Dense(15, activation="relu")

        self.dense8 = tf.keras.layers.Dense(8, activation="relu")

        self.dense9 = tf.keras.layers.Dense(2)


    def call(self, inputs):
        dense_1 = self.dense1(inputs)

        dense_2 = self.dense2(dense_1)
        dense_3 = self.dense3(dense_2)

        dense_4 = self.dense4(dense_3)

        dense_5 = self.dense5(dense_4)

        dense_6 = self.dense6(dense_5)
        dense_7 = self.dense7(dense_6)
        dense_8 = self.dense8(dense_7)
        dense_9 = self.dense9(dense_8)

        softmax = tf.nn.softmax(dense_9)

        return softmax

    def loss(self, probs, labels):
        #for x in range(len(probs)):
            #print("******************************")
            #print("Home team predicted:"+str(probs[x]))
            #print("Home Team Win: "+str(labels[x]))

        return tf.reduce_mean(self.bce(labels, probs))


def train(model, train_inputs, train_labels):
    #train_inputs = np.reshape(train_inputs, (-1, model.window_size))
    #train_labels = np.reshape(train_labels, (-1, model.window_size))
    

    inputs = tf.convert_to_tensor(train_inputs)
    labels = tf.convert_to_tensor(train_labels)

    for i in range(0, len(train_inputs), model.batch_size):
        batch_ins = inputs[i:i+model.batch_size]
        batch_outs = labels[i:i+model.batch_size]

        indices = np.arange(0, len(batch_ins))
        indices_shuffled = tf.random.shuffle(indices)

        ins_shuffled = tf.gather(batch_ins, indices_shuffled)
        outs_shuffled = tf.gather(batch_outs, indices_shuffled)
        with tf.GradientTape() as tape:

            probs = model.call(ins_shuffled)

            loss = model.loss(probs, outs_shuffled)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
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
            if batch_outs[x][0] == 1 and probs[x][0] > probs[x][1]:
                correct +=1
            if batch_outs[x][1] == 1 and probs[x][0] < probs[x][1]:
                correct+=1

    return np.exp(avg_loss/steps), correct/ste

def main():
    (train_inputs, train_labels, test_inputs, test_labels) = get_win_data_with_players()

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


if __name__ == '__main__':
    main()
