from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

import math
import numpy as np
import pandas as pd
import tensorflow as tf

class ModelWrapper():
    def __init__(self, num_model=1):
        print("Selected model", num_model)
        self.num = num_model
        if num_model == 1:
            self.model = Model_1() 
        elif num_model == 2:
            self.model = Model_2()
        elif num_model == 3:
            self.model = Model_3()
        elif num_model == 4:
            self.model = Model_4()
        elif num_model == 5:
            self.model = Model_5()
        elif num_model == 6:
            self.model = Model_6()
        elif num_model == 7:
            self.model = Model_7()
        elif num_model == 8:
            self.model = Model_8()
        elif num_model == 9:
            self.model = Model_9()
        elif num_model == 10:
            self.model = Model_10()
        elif num_model == 11:
            self.model = Model_11()
        elif num_model == 12:
            self.model = Model_12()
        elif num_model == 13:
            self.model = Model_13()
        elif num_model == 14:
            self.model = Model_14()
        elif num_model == 15:
            self.model = Model_15()
        else:
            self.model = Model_3()
    def run(self, train_data, test_data, epochs=300):
        print("We are running Model", self.num, ": for", epochs, "num of episodes")
        weights_file = "run-model/model-{file}-cp.cpkt".format(file=int(self.num))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=weights_file) # save weights
        self.model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
        str_print = "Beginning training model {model_num}".format(model_num=self.num)
        print(str_print)
        history = self.model.fit(train_data, validation_data=test_data, callbacks=[cp_callback],
                epochs=epochs)
        #self.model.summary(print_fn=self.myprint)
    def myprint(self, s):
        #file_name = '../run-model/{file}summary.txt','w+'.format(file=self.num)
        with open('../run-model/{file}summary.txt','w+'.format(file=self.num),'w+') as f:
            print(s, file=f)
    def change_model(self, num_model):
        if num_model == 1:
            self.model = Model_1() 
        elif num_model == 2:
            self.model = Model_2()
        elif num_model == 3:
            self.model = Model_3()
        elif num_model == 4:
            self.model = Model_4()
        elif num_model == 5:
            self.model = Model_5()
        elif num_model == 6:
            self.model = Model_6()
        elif num_model == 7:
            self.model = Model_7()
        elif num_model == 8:
            self.model = Model_8()
        elif num_model == 9:
            self.model = Model_9()
        elif num_model == 10:
            self.model = Model_10()
        elif num_model == 11:
            self.model = Model_11()
        elif num_model == 12:
            self.model = Model_12()
        elif num_model == 13:
            self.model = Model_13()
        elif num_model == 14:
            self.model = Model_14()
        elif num_model == 15:
            self.model = Model_15()
        else:
            self.model = Model_3()
        return self.model 
        
class Model_1(Model):
    def __init__(self, player_features=57):
        self.player_feat = player_features
        super(Model_1, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
            layers.Dense(27, activation='relu'),
            layers.Dense(10, activation='relu')
        ])
        
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(17, activation='relu'),
            layers.Dense(5, activation='relu')
        ])
        self.predictor = tf.keras.Sequential([
            layers.LSTM(256, activation='softmax', return_sequences=True),
            layers.LSTM(256, activation='softmax', return_sequences=True),
            layers.LSTM(256, activation='softmax', return_sequences=False),
            layers.Dense(100, activation='relu'),
            layers.Dense(40, activation='relu'),
            layers.Dense(2, activation='softmax')         
        ])
    def call(self, x):
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = self.player_feat
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        return self.predictor(feat)

class Model_13(Model):
    def __init__(self, player_feat=57):
        super(Model_13, self).__init__() 
        self.player_feat = player_feat
        self.encoder_player = tf.keras.Sequential([
            layers.Dense(10, activation='relu')
        ])
        
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(5, activation='relu')
        ])
        self.predictor = tf.keras.Sequential([
            layers.LSTM(256, activation='softmax', return_sequences=True),
            layers.LSTM(256, activation='softmax', return_sequences=True),
            layers.LSTM(256, activation='softmax', return_sequences=False),
            layers.Dense(100, activation='relu'),
            layers.Dense(40, activation='relu'),
            layers.Dense(2, activation='softmax')         
        ])
    def call(self, x):
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        return self.predictor(feat)

class Model_14(Model):
    def __init__(self):
        super(Model_14, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
            layers.Dense(10, activation='relu')
        ])
        
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(5, activation='relu')
        ])
        self.predictor = tf.keras.Sequential([
            layers.LSTM(256, activation='softmax', return_sequences=False),
            layers.Dense(100, activation='relu'),
            layers.Dense(2, activation='softmax')         
        ])
    def call(self, x):
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        return self.predictor(feat)



class Model_2(Model):
    def __init__(self):
        super(Model_2, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
            layers.Dense(37, activation='relu'),
            layers.Dense(37, activation='relu'),
            layers.Dense(37, activation='relu'),
            layers.Dense(12, activation='sigmoid')
        ])
    
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(27, activation='relu'),
            layers.Dense(27, activation='relu'),
            layers.Dense(8, activation='sigmoid')
        ])
    
  

        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation='relu',),
         ])

        

        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
            layers.LSTM(45, activation='tanh', return_sequences=True),
            layers.Dense(45),
            layers.LSTM(35, activation='tanh', return_sequences=True),
            layers.Dense(35),
            layers.LSTM(25, activation='tanh', return_sequences=False),
            layers.Dense(20, activation='relu',),
            layers.Dense(15, activation='relu',),
            layers.Dense(8, activation='relu',),
            layers.Dense(2, activation='softmax',)
        ])
    


    def call(self, x):
    
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        conv = self.conv(feat)
        #conv = tf.pad(conv, (0,0,3))
        lstm = self.lstm(feat)
        ##val = lstm.shape[2] - conv.shape[2]
        print("val")
        #print("shape conv before", conv.shape, "lstm", lstm.shape)
        paddings = [[0, 0], [0, 0], [0,3]]
        conv = tf.pad(conv, paddings, 'CONSTANT', constant_values=0)
        #print("shape conv after", conv.shape, "lstm", lstm.shape)
        return self.predictor(
            tf.concat([conv, lstm], axis=1)
        )
        # return self.predictor(
        #     tf.concat([self.conv(feat), self.lstm(feat)], axis=1)
        # )

class Model_15(Model):
    def __init__(self):
        super(Model_15, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
            layers.Dense(37, activation='relu'),
            layers.Dense(12, activation='sigmoid')
        ])
    
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(27, activation='relu'),
            layers.Dense(8, activation='sigmoid')
        ])
    

        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
            layers.Dense(47, activation='relu',),
         ])

        

        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
            layers.LSTM(45, activation='tanh', return_sequences=True),
            layers.Dense(45),
            layers.LSTM(35, activation='tanh', return_sequences=True),
            layers.Dense(35),
            layers.LSTM(25, activation='tanh', return_sequences=False),
            layers.Dense(20, activation='relu',),
            layers.Dense(15, activation='relu',),
            layers.Dense(8, activation='relu',),
            layers.Dense(2, activation='softmax',)
        ])
    


    def call(self, x):
    
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        return self.predictor(
            tf.concat([self.conv(feat), self.lstm(feat)], axis=1)
        )
class Model_3(Model):
    def __init__(self):
        super(Model_3, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(10, activation='sigmoid')
        ])
        
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(17, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(5, activation='sigmoid')
        ])
        
        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation='relu',),
        ])

            

        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
        layers.LSTM(45, activation='tanh', return_sequences=True),
        layers.Dense(45),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(35, activation='tanh', return_sequences=True),
        layers.Dense(35),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(25, activation='tanh', return_sequences=False),
        layers.Dense(20, activation='relu',),
        layers.Dense(15, activation='relu',),
        layers.Dense(8, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])
    def call(self, x):
    
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        conv = self.conv(feat)
        #conv = tf.pad(conv, (0,0,3))
        lstm = self.lstm(feat)
        ###val = lstm.shape[2] - conv.shape[2]
        #print("shape conv before", conv.shape, "lstm", lstm.shape)
        paddings = [[0, 0], [0, 0], [0,3]]
        conv_shape = conv.shape.as_list()
        lstm_shape = lstm.shape.as_list()
        #print("pad", pad)
        #paddings = [[0, m - conv.get_shape().as_list()[i]] for (i,m) in enumerate(lstm.get_shape().as_list())]
        conv = tf.pad(conv, paddings, 'CONSTANT', constant_values=0)
        #print("shape conv after", conv.shape, "lstm", lstm.shape)
        return self.predictor(
            tf.concat([conv, lstm], axis=1)
        )

class Model_4(Model):
    def __init__(self):
        super(Model_4, self).__init__() 

        self.encoder_player = tf.keras.Sequential([
            layers.Dense(27, activation='relu'),
            layers.Dense(27, activation='relu'),
            layers.Dense(27, activation='relu'),
            layers.Dense(10, activation='sigmoid')
        ])
    
        self.encoder_team = tf.keras.Sequential([
                                                
            
            layers.Dense(17, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(5, activation='sigmoid')
        ])
      
        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation='relu',),
        ])

            
        self.predictor = tf.keras.Sequential([ 
        layers.LSTM(45, activation='tanh', return_sequences=True),
        layers.Dense(45),
        layers.LSTM(35, activation='tanh', return_sequences=True),
        layers.Dense(35),
        layers.LSTM(25, activation='tanh', return_sequences=False),
        layers.Dense(20, activation='relu',),
        layers.Dense(15, activation='relu',),
        layers.Dense(8, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])

    def call(self, x):
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        conv = self.conv(feat)
        #conv = tf.pad(conv, (0,0,3))
        lstm = self.lstm(feat)
        ##val = lstm.shape[2] - conv.shape[2]
        #print("shape conv before", conv.shape, "lstm", lstm.shape)
        paddings = [[0, 0], [0, 0], [0,3]]
        conv = tf.pad(conv, paddings, 'CONSTANT', constant_values=0)
        return self.predictor(
            tf.concat([conv, lstm], axis=1)
        )
class Model_5(Model):
    def __init__(self):
        super(Model_5, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(10, activation='sigmoid')
        ])
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(17, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(5, activation='sigmoid')
        ])
        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation='relu',),
        ])

            

        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
        layers.LSTM(45, activation='tanh', return_sequences=True),
        layers.Dense(45),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(35, activation='tanh', return_sequences=True),
        layers.Dense(35),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(25, activation='tanh', return_sequences=False),
        layers.Dense(20, activation='relu',),
        layers.Dense(15, activation='relu',),
        layers.Dense(8, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])
    def call(self, x):
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        conv = self.conv(feat)
        #conv = tf.pad(conv, (0,0,3))
        lstm = self.lstm(feat)
        ##val = lstm.shape[2] - conv.shape[2]
        #print("shape conv before", conv.shape, "lstm", lstm.shape)
        paddings = [[0, 0], [0, 0], [0,3]]
        #paddings = [[0, m - conv.numpy()[i]] for (i,m) in enumerate(lstm.numpy())]
        conv = tf.pad(conv, paddings, 'CONSTANT', constant_values=0)
        return self.predictor(
            tf.concat([conv, lstm], axis=1)
        )

class Model_6(Model):
    def __init__(self):
        super(Model_6, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(10, activation='sigmoid')
        ])
        
        self.encoder_team = tf.keras.Sequential([
                                                
            
            layers.Dense(17, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(5, activation='sigmoid')
        ])
        
        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
        layers.LSTM(45, activation='tanh', return_sequences=True),
        layers.Dense(45),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(35, activation='tanh', return_sequences=True),
        layers.Dense(35),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(25, activation='tanh', return_sequences=False),
        layers.Dense(20, activation='relu',),
        layers.Dense(15, activation='relu',),
        layers.Dense(8, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])
    


    def call(self, x):
    
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        return self.predictor(self.conv(feat))
    

class Model_7(Model):
    def __init__(self, player_features=57):
        super(Model_7, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(10, activation='sigmoid')
        ])
        
        self.encoder_team = tf.keras.Sequential([
                                            
            layers.Dense(17, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(5, activation='sigmoid')
        ])
        
        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
            layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
        layers.LSTM(45, activation='tanh', return_sequences=True),
        layers.Dense(45),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(35, activation='tanh', return_sequences=True),
        layers.Dense(35),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(25, activation='tanh', return_sequences=False),
        layers.Dense(20, activation='relu',),
        layers.Dense(15, activation='relu',),
        layers.Dense(8, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])
       


    def call(self, x):
    
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]
        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        return self.predictor(self.conv(feat))
class Model_8(Model):
    def __init__(self):
        super(Model_8, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(27, activation='relu'),
        layers.Dense(10, activation='sigmoid')
        ])
        
        self.encoder_team = tf.keras.Sequential([
                                                
            
            layers.Dense(17, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(5, activation='sigmoid')
        ])
        

        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(64, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation='relu',),
        ])

            

        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
        layers.LSTM(45, activation='tanh', return_sequences=True),
        layers.Dense(45),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(35, activation='tanh', return_sequences=True),
        layers.Dense(35),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(16, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(2,1) , padding='valid'),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(25, activation='tanh', return_sequences=False),
        layers.Dense(20, activation='relu',),
        layers.Dense(15, activation='relu',),
        layers.Dense(8, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])
        

    def call(self, x):
        
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        conv = self.conv(feat)
        #conv = tf.pad(conv, (0,0,3))
        lstm = self.lstm(feat)
        ##val = lstm.shape[2] - conv.shape[2]
        #print("shape conv before", conv.shape, "lstm", lstm.shape)
        paddings = [[0, 0], [0, 0], [0,3]]
        conv = tf.pad(conv, paddings, 'CONSTANT', constant_values=0)
        return self.predictor(
            tf.concat([conv, lstm], axis=1)
        )
class Model_9(Model):
    def __init__(self):
        super(Model_9, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(10, activation='sigmoid')
        ])
        
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(5, activation='sigmoid')
        ])
        self.predictor = tf.keras.Sequential([ 
        layers.Dense(30),
        layers.LSTM(30, activation='tanh', return_sequences=True),
        layers.LSTM(30, activation='tanh', return_sequences=False),
        layers.Dense(20, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])
        


    def call(self, x):
        
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
    
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        
        return self.predictor(feat)
    
class Model_10(Model):
    def __init__(self):
        super(Model_10, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(10, activation='sigmoid')
        ])
        
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(5, activation='sigmoid')
        ])
        
        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
        ])

        
        self.predictor = tf.keras.Sequential([ # maybe interleave pool/cnn between lstm to reduce time steps???
        layers.Dense(30),
        layers.LSTM(30, activation='tanh', return_sequences=True),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(20, activation='tanh', return_sequences=False),
        layers.Dense(10, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])
        


    def call(self, x):
        num_seq = tf.shape(x)[1]#695 #x.shape[1]
        num_games = tf.shape(x)[0] #x.shape[0]

        team_length = 35
        player_length = 57
        
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
 
        conv = self.conv(feat)
        #conv = tf.pad(conv, (0,0,3))
        lstm = self.lstm(feat)
        ##val = lstm.shape[2] - conv.shape[2]
        #print("shape conv before", conv.shape, "lstm", lstm.shape)
        paddings = [[0, 0], [0, 0], [0,3]]
        conv = tf.pad(conv, paddings, 'CONSTANT', constant_values=0)
        return self.predictor(
            tf.concat([conv, lstm], axis=1)
        )
class Model_11(Model):
    def __init__(self):
        super(Model_11, self).__init__() 
        self.encoder_player = tf.keras.Sequential([
        layers.Dense(10, activation='sigmoid')
        ])
        
        self.encoder_team = tf.keras.Sequential([
            layers.Dense(5, activation='sigmoid')
        ])
        
        self.conv = tf.keras.Sequential([
            layers.Lambda(lambda x: tf.expand_dims(x, 3)),
            layers.Conv2D(32, (3, 1), activation='relu', padding='same'),
            layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
            layers.Lambda(lambda x: x[:,:,:,0]),
        ])

        self.lstm = tf.keras.Sequential([
            layers.LSTM(47, activation='tanh', return_sequences=True),
            layers.Dense(47, activation="relu"),
        ])

        self.predictor = tf.keras.Sequential([ 
        layers.Dense(30),
        layers.LSTM(30, activation='tanh', return_sequences=True),
        layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        layers.Conv2D(1, (3, 1), activation='relu', padding='same'),
        layers.MaxPool2D( (2,1), strides=(1,1) , padding='same'),
        layers.Lambda(lambda x: x[:,:,:,0]),
        layers.LSTM(20, activation='tanh', return_sequences=False),
        layers.Dense(10, activation='relu',),
        layers.Dense(2, activation='softmax',)
        ])

    def call(self, x):
        num_seq = tf.shape(x)[1]
        num_games = tf.shape(x)[0]
        team_length = 35
        player_length = 57
        shooter_start = team_length
        shooter_end = team_length+player_length
        
        assister_start=team_length+player_length
        assister_end = team_length+player_length*2
        
        blocker_start=team_length+player_length*2
        blocker_end=team_length+player_length*3
        
        team_info = x[:, :, :team_length]
        team_info = tf.reshape(team_info, (-1, team_length))
        
        team_embed = self.encoder_team(team_info)
        
        team_embed = tf.reshape(team_embed, (num_games, num_seq, -1))
        
        
        shooter_info = x[:, :, shooter_start:shooter_end]
        shooter_info = tf.reshape(shooter_info, (-1,player_length))
        
        assister_info = x[:, :, assister_start:assister_end]
        assister_info = tf.reshape(assister_info, (-1,player_length))
        
        blocker_info = x[:, :, blocker_start:blocker_end]
        blocker_info = tf.reshape(blocker_info, (-1,player_length))
        
        player_info = tf.concat([shooter_info, assister_info, blocker_info], axis=1)
        player_info = tf.reshape(player_info, (-1, player_length))
        
        player_embed = self.encoder_player(player_info)
        player_embed = tf.reshape(player_embed, (num_games, num_seq, -1))
        
        shooter_embed = player_embed[:, :, :player_length]
        assister_embed = player_embed[:, :, player_length:2*player_length]
        blocker_embed = player_embed[:, :, 2*player_length:]

        feat = tf.concat([team_embed, shooter_embed, assister_embed, blocker_embed, x[:,:,team_length+player_length*3:]], axis=2)
        conv = self.conv(feat)
        #conv = tf.pad(conv, (0,0,3))
        lstm = self.lstm(feat)
        ##val = lstm.shape[2] - conv.shape[2]
        #print("shape conv before", conv.shape, "lstm", lstm.shape)
        paddings = [[0, 0], [0, 0], [0,3]]
        conv = tf.pad(conv, paddings, 'CONSTANT', constant_values=0)
        return self.predictor(
            tf.concat([conv, lstm], axis=1)
        )


class Model_12(Model):
    """
    This model takes a Dense --> LSTM --> Dense
    """
    def __init__(self):
        super(Model_12, self).__init__() 
        self.predictor = tf.keras.Sequential([ 
        layers.Dense(128, activation='sigmoid'),
        layers.LSTM(128, activation='sigmoid', return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax',)
        ])

    def call(self, x):
        return self.predictor(x)

def shape_array(self, a, b, axis=0):
        shape = np.shape(a)
        padded_array_shape = np.shape(b)
        padded_array = np.zeros(padded_array_shape)
        padded_array[:shape[0], :shape[1]] = a 
        #print("shape of pad", padded_array.shape)
        return padded_array

def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)