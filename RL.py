import numpy as np
import random
xxx=937162211
np.random.seed(xxx)
random.seed(xxx)

from tensorflow import set_random_seed, Session, ConfigProto
set_random_seed(xxx)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras import backend
from keras.initializers import lecun_uniform

sess = Session(config=ConfigProto(inter_op_parallelism_threads=1))
backend.set_session(sess)

from collections import deque

"""
Here are two values you can use to tune your Qnet
You may choose not to use them, but the training time
would be significantly longer.
Other than the inputs of each function, this is the only information
about the nature of the game itself that you can use.
"""
PIPEGAPSIZE  = 100
BIRDHEIGHT = 24

class QNet(object):

    def __init__(self):
        """
        Initialize neural net here.
        You may change the values.

        Args:
            num_inputs: Number of nodes in input layer
            num_hidden1: Number of nodes in the first hidden layer
            num_hidden2: Number of nodes in the second hidden layer
            num_output: Number of nodes in the output layer
            lr: learning rate
        """
        self.num_inputs = 2
        self.num_hidden1 = 10
        self.num_hidden2 = 10
        self.num_output = 2
        self.lr = 0.05 # .05 to 1

        self.gamma = .7
        # use this somehow: https://www.checkmarket.com/sample-size-calculator/
        self.batch_size = 1000
        self.sample_size = 400
        self.batch = {"state": deque(maxlen=self.batch_size), "prediction": deque(maxlen=self.batch_size)}

        self.score = 0
        self.minibatching_threshold_score = 4
        self.lr_threshold_score = 50

        self.build()

    def build(self):
        """
        Builds the neural network using keras, and stores the model in self.model.
        Uses shape parameters from init and the learning rate self.lr.
        You may change this, though what is given should be a good start.
        """
        model = Sequential()
        #model.add(Dense(self.num_hidden1, init='lecun_uniform', input_shape=(self.num_inputs,)))
        model.add(Dense(self.num_hidden1, init=lecun_uniform(seed=xxx), input_shape=(self.num_inputs,)))
        model.add(Activation('relu'))

        #model.add(Dense(self.num_hidden2, init='lecun_uniform'))
        model.add(Dense(self.num_hidden2, init=lecun_uniform(seed=xxx), input_shape=(self.num_inputs,)))
        model.add(Activation('relu'))

        #model.add(Dense(self.num_output, init='lecun_uniform'))
        model.add(Dense(self.num_output, init=lecun_uniform(seed=xxx)))
        model.add(Activation('linear'))

        rms = RMSprop(lr=self.lr)
        model.compile(loss='mse', optimizer=rms)
        self.model = model


    def flap(self, input_data):
        """
        Use the neural net as a Q function to act.
        Use self.model.predict to do the prediction.

        Args:
            input_data (Input object): contains information you may use about the 
            current state.

        Returns:
            (choice, prediction, debug_str): 
                choice (int) is 1 if bird flaps, 0 otherwise. Will be passed
                    into the update function below.
                prediction (array-like) is the raw output of your neural network,
                    returned by self.model.predict. Will be passed into the update function below.
                debug_str (str) will be printed on the bottom of the game
        """

        state = np.array([input_data.distX, input_data.distY])
        prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size=1)[0]
        choice = 1 if prediction[0] > prediction[1] else 0
        debug_str = str(prediction)
        return (choice, prediction, debug_str)


    def computeReward(self, oldDistX, oldDistY, distX, distY, last_choice, crash, scored):
        # reward model from OH with Ziyang
        reward, CRASHPENALTY, AWARD, PIPEGAPSIZE_QUARTER = 0, 1000, 200, PIPEGAPSIZE/4
        if crash:
            self.score = 0
            reward -= CRASHPENALTY
        elif last_choice == 1 and distY > PIPEGAPSIZE / 2 + PIPEGAPSIZE_QUARTER:
            reward -= CRASHPENALTY/2
        elif last_choice == 0 and distY < PIPEGAPSIZE_QUARTER:
            reward -= CRASHPENALTY/2
        elif scored:
            self.score += 1
            reward += AWARD
        else:
            reward += AWARD / 2
        return reward
        

    def update(self, last_input, last_choice, last_prediction, crash, scored, playerY, pipVelX):
        """
        Use Q-learning to update the neural net here
        Use self.model.fit to back propagate

        Args:
            last_input (Input object): contains information you may use about the
                input used by the most recent flap() 
            last_choice: the choice made by the most recent flap()
            last_prediction: the prediction made by the most recent flap()
            crash: boolean value whether the bird crashed
            scored: boolean value whether the bird scored
            playerY: y position of the bird, used for calculating new state
            pipVelX: velocity of pipe, used for calculating new state

        Returns:
            None
        """

        # This is how you calculate the new (x,y) distances
        new_distX = last_input.distX + pipVelX
        new_distY = last_input.pipeY - playerY # why is it like this? why doesn't take into account velocity, or if we've moved or not

        oldState = np.array([last_input.distX, last_input.distY])
        state = np.array([new_distX, new_distY])

        prediction = self.model.predict(state.reshape(1, self.num_inputs), batch_size = 1)

        reward = self.computeReward(last_input.distX, last_input.distY, new_distX, new_distY, 
                                    last_choice, crash, scored) # compute your reward

        # update old prediction from flap() with reward + gamma * np.max(prediction)
        updatedPrediction = last_prediction
        
        if last_choice == 1: # if it flaps
            updatedPrediction[0] = reward + self.gamma * np.max(prediction)
        else: 
            updatedPrediction[1] = reward + self.gamma * np.max(prediction)

        # record updated prediction and old state in your mini-batch
        self.batch["state"].append(oldState)
        self.batch["prediction"].append(updatedPrediction)

        # if batch size is large enough, back propagate
        if len(self.batch["state"]) >= self.batch_size:
            if self.score <= self.minibatching_threshold_score:
                randomChoices = np.random.choice(len(self.batch["state"]), self.sample_size)
                self.model.fit(np.asarray(self.batch["state"])[randomChoices,],
                               np.asarray(self.batch["prediction"])[randomChoices,],
                               batch_size=self.sample_size, epochs=1)

        # self.model.fit(old states, updated predictions, batch_size=size, epochs=1)


        #self.model.fit(oldState.reshape(1, 2), updatedPrediction.reshape(1, 2), batch_size=1, epochs=1)
              
class Input:
    def __init__(self, playerX, playerY, pipeX, pipeY,
                distX, distY):
        """
        playerX: x position of the bird
        playerY: y position of the bird
        pipeX: x position of the next pipe
        pipeY: y position of the next pipe
        distX: x distance between the bird and the next pipe
        distY: y distance between the bird and the next pipe
        """
        self.playerX = playerX
        self.playerY = playerY
        self.pipeX = pipeX
        self.pipeY = pipeY
        self.distX = distX
        self.distY = distY



