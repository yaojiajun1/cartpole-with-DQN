# _*_ coding:UTF-8 _*_
# developer: yaoji
# Time: 12/9/20217:57 AM

import tensorflow
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# 测试环境

env = gym.make("CartPole-v1")
observation = env.reset()
'''for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()'''



# 使用Keras-rl建模
ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)
env._max_episode_steps = 1000
nb_actions = env.action_space.n
status = env.observation_space.shape


'''model = Sequential()
model.add(Flatten(input_shape=(1,) + status))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
#print(model.summary())'''

model = tensorflow.keras.models.load_model('complete_saved_model/')

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
#dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)


#weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
#dqn.load_weights(weights_filename)
dqn.test(env, nb_episodes=1, visualize=True)

'''
model = tensorflow.keras.models.load_model('complete_saved_model/')
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
'''

#dqn.test(env, nb_episodes=10, visualize=True)
