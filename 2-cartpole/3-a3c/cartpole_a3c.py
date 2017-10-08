import sys
import gym
import pylab
import numpy as np
import argparse
import tensorflow as tf
import threading
import time

from keras.layers import Dense, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import backend as K
from gym import wrappers
from threading import current_thread


# 멀티쓰레딩을 위한 글로벌 변수
global episode
global scores
global skip_train
skip_train = False
scores = []
episode = 0
EPISODES = 1000
env_name = 'CartPole-v1'


# 카트폴 예제에서의 액터-크리틱(A3C) 에이전트
class A3CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.01

        # 쓰레드의 갯수
        self.threads = 6

        # 정책신경망과 가치신경망 생성
        self.actor, self.critic = self.build_model()

        # 정책신경망과 가치신경망을 업데이트하는 함수 생성
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/cartpole_v1_a3c', self.sess.graph)

    # 쓰레드를 만들어 학습을 하는 함수
    def train(self):
        # 쓰레드 수만큼 Agent 클래스 생성
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.threads)]

        # 각 쓰레드 시작
        tn = 0
        for i, agent in enumerate(agents):
            print("Thread:{}".format(i))
            tn += 1
            time.sleep(1)
            agent.start()

        print("agent started")
        # (30초)에 한번씩 모델을 저장
        while True:
            time.sleep(30)
            self.save_model("./save_model/cartpole_a3c")
            if episode >= EPISODES:
                sys.exit()

    # 정책신경망과 가치신경망을 생성
    def build_model(self):
        input = Input(shape=(self.state_size, ))
        fc1 = Dense(12, activation='relu')(input)
        fc2 = Dense(24, activation='relu')(fc1)
        policy = Dense(self.action_size, activation='softmax')(fc2)
        value = Dense(1, activation='linear')(fc2)
        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])

        policy = self.actor.output

        # 정책 크로스 엔트로피 오류함수
        action_prob = K.sum(action * policy, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # 탐색을 지속적으로 하기 위한 엔트로피 오류
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages], [loss], updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))

        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def play(self):
        episode = 0
        env = gym.make(env_name)
        env = wrappers.Monitor(env, './cartpole_upload', force=True)
        self.actor.load_weights("./save_model/cartpole_a3c_actor.h5")
        scores = []

        while episode < 100:
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done:
                # env.render()
                action, _ = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                score += reward
                if done:
                    episode += 1
                    scores.append(score)
                    print("Test episode:", episode, "  score:", score)
        print("average scores:{}".format(np.mean(scores[:])))

    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy


# 액터러너 클래스(쓰레드)
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess, optimizer, discount_factor, summary_ops):
        threading.Thread.__init__(self)
        # A3CAgent 클래스에서 상속
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
             self.update_ops, self.summary_writer] = summary_ops

        # 지정된 타임스텝동안 샘플을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []

        # 로컬 모델 생성
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # 모델 업데이트 주기
        self.t_max = 5
        self.t = 0

    def run(self):
        global episode
        global scores
        global skip_train
        env = gym.make(env_name)
        print ("current thread:{}".format(current_thread().name))
        # env = wrappers.Monitor(env, './cartpole_upload', force=True)

        step = 0

        while episode < EPISODES:
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])

            if skip_train is True:
                self.update_local_model()

            while not done:
                step += 1
                self.t += 1

                action, policy = self.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                # 정책의 최대값
                self.avg_p_max += np.amax(self.actor.predict(state))

                score += reward
                reward = np.clip(reward, -1., 1.)
                state = next_state

                # 샘플을 저장
                self.append_sample(state, action, reward)

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if (self.t >= self.t_max or done) and skip_train is False:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0

                if done:
                    # change model update interval by score
                    self.t_max = max(min(20, int(score/10)), 5)

                    # 각 에피소드 당 학습 정보를 기록
                    scores.append(score)
                    episode += 1
                    print("episode:", episode, "  score:", score, "  step:",
                          step, "t_max", self.t_max)

                    # 이전 3개 에피소드의 점수 평균이 490보다 크면 학습 skip
                    if np.mean(scores[-min(2, len(scores)):]) > 490 and skip_train is False:
                        print ("set skip train:{}".format(episode))
                        skip_train = True
                        scores = scores[-2:]

                    if skip_train is True and len(scores) >= 10 and np.mean(scores[-10:]) < 450:
                        print ("set train:{}, scores length:{}".format(episode, len(scores)))
                        skip_train = False

                    stats = [score, self.avg_p_max / float(step),
                             step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction = np.zeros_like(rewards)
        running_add = 0

        if not done:
            running_add = self.critic.predict(self.states[-1])[0]

        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_prediction[t] = running_add
        return discounted_prediction


    # 정책신경망과 가치신경망을 업데이트
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.zeros((len(self.states), self.state_size))
        for i in range(len(self.states)):
            states[i] = self.states[i]

        values = self.critic.predict(states)[0]
        values = np.reshape(values, len(values))

        advantages = discounted_prediction - values

        self.optimizer[0]([states, self.actions, advantages])
        self.optimizer[1]([states, discounted_prediction])
        self.states, self.actions, self.rewards = [], [], []

    # 로컬신경망을 생성하는 함수
    def build_local_model(self):
        input = Input(shape=(self.state_size,))
        fc1 = Dense(12, activation='relu')(input)
        fc2 = Dense(24, activation='relu')(fc1)
        policy = Dense(self.action_size, activation='softmax')(fc2)
        value = Dense(1, activation='linear')(fc2)
        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        # 가치와 정책을 예측하는 함수를 만들어냄
        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic

    # 로컬신경망을 글로벌신경망으로 업데이트
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # 정책신경망의 출력을 받아서 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.local_actor.predict(state, batch_size=1).flatten()
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'train'], default='train')
    args = parser.parse_args()

    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    global_agent = A3CAgent(state_size=state_size, action_size=action_size)
    if args.task == 'train':
        global_agent.train()
    else:
        global_agent.play()
