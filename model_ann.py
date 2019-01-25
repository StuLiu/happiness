'''
--------------------------------------------------------
@File    :   model_ann.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2019/1/24 12:53     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
'''

import tensorflow as tf
import numpy as np
import os
from utils import *

class ANN(object):

	def __init__(self, input_size, output_size, epoch=100, batch_size=128, learn_rate=1e-6, model_name='ANN'):
		self.__model_name = model_name
		self.__input_size = input_size
		self.__output_size = output_size
		self.__epoch = epoch
		self.__batch_size = batch_size
		self.__learn_rate = learn_rate
		self.__build_model()
		self.__create_checkpoint_dir()

	def train(self, train_X, train_Y):
		print(train_X.dtype, train_Y.dtype)
		print(train_X.shape, train_Y.shape)
		model_saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(self.__epoch):
				loss_total = 0
				for j in range(0, int(train_X.shape[0] / self.__batch_size)):
					x_input = train_X[j * self.__batch_size: (j + 1) * self.__batch_size]
					y_input = train_Y[j * self.__batch_size: (j + 1) * self.__batch_size]
					loss, _ = sess.run([self.__cross_entropy, self.__train_step],
					                   feed_dict={self.__x_input: x_input, self.__y_input: y_input})
					loss_total += loss
				print("step{}, training loss {}".format(i, loss_total))
				model_saver.save(sess, os.path.join(self.__checkpoint_dir, '{}.model'.format(self.__model_name)), global_step=i)
				np.random.seed(i)
				np.random.shuffle(train_X)
				np.random.seed(i)
				np.random.shuffle(train_Y)

	def test(self, test_X, test_Y):
		print(test_X.shape, test_Y.shape)
		print(test_X.dtype, test_Y.dtype)
		with tf.Session() as sess:
			if self.__load_model(sess):
				print("load model successfully!")
			else:
				print("load model failed!")
			currect_count = 0
			s = 0
			for j in range(0, int(test_X.shape[0] / self.__batch_size)):
				x_input = test_X[j * self.__batch_size: (j + 1) * self.__batch_size]
				y_label = test_Y[j * self.__batch_size: (j + 1) * self.__batch_size]
				y_output = sess.run(self.__y_output, feed_dict={self.__x_input : x_input})
				for i in range(self.__batch_size):
					if np.argmax(y_output[i]) == np.argmax(y_label[i]):
						currect_count += 1
					s += 1
			x_input = test_X[int(test_X.shape[0] / self.__batch_size) * self.__batch_size : ]
			y_label = test_Y[int(test_X.shape[0] / self.__batch_size) * self.__batch_size : ]
			y_output = sess.run(self.__y_output, feed_dict={self.__x_input : x_input})
			for i in range(len(y_output)):
				if np.argmax(y_output[i]) == np.argmax(y_label[i]):
					currect_count += 1
			s += len(test_X) - int(test_X.shape[0] / self.__batch_size) * self.__batch_size
			print("accuracy:{}----{}".format(float(currect_count)/len(test_X), s / len(test_X)))

	def predict(self, X_input):
		lines = ['id,happiness\n']
		with tf.Session() as sess:
			if self.__load_model(sess):
				print("load model successfully!")
			else:
				print("load model failed!")
			for i in range(len(X_input)):
				line = str(8001 + i) + ','
				x_input = X_input[i].reshape((1,self.__input_size))
				y_output = sess.run(self.__y_output, feed_dict={self.__x_input: x_input})
				y_output = y_output.reshape((self.__output_size,))
				line += str(np.argmax(y_output) + 1) + '\n'
				lines.append(line)
		with open('./happiness_submit.csv', 'w', encoding='utf-8') as file:
			file.writelines(lines)

	def __build_model(self):
		"""定义网络结构、损失函数和参数调整算法"""
		self.__x_input = tf.placeholder(tf.float32, shape=(None, self.__input_size))
		self.__y_input = tf.placeholder(tf.float32, shape=(None, self.__output_size))

		self.__w1 = tf.Variable(tf.random_normal([self.__input_size, 64], stddev=0.01))
		self.__b1 = tf.Variable(tf.random_normal([64], stddev=0.01))
		self.__oper_1 = tf.nn.relu(tf.matmul(self.__x_input, self.__w1) + self.__b1)

		self.__w2 = tf.Variable(tf.random_normal([64, 16], stddev=0.01))
		self.__b2 = tf.Variable(tf.random_normal([16], stddev=0.01))
		self.__oper_2 = tf.nn.relu(tf.matmul(self.__oper_1, self.__w2) + self.__b2)

		self.__w3 = tf.Variable(tf.random_normal([16, self.__output_size], stddev=0.01))
		self.__b3 = tf.Variable(tf.random_normal([self.__output_size], stddev=0.01))
		self.__y_output = tf.nn.softmax(tf.matmul(self.__oper_2, self.__w3) + self.__b3)

		# tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
		# 小于min的让它等于min，大于max的元素的值等于max, 这里防止计算log(0)
		self.__cross_entropy = tf.reduce_mean(
			-tf.reduce_sum(
				self.__y_input * tf.log(tf.clip_by_value(self.__y_output, 1e-10, 1.0)), reduction_indices=[1]
			)
		)
		self.__train_step = tf.train.AdamOptimizer(self.__learn_rate).minimize(self.__cross_entropy)

	def __create_checkpoint_dir(self):
		self.__checkpoint_dir = os.path.join(os.path.join(os.path.curdir, 'checkpoints'), self.__model_name)
		if not os.path.exists(self.__checkpoint_dir):
			os.makedirs(self.__checkpoint_dir)

	def __load_model(self, sess):
		ckpt = tf.train.get_checkpoint_state(self.__checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			tf.train.Saver().restore(sess, os.path.join(self.__checkpoint_dir, ckpt_name))
			return True
		else:
			return False

if __name__ == '__main__':

	a = ANN(input_size=137, output_size=5, epoch=200, batch_size=64, learn_rate=1e-6, model_name='ANN')

	train_X, train_Y = load_train_data_comp()
	a.train(train_X, train_Y)

	print('train set acc:')
	a.test(train_X, train_Y)

	X_input = load_test_data_comp()
	a.predict(X_input)



