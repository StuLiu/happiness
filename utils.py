'''
--------------------------------------------------------
@File    :   utils.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2019/1/24 12:07     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
'''


import os
import numpy as np
import re

DATASET_DIR = os.path.join(os.path.abspath('.'), 'dataset')

def transfor_to_onehot(possibility_list):
	max_p, max_index = 0, 0
	for index in range(len(possibility_list)):
		if possibility_list[index] > max_p:
			max_p = possibility_list[index]
			max_index = index
	result = [0] * len(possibility_list)
	result[max_index] = 1
	return np.array(result, dtype='float32')

def __read_data(file_name):
	result = list()
	file_path = os.path.join(DATASET_DIR, file_name)
	with open(file_path, 'r', encoding='utf-8') as file:
		head_line = file.readline()     # read head line
		print(len(head_line.split(',')))
		line_list = file.readlines()
		for line in line_list:
			line = line.replace('\n','')    # remove \n
			line = re.sub(r'[^0-9\.\-,\/].*?', '', line)
			item_list = line.split(',')
			result.append(item_list)
	print(len(result))
	return result

def load_train_data_abbr():
	X, Y = list(), list()
	example_list = __read_data('happiness_train_abbr.csv')
	for example in example_list:
		del example[6]
		del example[0]
		while '' in example:
			example[int(example.index(''))] = '0'
		X.append(example[1:])
		y = [0, 0, 0, 0, 0]
		index = int(example[0]) - 1
		if index > 4: index = 4
		if index < 0: index = 0
		y[index] = 1
		Y.append(y)
	return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

def load_test_data_abbr():
	X = list()
	example_list = __read_data('happiness_test_abbr.csv')
	for example in example_list:
		del example[5]
		del example[0]
		while '' in example:
			example[int(example.index(''))] = '0'
		X.append(example)
	return np.array(X, dtype='float32')

def load_train_data_comp():
	X, Y = list(), list()
	example_list = __read_data('happiness_train_complete.csv')
	for example in example_list:
		del example[6]
		del example[0]
		while '' in example:
			example[int(example.index(''))] = '0'
		X.append(example[1:])
		y = [0, 0, 0, 0, 0]
		index = int(example[0]) - 1
		if index > 4: index = 4
		if index < 0: index = 0
		y[index] = 1
		Y.append(y)
	return np.array(X, dtype='float32'), np.array(Y, dtype='float32')

def load_test_data_comp():
	X = list()
	example_list = __read_data('happiness_test_complete.csv')
	for example in example_list:
		del example[5]
		del example[0]
		while '' in example:
			example[int(example.index(''))] = '0'
		X.append(example)
	return np.array(X, dtype='float32')

if __name__ == '__main__':
	x, y = load_train_data_comp()
	print(x[0], y[0])
	print(x.shape, y.shape)
	print(x.dtype, y.dtype)

	x= load_test_data_comp()
	print(x[0])
	print(x.shape)
	print(x.dtype)

	# str = '6100,4,2,13,36,67,2015/9/19 8:40,1,1969,1,1,1,3,,2,,3000,1,,40,0,0,0,0,0,0,0,0,0,拆迁分配，还没房产证,1,-2'
	# line = re.sub(r'[^0-9\.\-,\/].*?', '', str)
	# print(line)