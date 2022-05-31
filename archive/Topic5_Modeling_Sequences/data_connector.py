# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import _pickle as cPickle
import h5py


class DataConnector():

	# input here is individual text
	def __init__(self, filepath, filename, data=None):

		self.filepath = filepath
		self.filename = filename
		self.data = data
		self.read_file = []
		
	def read_pickle(self):

		f = open(os.path.join(self.filepath, self.filename), 'rb')
		self.read_file = cPickle.load(f)
		f.close()

		return self.read_file

	def save_pickle(self):

		f = open(os.path.join(self.filepath, self.filename), 'wb')
		cPickle.dump(self.data, f)
		print(" file saved to: %s"%(os.path.join(self.filepath, self.filename)))
		f.close()

	def read_numpys(self):

		self.read_file = np.load(os.path.join(self.filepath, self.filename))


	def save_numpys(self):

		np.save(os.path.join(self.filepath, self.filename), self.data)
		print(" file saved to: %s"%(os.path.join(self.filepath, self.filename)))

	def save_txt(self):

		with open(os.path.join(self.filepath, self.filename), "w") as text_file:
			print("{}".format(self.data), file=text_file)


	# saving file into hdf5 format
	# only works for array with same length
	
	def save_H5File(self):

		# h5Filename should be in format 'name-of-file.h5'
		h5Filename = self.filename

		# datasetName in String "" format
		datasetName = "data"

		with h5py.File(h5Filename, 'w') as hf:
			hf.create_dataset(datasetName,  data=self.data)


	# reading file in hdf5 format
	# only works for array with same length
	
	def read_H5File(self):

		# h5Filename should be in format 'name-of-file.h5'
		h5Filename = self.filename
		# datasetName in String "" format
		datasetName = "data"

		with h5py.File(h5Filename, 'r') as hf:
			data = hf[datasetName][:]

		self.read_file = data

	def read_pickles_all(self, result_path, filepaths):

		self.filepath = result_path

		all_files = []
		for file in self.read_pickles_generator(filepaths):
			all_files.extend(file)

		return all_files

	def read_pickles_doc(self, result_path, filepaths):

		self.filepath = result_path

		all_files = []
		for file in self.read_pickles_generator(filepaths):
			all_files.append(file)

		return all_files



	def read_pickles_generator(self, filepaths):

		for path in filepaths:

			#print("path:%s"%path)
			#print("filepath: %s"%(self.filepath))

			f = open(os.path.join(self.filepath, path), 'rb')
			self.read_file = cPickle.load(f)

			#print("Generated kps: %s"%(self.read_file))

			f.close()

			yield self.read_file

	# function to load pretrained embedding
	def load_embedding(self, indices_words, words_indices, embedding_dim):
		print('loading embeddings from "%s"' % self.filename, file=sys.stderr)
		vocab_size = len(indices_words)
		embedding = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
		seen = set()
		pretrained_words = set()
		with open(os.path.join(self.filepath, self.filename)) as fp:
			for line in fp:
				tokens = line.strip().split(' ')
				if len(tokens) == embedding_dim + 1:
					word = tokens[0]
					pretrained_words.add(word)
					if word in words_indices.keys():
						embedding[words_indices[word]] = [float(x) for x in tokens[1:]]
						seen.add(word)
						if len(seen) == vocab_size:
							break
		return embedding, pretrained_words

	def create_oov_embedding(self, pretrained_words, indices_words, words_indices, embedding_dim):

		print("creating list of OOV...")
		oov_indices_words = {}
		oov = set()
		for k, v in indices_words.items():
			if v not in list(pretrained_words):
				oov.add(v)

		oov_size = len(list(oov))
		print("oov_size:%s" %oov_size)
		sys.stdout.flush()

		print("creating empty matrix for OOV embedding...")

		oov_embedding = np.zeros((oov_size, embedding_dim), dtype=np.float32)

		return oov_embedding, list(oov)


