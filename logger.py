import _pickle

import datetime
from time import time

import sys
import os

class Logger():
	def __init__(self, base_dir=None, tag=None):
		self.store = dict()
		self.t0 = time()
		self.script = sys.argv[0]
		self.date = datetime.datetime.fromtimestamp(int(time())).strftime('%Y-%m-%d_%H-%M-%S')
		self.tag = tag

		if base_dir is None:
			base_dir = os.path.join(os.getcwd(), 'logs')

		if not os.path.exists(base_dir):
			os.makedirs(base_dir)

		if tag is None:
			self.filename = os.path.join(base_dir, '{}_{}'.format(self.script, self.date))
		else:
			self.filename = os.path.join(base_dir, tag)


	def add(self, name, value, step):
		
		if name not in self.store.keys():
			self.store[name] = {'values':[], 'steps':[], 'timestamps':[]}

		self.store[name]['values'].append(value)
		self.store[name]['steps'].append(step)
		self.store[name]['timestamps'].append(time() - self.t0)


	def load(self, filename):
		self.filename = filename
		log = _pickle.load(open(self.filename, 'rb'))
		
		self.store = log['store']
		self.script = log['script']
		self.date = log['date']
		self.tag = log['tag']


	def save(self):
		log = {'script': self.script, 'date': self.date, 'tag': self.tag, 'store': self.store}
		
		with open(self.filename, 'wb') as f:
			_pickle.dump(log, f)


	def close(self):
		log = {'script': self.script, 'date': self.date, 'tag': self.tag, 'store': self.store}
		
		with open(self.filename, 'wb') as f:
			_pickle.dump(log, f)