# by Albert Zeyer, www.az2000.de
# 2011-04-17

import random

class Program:
	def __init__(self):
		self.code = []
	def eval(self): pass

class MemoryBackend:
	def __init__(self):
		self.baseobjects = []
		self.dict = {} # int32,int32 -> int32
		self.objects = set()
	def get(self, subject, attrib):
		if (subject,attrib) in self.dict: return self.dict[(subject,attrib)]
		return 0
	def set(self, subject, attrib, value):
		self.dict[(subject,attrib)] = value
		if value not in self.objects: self.objects.add(value)
	def _unusedObject(self):
		while True:
			x = random.randint(1,2**32)
			if x not in self.objects: return x
	def create(self, subject, attrib):
		value = self._unusedObject()
		self.dict[(subject,attrib)] = value
		return value
	def execPrimitive(self, primitive, subject, attrib, value):
		if primitive == 0: value = self.get(subject, attrib)
		elif primitive == 1: self.set(subject, attrib, value)
		elif primitive == 2: value = self.create(subject, attrib)
		else: subject = attrib = value = 0
		return (subject,attrib,value)

