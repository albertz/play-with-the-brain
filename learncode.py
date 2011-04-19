# by Albert Zeyer, www.az2000.de
# 2011-04-17

import random

def _highestBit(n):
	c = 0
	while n > 0:
		n /= 2
		c += 1
	return c

def _numFromBinaryVec(vec, indexStart, indexEnd):
	num = 0
	c = 1
	for i in xrange(indexStart,indexEnd):
		if vec[i] > 0.0: num += c
		c *= 2
	return num

def _numToBinaryVec(num, bits):
	vec = ()
	while num > 0 and len(vec) < bits:
		vec += (num % 2,)
		num /= 2
	if len(vec) < bits: vec += (0,) * (bits - len(vec))
	return vec

class Program:
	def __init__(self):
		self.code = []
	def eval(self): pass

ObjectDim = 16
PrimitiveCount = 4
PrimitiveDim = _highestBit(PrimitiveCount)
MemoryActivationDim = ObjectDim * 3

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
			x = random.randint(1,2**ObjectDim)
			if x not in self.objects: return x
	def create(self, subject, attrib):
		value = self.get(subject, attrib)
		if value: return value
		value = self._unusedObject()
		self.set(subject, attrib, value)
		return value
	def execPrimitive(self, primitive, subject, attrib, value):
		if primitive == 1: value = self.get(subject, attrib)
		elif primitive == 2: self.set(subject, attrib, value)
		elif primitive == 3: value = self.create(subject, attrib)
		else: subject = attrib = value = 0
		return (subject,attrib,value)

# unused atm
class MemoryBackend2:
	def __init__(self):
		self.dict = {} # int32 -> int32
	def get(self, ptr):
		if ptr in self.dict: return self.dict[ptr]
		return 0
	def set(self, ptr, value):
		self.dict[ptr] = value
	def execPrimitive(self, primitive, ptr, value):
		if primitive == 1: value = self.get(ptr)
		elif primitive == 2: self.set(ptr, value)
		else: ptr = value = 0
		return (ptr,value)


def netOutToAction(netOut):
	assert len(netOut) == PrimitiveDim + ObjectDim * 3
	primitive = _numFromBinaryVec(netOut, 0, PrimitiveDim-1)
	subject = _numFromBinaryVec(netOut, PrimitiveDim, PrimitiveDim+ObjectDim-1)
	attrib = _numFromBinaryVec(netOut, PrimitiveDim+ObjectDim, PrimitiveDim+ObjectDim*2-1)
	value = _numFromBinaryVec(netOut, PrimitiveDim+ObjectDim*2, PrimitiveDim+ObjectDim*3-1)
	return lambda mem: mem.execPrimitive(primitive, subject, attrib, value)

def memoryOutToNetIn(subject, attrib, value):
	vec = ()
	vec += _numToBinaryVec(subject, ObjectDim)
	vec += _numToBinaryVec(attrib, ObjectDim)
	vec += _numToBinaryVec(value, ObjectDim)
	return vec

class Run:
	def __init__(self):
		self.memory = MemoryBackend()
		self.program = Program()
		self.memoryActivation = (0,) * MemoryActivationDim
	def executeNetOut(self, netOut):
		action = netOutToAction(netOut)
		memoryOut = action(self.memory)
		self.memoryActivation = memoryOutToNetIn(*memoryOut)
		