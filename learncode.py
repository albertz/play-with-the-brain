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

ObjectDim = 16
PrimitiveCount = 4
PrimitiveDim = _highestBit(PrimitiveCount)
MemoryActivationDim = ObjectDim * 3
ActionDim = PrimitiveDim + ObjectDim * 3


def netOutToPrimitive(netOut):
	assert len(netOut) == ActionDim
	primitive = _numFromBinaryVec(netOut, 0, PrimitiveDim-1)
	subject = _numFromBinaryVec(netOut, PrimitiveDim, PrimitiveDim+ObjectDim-1)
	attrib = _numFromBinaryVec(netOut, PrimitiveDim+ObjectDim, PrimitiveDim+ObjectDim*2-1)
	value = _numFromBinaryVec(netOut, PrimitiveDim+ObjectDim*2, PrimitiveDim+ObjectDim*3-1)
	return (primitive,subject,attrib,value)

class Program:
	ProgRandomMaxLocalVars = 10
	ProgRandomMaxGlobalConsts = 100

	class Node:
		class Action:
			def execute(self, memory, contextSubject): pass
			@classmethod
			def Random(cls):
				r = random.random()
				if r <= 0.2: return Action()
				elif r <= 0.7: return PrimitiveAction.Random()
				return CallAction.Random()
		class PrimitiveAction(Action):
			def __init__(self):
				self.primitive = 0
				self.subject = 0
				self.attrib = 0
				self.value = 0
				self.target = 0
				self.subjectIsLocal = False
				self.attribIsLocal = False
				self.valueIsLocal = False
			@classmethod
			def Random(cls):
				action = cls()
				action.primitive = random.randint(1,4)
				action.subjectIsLocal = random.random() <= 0.6
				action.attribIsLocal = random.random() <= 0.6
				action.valueIsLocal = random.random() <= 0.6
				action.subject = random.randint(0, action.subjectIsLocal and ProgRandomMaxLocalVars or ProgRandomMaxGlobalConsts)
				action.attrib = random.randint(0, action.attribIsLocal and ProgRandomMaxLocalVars or ProgRandomMaxGlobalConsts)
				action.value = random.randint(0, action.valueIsLocal and ProgRandomMaxLocalVars or ProgRandomMaxGlobalConsts)
				return action
			def execute(self, memory, contextSubject):
				subject = self.subject
				attrib = self.attrib
				value = self.value
				if self.subjectIsLocal: subject = memory.get(contextSubject, subject)
				if self.attribIsLocal: attrib = memory.get(contextSubject, attrib)
				if self.valueIsLocal: value = memory.get(contextSubject, value)
				_,_,value = memory.execPrimitive(self.primitive, subject, attrib, value)
				memory.set(contextSubject,self.target,value)
		class CallAction(Action):
			def __init__(self):
				self.prog = Program()
				self.context = 0
			@classmethod
			def Random(cls):
				action = cls()
				action.prog = Program.Random()
				action.context = random.randint(0, ProgRandomMaxLocalVars)
				return action
			def execute(self, memory, contextSubject):
				newContextSubject = memory.get(contextSubject, self.context)
				self.prog.execute(memory, newContextSubject)
			
		def __init__(self):
			self.edges = [] # (check , node)
			# whereby check is attrib and it checks
			# if attrib == 0 or (subject,attrib) != 0
			# whereby subject is current context
			self.action = Action()

		def uninit(self):
			if getattr(self, "uniniting", False): return
			self.uniniting = True
			for _,node in self.edges: node.uninit()
			self.edges = []
			self.action = None

		@classmethod
		def Random(cls):
			node = cls()
			node.action = Action.Random()
			return node
		
	def __init__(self):
		self.startnode = None

	def __del__(self):
		if self.startnode:
			self.startnode.uninit()
			self.startnode = None

	@classmethod
	def Random(cls):
		prog = cls()
		N = random.randint(1, 10)
		nodes = map(lambda _: Node.Random(), [None] * N)
		for _ in xrange(random.randint(1, 10)):
			i = random.randint(0, N-1)
			j = random.randint(0, N-1)
			checkAttrib = random.randint(0, cls.ProgRandomMaxLocalVars)
			nodes[i].edges.append((checkAttrib,nodes[j]))
		for i in xrange(N-1):
			nodes[i].edges.append((None,nodes[i+1]))
		prog.startnode = nodes[0]
		return prog

	def execute(self, memory, contextSubject):
		node = self.startnode
		while node is not None:
			node.action.execute(memory, contextSubject)
			nextnode = None
			for edgecheck, edgenode in node.edges:
				if edgecheck is None or memory.get(contextSubject, edgecheck):
					nextnode = edgenode
					break
			node = nextnode


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
	primitive, subject, attrib, value = netOutToPrimitive(netOut)
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
		