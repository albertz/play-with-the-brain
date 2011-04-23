# by Albert Zeyer, www.az2000.de
# 2011-04-17

import random
from itertools import *

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
	class Node:
		class Action:
			def execute(self, memory, contextSubject): pass
			@classmethod
			def Random(cls, rndContext):
				r = random.random()
				if r <= 0.2: return Action()
				elif r <= 0.7: return PrimitiveAction.Random(rndContext)
				elif len(rndContext.progPool) == 0: return Action()
				return CallAction.Random(rndContext)
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
			@staticmethod
			def RandomObject(): return MemoryBackend.RandomConstInt()
			@classmethod
			def Random(cls, rndContext):
				action = cls()
				action.primitive = random.randint(1,4)
				action.subjectIsLocal = random.random() <= 0.6
				action.attribIsLocal = random.random() <= 0.6
				action.valueIsLocal = random.random() <= 0.6
				action.subject = cls.RandomObject()
				action.attrib = cls.RandomObject()
				action.value = cls.RandomObject()
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
			def Random(cls, rndContext):
				action = cls()
				action.prog = random.choice(rndContext.progPool)
				action.context = random.randint(0, rndContext.ProgRandomMaxLocalVars)
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
		def Random(cls, rndContext):
			node = cls()
			node.action = Action.Random(rndContext)
			return node
		
	def __init__(self):
		self.startnode = None

	def __del__(self):
		if self.startnode:
			self.startnode.uninit()
			self.startnode = None

	class RandomContext:
		ProgRandomMaxLocalVars = 10
		ProgRandomMaxGlobalConsts = 100
		def __init__(self, **kwargs):
			self.progPool = []
			for key,value in kwargs:
				setattr(self, key, value)
		def generate(self, N=10):
			progs = map(lambda _: Program.Random(self), [None] * N)
			self.progPool += progs
			return progs
		
	def randomize(self, rndContext):
		N = random.randint(1, 10)
		nodes = map(lambda _: Node.Random(rndContext), [None] * N)
		for _ in xrange(random.randint(1, 10)):
			i = random.randint(0, N-1)
			j = random.randint(0, N-1)
			checkAttrib = random.randint(0, cls.ProgRandomMaxLocalVars)
			nodes[i].edges.append((checkAttrib,nodes[j]))
		for i in xrange(N-1):
			nodes[i].edges.append((None,nodes[i+1]))
		self.startnode = nodes[0]

	@classmethod
	def Random(cls, rndContext):
		prog = cls()
		prog.randomize(rndContext)
		return prog

	@classmethod
	def RandomPool(cls, N=50):
		rndContext = cls.RandomContext()
		rndContext.progPool = map(lambda _: cls(), [None] * N)
		for i in xrange(N):
			rndContext.progPool[i].randomize(rndContext)
		return rndContext.progPool
	
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

	def allNodes(self):pass

class MemoryBackend:
	ConstZero = 0
	ConstAttribIs = 2**4 + 1
	ConstAttribInSet = 2**4 + 2
	ConstAttribName = 2**4 + 5
	ConstAttribValue = 2**4 + 10
	ConstAttribNext = 2*+4 + 15
	ConstAttribLen = 2**4 + 100
	ConstValueType = 2*+15 + 1
	ConstValueStopMark = 2**15 + 10
	ConstValueDigit0 = 2**15 + 100
	ConstValueDigit1 = 2**15 + 101

	@classmethod
	def IterConsts(cls):
		for key in dir(cls):
			if key.startswith("Const") and type(getattr(cls, key)) is int:
				yield key, getattr(cls, key)
	@classmethod
	def IterConstsInt(cls):
		return imap(lambda (_,x): x, cls.IterConsts())
	@classmethod
	def RandomConstInt(cls):
		return random.choice(list(cls.IterConstsInt()))
	
	def __init__(self):
		self.baseobjects = list(self.IterConstsInt())
		self.dict = {} # int32,int32 -> int32
		self.objects = set(self.baseobjects)
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

def netOutToAction(netOut):
	primitive, subject, attrib, value = netOutToPrimitive(netOut)
	return lambda mem: mem.execPrimitive(primitive, subject, attrib, value)

def memoryOutToNetIn(subject, attrib, value):
	vec = ()
	vec += _numToBinaryVec(subject, ObjectDim)
	vec += _numToBinaryVec(attrib, ObjectDim)
	vec += _numToBinaryVec(value, ObjectDim)
	return vec

from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer, SoftmaxLayer

# a neural interface has some input and some output
# which can be used to communicate with a neural network
class NeuralInterface:
	class IO(LinearLayer):
		dim = 0
		def __init__(self, parent, **kwargs):
			LinearLayer.__init__(self, self.dim, **kwargs)
			self.parent = parent
	Input = IO
	Output = IO
	def __init__(self):
		self.input = self.Input(self)
		self.output = self.Output(self)
	
class MemoryNeuralInterface(NeuralInterface):
	class Input(NeuralInterface.Input):
		dim = ActionDim
		def _forwardImplementation(self, inbuf, outbuf):
			super(Input, self)._forwardImplementation(inbuf, outbuf)
			action = netOutToAction(inbuf)
			memoryOut = action(self.parent.memory)
			self.parent.output.activate(memoryOutToNetIn(*memoryOut))
	class Output(NeuralInterface.Output):
		dim = MemoryActivationDim
	def __init__(self, memory):
		NeuralInterface.__init__(self)
		self.memory = memory

class ProgNeuralInterface(NeuralInterface):
	class Input(NeuralInterface.Input):
		dim = 10
		def _forwardImplementation(self, inbuf, outbuf):
			super(Input, self)._forwardImplementation(inbuf, outbuf)
			
	class Output(NeuralInterface.Output):
		dim = 10
	def __init__(self, progPool):
		NeuralInterface.__init__(self)
		self.progPool = dict(map(lambda prog: (id(prog), prog), progPool))
		self.nodes = {}
		for i,prog in self.progPool.iteritems():
			pass

class LearnCodeTask:
	Prog = None
	input = [Prog]
	interfaces = [MemoryNeuralInterface, ProgNeuralInterface]
	output = None # ....
	
class Run:
	def __init__(self):
		self.memory = MemoryBackend()
		self.progPool = []
		self.memoryActivation = (0,) * MemoryActivationDim
	def executeNetOut(self, netOut):
		action = netOutToAction(netOut)
		memoryOut = action(self.memory)
		self.memoryActivation = memoryOutToNetIn(*memoryOut)
	def learnProg(self, prog):
		# show how to iterate through whole prog
		# train to do the same steps in NN
		# repeat and train to do the same without iterating through the prog
		pass
		
