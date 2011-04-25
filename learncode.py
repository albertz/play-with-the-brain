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

	def allNodes(self):
		nodeSet = set()
		nodeList = []
		def addNode(node):
			if node is None: return
			if node not in nodeSet:
				nodeSet.add(node)
				nodeList.append(node)
		addNode(self.startnode)
		i = 0
		while i < len(nodeList):
			for check, node in sorted(nodeList[i].edges):
				addNode(node)
			i += 1
		return nodeList
	
class MemoryBackend:
	ConstZero = 0
	ConstAttribIs = 2**4 + 1
	ConstAttribInSet = 2**4 + 2
	ConstAttribSubSet = 2**4 + 3
	ConstAttribValue = 2**4 + 10
	ConstAttribResult = 2**4 + 15
	ConstAttribContext = 2**4 + 20
	ConstAttribNext = 2*+4 + 50
	ConstAttribLast = 2**4 + 55
	ConstAttribParent = 2**4 + 60
	ConstAttribChild = 2**4 + 62
	ConstAttribLen = 2**4 + 100
	ConstAttribName = 2**4 + 120
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


# General NN IO. slots:
# obj <->
#   NN:
#     id(obj)
#     id(left obj) or 0
#     id(right obj) or 0
#     additionalInfo(obj)

class GenericNeuralInterface(NeuralInterface):
	IdVecLen = 32
	@classmethod
	def idToVec(cls, id):
		if id is None: id = 0
		v = _numToBinaryVec(id, cls.IdVecLen)
		v = map(float, v)
		return v
	@classmethod
	def vecToId(cls, v):
		id = _numFromBinaryVec(v)
		return id
	@classmethod
	def objToVec(cls, obj):
		if obj is None: return idToVec(None)
		else: return idToVec(id(obj))
	
	class Input(NeuralInterface.Input):
		def __init__(self, parent):
			self.dim = parent.inputVecLen()
			super(Input, self).__init__(parent)
		def _forwardImplementation(self, inbuf, outbuf):
			super(Input, self)._forwardImplementation(inbuf, outbuf)
			levelInputs = [None] * self.parent.levelCount
			vecOffset = 0
			for level in xrange(0, self.parent.levelCount - 1):
				newVecOffset = vecOffset + self.parent.inputVecLenOfLevel(level)
				levelInputs[level] = self.parent.vecToId(inbuf[vecOffset:newVecOffset])
				vecOffset = newVecOffset
			for level in xrange(0, self.parent.levelCount - 1):
				if not self.parent.selectObjById(level, levelInputs[level]):
					# it means the objId is invalid
					# just ignore the rest
					break
	# Output is activated through updateLevelOutput
	class Output(NeuralInterface.Output):
		def __init__(self, parent):
			self.dim = parent.outputVecLen()
			super(Output, self).__init__(parent)
	class LevelRepr:
		objList = []
		objDict = {} # id(obj) -> objList index
		curObj = None
		curObjIndex = 0
		def reset(self, newObjList):
			self.objList = newObjList
			self.objDict = dict(map(lambda (index,obj): (id(obj),index), enumerate(newObjList)))
			self.curObj = newObjList and newObjList[0] or None
			self.curObjIndex = 0
		def asOutputVec(self, objToVec, additionalInfoFunc):
			leftObj = (self.curObjIndex > 0) and self.objList[self.curObjIndex-1] or None
			rightObj = (self.curObjIndex+1 < len(self.objList)) and self.objList[self.curObjIndex+1] or None
			leftObjVec = objToVec(leftObj)
			curObjVec = objToVec(self.curObj)
			rightObjVec = objToVec(rightObj)
			additionalInfo = additionalInfoFunc(self.curObj)
			return leftObjVec + curObjVec + rightObjVec + additionalInfo
	def __init__(self, topLevelList, childsFuncs, additionalInfoFuncs):
		self.topLevelList = topLevelList
		self.childsFuncs = childsFuncs # list(obj -> list(obj))
		self.levelCount = len(childsFuncs) + 1
		additionalInfoFuncs = additionalInfoFuncs or ([None] * self.levelCount)
		assert len(additionalInfoFuncs) == self.levelCount
		additionalInfoFuncs = map(lambda f: (f or (lambda _:())), additionalInfoFuncs)
		self.additionalInfoFuncs = additionalInfoFuncs
		self.levels = map(lambda _: LevelRepr(), [None] * self.levelCount)
		super(GenericNeuralInterface, self).__init__()
		self.resetLevel(0)

	def inputVecLenOfLevel(self, level): return self.IdVecLen
	def inputVecLen(self): return sum(map(self.inputVecLenOfLevel, range(0,self.levelCount-1)))
	def outputVecLenOfLevel(self, level): return 3 * self.IdVecLen + len(self.additionalInfoFuncs[level](None))
	def outputVecLen(self): return sum(map(self.outputVecLenOfLevel, range(0,self.levelCount-1)))

	def updateLevelOutput(self, level):
		outVec = self.levels[level].asOutputVec(self.objToVec, self.additionalInfoFuncs[level])
		self.output.activate(outVec)
	def resetLevel(self, level):
		if level == 0: newObjList = self.topLevelList
		else:
			parentObj = self.levels[level-1].curObj
			if parentObj is None: newObjList = []
			else: newObjList = self.childsFuncs[level-1](parentObj)
		self.levels[level].reset(newObjList)
		self.updateLevelOutput(level)
		if level+1 < len(self.levels):
			self.resetLevel(level + 1)
	def selectObjById(self, level, objId):
		if objId == id(self.levels[level].curObj): return True
		if objId in self.levels[level].objDict:
			self.levels[level].curObjIndex = idx = self.levels[level].objDict[objId]
			self.levels[level].curObj = self.levels[level].objList[idx]
			self.updateLevelOutput(level)
			if level+1 < len(self.levels):
				self.resetLevel(level + 1)
			return True
		return False


class ProgNeuralInterface(GenericNeuralInterface):
	def childsOfProg(self, prog): return prog.allNodes()
	def childsOfNode(self, node): return sorted(node.edges)
	def nodeEdgeInfo(self, nodeEdge):
		if nodeEdge is None: return (0,) * (ObjectDim + self.IdVecLen)
		check, nextNode = nodeEdge
		assert type(check) is int # check is an object-int of MemoryBackend
		return _numToBinaryVec(check, ObjectDim) + self.objToVec(nextNode)
	def __init__(self, progPool):
		self.progPool = progPool
		GenericNeuralInterface.__init__(self,
		    topLevelList = progPool,
		    childsFuncs = [self.childsOfProg, self.childsOfNode],
		    additionalInfoFuncs = [None, None, self.nodeEdgeInfo]
		    )

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
		
