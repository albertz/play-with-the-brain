#!/usr/bin/python

class SeqGenerator:
    def __init__(self):
        self.lastNum = self.lastLetter = ""

    def peek(self, nextInput):
        if nextInput in ["1","2"]:
            return "L"
        elif nextInput in ["A","B"]:
            return "L"
        elif nextInput in ["X","Y"]:
            seq = self.lastNum + self.lastLetter + nextInput
            if seq in ["1AX","2BY"]: return "R"
            return "L"
        return ""

    def next(self, nextInput):
        out = self.peek(nextInput)
        if nextInput in ["1","2"]:
            self.lastNum = nextInput
            self.lastLetter = ""
        elif nextInput in ["A","B","X","Y"]:
            self.lastLetter = nextInput
        return out

    def nextSeq(self, nextInputs):
        return [ self.next(c) for c in nextInputs ]

    def nextStr(self, nextInputs):
        return "".join([ self.next(c) for c in nextInputs ])

def seqStr(s): return SeqGenerator().nextStr(s)

import pybrain
import pybrain.tools.shortcuts as bs
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer, SoftmaxLayer
import pybrain.structure.networks as bn
import pybrain.structure.connections as bc
import pybrain.datasets.sequential as bd


print "preparing network ...",
nn = bn.RecurrentNetwork()
nn.addInputModule(LinearLayer(9, name="in"))
nn.addModule(LSTMLayer(6, name="hidden"))
nn.addOutputModule(LinearLayer(2, name="out"))
nn.addConnection(bc.FullConnection(nn["in"], nn["hidden"], name="c1"))
nn.addConnection(bc.FullConnection(nn["hidden"], nn["out"], name="c2"))
nn.addRecurrentConnection(bc.FullConnection(nn["hidden"], nn["hidden"], name="c3"))
nn.sortModules()
print "done"


import random

def getRandomSeq(seqlen, ratevarlimit=0.2):
    s = ""
    count = 0
    gen = SeqGenerator()
    for i in xrange(seqlen):
        if(float(count) / (i+1) < random.uniform(0.0,ratevarlimit)):
            # ignore lastNumber - make it only 50% of the cases right -> to point out the difference in learning
            if gen.lastLetter == "A": c = "X"
            elif gen.lastLetter == "B": c = "Y"
            elif gen.lastNum != "": c = random.choice("AB")
            else: c = random.choice("12")
            #if gen.lastNum + gen.lastLetter == "1A": c = "X"
            #elif gen.lastNum + gen.lastLetter == "2B": c = "Y"
            #elif gen.lastNum == "1": c = "A"
            #elif gen.lastNum == "2": c = "B"
            #else: c = random.choice("12")
        else:
            c = random.choice("123ABCXYZ")
        s += c
        if gen.next(c) == "R": count += 1
    return s

import pybrain.utilities
def inputAsVec(c): return pybrain.utilities.one_to_n("123ABCXYZ".index(c), 9)
def outputAsVec(c):
    if c == "": return (0.0,0.0)
    else: return pybrain.utilities.one_to_n("LR".index(c), 2)

def addSequence(dataset, seqlen, ratevarlimit):
    dataset.newSequence()
    s = getRandomSeq(seqlen, ratevarlimit)
    for i,o in zip(s, SeqGenerator().nextSeq(s)):
        dataset.addSample(inputAsVec(i), outputAsVec(o))

def generateData(seqlen = 100, nseq = 20, ratevarlimit = 0.2):
    dataset = bd.SequentialDataSet(9, 2)
    for i in xrange(nseq): addSequence(dataset, seqlen, ratevarlimit)
    return dataset

def getActionFromNNOutput(nnoutput):
    l,r = nnoutput
    l,r = l > 0.5, r > 0.5
    if l and not r: c = "L"
    elif not l and r: c = "R"
    elif not l and not r: c = ""
    else: c = "?"
    return c

def getSeqOutputFromNN(module, seq):
    outputs = ""
    module.reset()
    for i in xrange(len(seq)):
        output = module.activate(inputAsVec(seq[i]))
        c = getActionFromNNOutput(output)
        outputs += c
    return outputs

def rewardFunc(seq, nnoutput):
    cl,cr = outputAsVec(SeqGenerator().nextSeq(seq)[-1])
    nl,nr = nnoutput
    reward = 0.0
    if 1.5 > nl > 0.5 and cl > 0.5: reward += 0.5
    if -0.5 < nl < 0.5 and cl < 0.5: reward += 0.5
    if 1.5 > nr > 0.5 and cr > 0.5: reward += 0.5
    if -0.5 < nr < 0.5 and cr < 0.5: reward += 0.5
    if nl < -0.5 or nl > 1.5: reward -= 0.5
    if nr < 0.5 or nr > 1.5: reward -= 0.5
    return reward

import pybrain.rl.environments as be
class Task12AX(be.EpisodicTask):
    def __init__(self): self.reset()
    #def setMaxLength(self, n): pass #ignore
    def getReward(self):
        return rewardFunc(self.seq[:self.t], self.actions[-1])
    def reset(self):
        self.cumreward = 0
        self.t = 0
        self.seq = getRandomSeq(seqlen = 100, ratevarlimit = random.uniform(0.0,0.3))
        self.actions = []
    def performAction(self, action):
        self.t += 1
        self.actions.append(action)
        self.addReward()
    def isFinished(self):
        return len(self.actions) >= len(self.seq)
    def getObservation(self):
        return inputAsVec(self.seq[self.t])

from pybrain.optimization import *
from pybrain.tools.validation import ModuleValidator
from numpy.random import randn

maxLearningSteps = 10
thetask = Task12AX()
tstdata = generateData(nseq = 20)
blackboxoptimmethod = ES

def mutator():
    nn._params += randn(nn.paramdim) * random.uniform(0.0,1.0)
nn.mutate = mutator


while True:
    nn, value = blackboxoptimmethod(thetask, nn, maxLearningSteps=maxLearningSteps, elitism=True).learn()
    print "best evaluation:", value
    
    tstresult = 100. * (ModuleValidator.MSE(nn, tstdata))
    print "test error: %5.2f%%" % tstresult

    s = getRandomSeq(100, ratevarlimit=random.uniform(0.0,1.0))
    print " real:", seqStr(s)
    print "   nn:", getSeqOutputFromNN(nn, s)

