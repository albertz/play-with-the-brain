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



import itertools, operator
import scipy
import pybrain.supervised as bt

# We just use bt.BackpropTrainer as a base.
# We ignore the target of the dataset though.
class ReinforcedTrainer(bt.BackpropTrainer):
    def __init__(self, module, rewarder, *args, **kwargs):
        bt.BackpropTrainer.__init__(self, module, *args, **kwargs)
        self.rewarder = rewarder # func (seq,last module-output) -> reward in [0,1]
	
    def _calcDerivs(self, seq):
        """Calculate error function and backpropagate output errors to yield
			the gradient."""
        self.module.reset()
        for sample in seq:
            self.module.activate(sample[0])
        error = 0.
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            subseq = itertools.imap(operator.itemgetter(0), seq[:offset+1])
            outerr2 = 1.0 - self.rewarder(subseq, self.module.outputbuffer[offset])
			
            target = sample[1]
            outerr = target - self.module.outputbuffer[offset]
            outerr2 = scipy.array([outerr2] * self.module.outdim)
            #print "derivs:", offset, ":", outerr, outerr2
            outerr = outerr2
            
            error += 0.5 * sum(outerr ** 2)
            ponderation += len(target)
            # FIXME: the next line keeps arac from producing NaNs. I don't
            # know why that is, but somehow the __str__ method of the
            # ndarray class fixes something,
            str(outerr)
            self.module.backActivate(outerr)
		
        return error, ponderation


#trainer = bt.RPropMinusTrainer(module=nn)
#trainer = bt.BackpropTrainer( nn, momentum=0.9, learningrate=0.00001 )
#trainer = bt.BackpropTrainer()

def errorFunc(seq, nnoutput):
    seq = [ "123ABCXYZ"[pybrain.utilities.n_to_one(sample)] for sample in seq ]
    lastout = outputAsVec(SeqGenerator().nextSeq(seq)[-1])
    diff = scipy.array(nnoutput) - scipy.array(lastout)
    err = 0.5 * scipy.sum(diff ** 2)
    return err

trainer = ReinforcedTrainer(module=nn, rewarder=rewardFunc)

from pybrain.tools.validation import ModuleValidator

import thread
def userthread():
    from IPython.Shell import IPShellEmbed
    ipshell = IPShellEmbed()
    ipshell()
#thread.start_new_thread(userthread, ())


# carry out the training
while True:
    trndata = generateData(nseq = 20, ratevarlimit = random.uniform(0.0,0.3))
    tstdata = generateData(nseq = 20)
    
    trainer.setData(trndata)
    trainer.train()
    trnresult = 100. * (ModuleValidator.MSE(nn, trndata))
    tstresult = 100. * (ModuleValidator.MSE(nn, tstdata))
    print "train error: %5.2f%%" % trnresult, ",  test error: %5.2f%%" % tstresult
	
    s = getRandomSeq(100, ratevarlimit=random.uniform(0.0,1.0))
    print " real:", seqStr(s)
    print "   nn:", getSeqOutputFromNN(nn, s)
