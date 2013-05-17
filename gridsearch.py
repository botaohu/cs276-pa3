import scipy
import scipy.optimize
import rank
import ndcg
import numpy as np

#[x0,fval,grid,jout] = scipy.optimize.brute(tune, 
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1))


def tune1(w):
	rank.weights = w
	rank.main(1,"queryDocTrainData")
	print w
	return -ndcg.main("ranked.txt", "queryDocTrainRel")


[x1,fval,grid,jout] = scipy.optimize.brute(tune1, 
	np.s_[0:1:0.3, 0:1:0.3,0.1:0.2:1,0.3:0.5:1,2:3:1])


def tune3(w):
	rank.weights_task3 = w[0:5]
	rank.Boost = w[5]
	rank.main(3,"queryDocTrainData")
	print w
	return -ndcg.main("ranked.txt", "queryDocTrainRel")

def tune3_(w):
	rank.weights_task3 = [1.0, 0.5, 0.1, 0.3, 2.0]
	rank.Boost = w[0]
	rank.main(3,"queryDocTrainData")
	print w
	return -ndcg.main("ranked.txt", "queryDocTrainRel")


#print scipy.optimize.anneal(tune1, [1.0, 0.5, 0.1, 0.3, 2.0], lower=-1,upper=0,feps=0.0001,T0=3)

#print scipy.optimize.anneal(tune3_, [600], lower=-1,upper=0,learn_rate=0.9,feps=0.0001,T0=0.2,maxiter=100)

