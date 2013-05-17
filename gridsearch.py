import scipy
import scipy.optimize
import rank
import ndcg
import numpy as np
def tune(w1,w2,w3,w4,w5):
	rank.weights = [w1,w2,w3,w4,w5]
	rank.main(1,"queryDocTrainData")
	return -ndcg.main("ranked.txt", "queryDocTrainRel")

#[x0,fval,grid,jout] = scipy.optimize.brute(tune, 
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1),
#	np.linspace(0.0, 3.0, num=1))

def tune3(w):
	rank.weights = w
	rank.main(1,"queryDocTrainData")
	print w
	return -ndcg.main("ranked.txt", "queryDocTrainRel")

print scipy.optimize.anneal(tune3, [1.0, 0.5, 0.1, 0.3, 2.0], lower=-1,upper=0,feps=0.0001,T0=3)

