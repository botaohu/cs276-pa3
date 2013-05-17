import scipy
import rank
import ndcg
import numpy as np
def tune(w1,w2,w3,w4,w5):
	rank.weights = [w1,w2,w3,w4,w5]
	rank.main(1,"queryDocTrainData")
	return -ndcg.main("ranked.txt", "queryDocTrainRel")

[x0,fval,grid,jout] = scipy.optimize.brute(tune, 
	np.linspace(0.0, 3.0, num=1),
	np.linspace(0.0, 3.0, num=1),
	np.linspace(0.0, 3.0, num=1),
	np.linspace(0.0, 3.0, num=1),
	np.linspace(0.0, 3.0, num=1))

def tune2(w):
	rank.weights = w
	rank.main(1,"queryDocTrainData")
	return -ndcg.main("ranked.txt", "queryDocTrainRel")

print scipy.optimize.anneal(tune2, [1.0, 0.5, 0.1, 0.3, 2.0])