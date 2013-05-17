import sys
import re
from math import log10, log, exp
import pickle

#[title, header, url, body, anchor]

############# Task 1 Parameter #############
weights = [1.0, 0.5, 0.1, 0.3, 2.0]

############# Task 2 Parameter #############
W = [1.0, 0.3, 0.1, 0.3, 2.0]
B = [1.0, 1.0, 1.0, 1.0, 1.0]
K = 0.5
lambda1 = 0.5
lambda2 = 0.5
lambda3 = 0.5

############# Task 3 Parameter #############
weights_task3 = [1.0, 0.5, 0.1, 0.3, 2.0]
Boost = 60.0

##################################


#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features) 

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features):
    rankedQueries = {}
    for query in queries.keys():
      results = queries[query]
      #features[query][x].setdefault('body_hits', {}).values() returns the list of body_hits for all query terms
      #present in the document, empty if nothing is there. We sum over the length of the body_hits array for all
      #query terms and sort results in decreasing order of this number
      rankedQueries[query] = sorted(results, 
                                    key = lambda x: sum([len(i) for i in 
                                    features[query][x].setdefault('body_hits', {}).values()]), reverse = True)

    return rankedQueries

######################################### Task 1 ##########################################################
def load_doc_freq(docFreqFile):
    term_doc_freq = {}
    with open(docFreqFile, 'rb') as ff:
        term_doc_freq = pickle.load(ff)
    return term_doc_freq

def vector_from_text(items, content):
    vec = [0] * len(items)
    content = content.split()
    for idx in range(len(items)):
        vec[idx] = content.count(items[idx])
    return vec

def vector_product(vec1, vec2):
    assert(len(vec1) == len(vec2))
    return [ (vec1[i] * vec2[i]) for i in range(len(vec1)) ]

def vector_dot_product(vec1, vec2):
    return sum(vector_product(vec1, vec2))

def vector_sum(vec1, vec2):
    assert(len(vec1) == len(vec2))
    return [ (vec1[i] + vec2[i]) for i in range(len(vec1)) ]

def vector_scale(vec, alpha):
    return [ (float(alpha) * float(u)) for u in vec ]

def sublinear_scale(vec):
    rvec = []
    for u in vec:
        if u == 0:
            rvec.append(0)
        else:
            rvec.append(1 + log(u))
    return rvec

def vector_doc_freq(items, doc_freq):
    vec = [0] * len(items)
    for idx in range(len(items)):
        vec[idx] = log10(doc_freq[items[idx]])
    return vec

def weight_average(vecList, normalizer):
    global weights
    rvec = [0] * len(vecList[0])
    for idx in range(len(vecList)):
        #rvec = vector_sum(rvec, vector_scale(sublinear_scale(vecList[idx]), float(weights[idx])/float(normalizer)))
        rvec = vector_sum(rvec, vector_scale(vecList[idx], float(weights[idx])/float(normalizer)))
    return rvec
        
def task1(queries, features, doc_freq):
    rankedQueries = {}
    for query in queries.keys():
        # Query item and query vector
        qitem = list(set(query.split()))
        qvec = sublinear_scale(vector_from_text(qitem, query))
        #qvec = vector_from_text(qitem, query)
        idf = vector_doc_freq(qitem, doc_freq)
        qvec = vector_product(qvec, idf)
        
        # Calculate vectors and scores
        results = queries[query]
        feat = {}
        for x in results:
            # title
            title = features[query][x]['title']
            title_vec = vector_from_text(qitem, title)
            # url
            url = re.sub(r'\W+', ' ', x)
            url_vec = vector_from_text(qitem, url)
            # header
            header_vec = [0] * len(qitem)
            if 'header' in features[query][x]:
                header_arr = features[query][x]['header']
                for header in header_arr:
                    header_vec = vector_sum(header_vec, vector_from_text(qitem, header))
            # body
            body_vec = [0] * len(qitem)
            if 'body_hits' in features[query][x]:
                body = features[query][x]['body_hits']
                body_vec = [len(body.setdefault(item, [])) for item in qitem]
            # achors
            anchor_vec = [0] * len(qitem)
            if 'anchors' in features[query][x]:
                anchor = features[query][x]['anchors']
                for key in anchor:
                    anchor_vec = vector_sum(anchor_vec, [anchor[key] * u for u in vector_from_text(qitem, key)])
            # length normalization
            norm = int(features[query][x]['body_length']) + 500
            dvec = weight_average([title_vec, header_vec, url_vec, body_vec, anchor_vec], norm)
            feat[x] = vector_dot_product(qvec, dvec)
        rankedQueries[query] = [u[0] for u in sorted(feat.items(), key=lambda x:x[1], reverse=True)]
    return rankedQueries


####################################### Task 2 ###############################################
def avg_field_len(features):
    title, header, body, url, anchor = [], [], [], [], []
    for query in features:
        for x in features[query]:
            title.append(len(features[query][x]['title'].split()))
            url.append(len(re.sub(r'\W+', ' ', x).split()))
            body.append(int(features[query][x]['body_length'])+500)
            if 'header' in features[query][x]:
                header_arr = features[query][x]['header']
                hlen = 0
                for h in header_arr:
                    hlen = hlen + len(h.split())
                header.append(hlen)
            if 'anchors' in features[query][x]:
                anchor_arr = features[query][x]['anchors']
                alen = 0
                for key in anchor_arr:
                    alen = alen + len(key.split()) * anchor_arr[key]
                anchor.append(alen)
    titleLen = float(sum(title)) / float(len(title))
    headerLen = float(sum(header)) / float(len(header))
    bodyLen = float(sum(body)) / float(len(body))
    urlLen = float(sum(url)) / float(len(url))
    anchorLen = float(sum(anchor)) / float(len(anchor))
    return titleLen, headerLen, bodyLen, urlLen, anchorLen

def V_log(score):
    global lambda2
    return log(score + lambda2)

def V_saturate(score):
    global lambda2
    return float(score) / float(score + lambda2)

def V_sigmoid(score):
    global lambda2, lambda3
    return 1.0 / (float(lambda2) + exp(-score * lambda3))

def BM2F_score(dvecList, lenRatios, pgrank, qvec):
    global B, W, K, lambda1
    # Weighted average
    dvec = [0] * len(dvecList[0])
    for idx in range(len(B)):
        if lenRatios[idx] != 0:
            dvecList[idx] = vector_scale(sublinear_scale(dvecList[idx]), W[idx] / (1.0 + B[idx] * (lenRatios[idx] - 1.0)))
            dvec = vector_sum(dvec, dvecList[idx])
    # Dot product
    score = 0.0
    for idx in range(len(dvec)):
        score = score + dvec[idx] / (dvec[idx] + K) * qvec[idx]
    score = score + lambda1 * V_log(pgrank)
    return score
    
def task2(queries, features, doc_freq):
    tL, hL, bL, uL, aL = avg_field_len(features)
    rankedQueries = {}
    for query in queries.keys():
        # Query item and query vector
        qitem = list(set(query.split()))
        qvec = sublinear_scale(vector_from_text(qitem, query))
        idf = vector_doc_freq(qitem, doc_freq)
        qvec = vector_product(qvec, idf)
        
        # Calculate vectors and scores
        results = queries[query]
        feat = {}
        for x in results:
            # title
            title = features[query][x]['title']
            title_vec = vector_from_text(qitem, title)
            tR = float(len(title.split())) / tL
            # url
            url = re.sub(r'\W+', ' ', x)
            url_vec = vector_from_text(qitem, url)
            uR = float(len(url.split())) / uL
            # header
            header_vec = [0] * len(qitem)
            hR = 0.0
            if 'header' in features[query][x]:
                header_arr = features[query][x]['header']
                hLen = 0
                for header in header_arr:
                    header_vec = vector_sum(header_vec, vector_from_text(qitem, header))
                    hLen = hLen + len(header.split())
                hR = float(hLen) / hL
            # body
            body_vec = [0] * len(qitem)
            bR = 0.0
            if 'body_hits' in features[query][x]:
                body = features[query][x]['body_hits']
                body_vec = [len(body.setdefault(item, [])) for item in qitem]
                bLen = int(features[query][x]['body_length']) + 500
                bR = float(bLen) / bL
            # achors
            anchor_vec = [0] * len(qitem)
            if 'anchors' in features[query][x]:
                anchor = features[query][x]['anchors']
                aLen = 0
                for key in anchor:
                    anchor_vec = vector_sum(anchor_vec, [anchor[key] * u for u in vector_from_text(qitem, key)])
                    aLen = aLen + len(key.split())
                aR = float(aLen) / aL
            # page rank
            pgrank = features[query][x]['pagerank']

            # BM2F scoring
            feat[x] = BM2F_score([title_vec, header_vec, url_vec, body_vec, anchor_vec], \
                                 [tR, hR, uR, bR, aR], pgrank, qvec)
        rankedQueries[query] = [u[0] for u in sorted(feat.items(), key=lambda x:x[1], reverse=True)]
    return rankedQueries


####################################### Task 3 ###############################################
def compute_window(qitem, text):
    if not set(text).issuperset(set(qitem)):
        return float("inf")
    if len(qitem) == 1:
        return 1.0
    index = [text.index(u) for u in qitem]
    win = max(index) - min(index) + 1
    for i in range(len(text)):
        for j in range(len(qitem)):
            if qitem[j] == text[i]:
                index[j] = i
                if max(index) - min(index) + 1 < win:
                    win = max(index) - min(index) + 1
                break
    return win

def compute_body_window(qitem, body):
    if not set(body.keys()).issuperset(set(qitem)):
        return float("inf")
    if len(qitem) == 1:
        return 1.0
    d = {}
    for key in body:
        if key in qitem:
            for pos in body[key]:
                temp = d.setdefault(pos, [])
                temp.append(key)
                d[pos] = temp
                
    win = float("inf")
    index = [float("inf")] * len(qitem)
    for pos in d:
        for word in d[pos]:
            index[qitem.index(word)] = pos
            if max(index) - min(index) + 1 < win:
                win = max(index) - min(index) + 1
    return win
    
def boosted_weighted_score(dvecList, normalizer, qvec, window_size):
    global weights_task3
    rvec = [0] * len(dvecList[0])
    for idx in range(len(dvecList)):
        #rvec = vector_sum(rvec, vector_scale(sublinear_scale(dvecList[idx]), float(weights_task3[idx])/float(normalizer)))
        rvec = vector_sum(rvec, vector_scale(dvecList[idx], float(weights_task3[idx])/float(normalizer)))
    score = vector_dot_product(rvec, qvec)
    if window_size == float("inf"):
        return score
    elif len(qvec) == 1:
        return score
    elif window_size == len(qvec):
        return Boost * score
    else:
        return float(Boost) / float(window_size) * float(score)


def task3(queries, features, doc_freq):
    rankedQueries = {}
    for query in queries.keys():
        # Query item and query vector
        qitem = list(set(query.split()))
        qvec = sublinear_scale(vector_from_text(qitem, query))
        #qvec = vector_from_text(qitem, query)
        idf = vector_doc_freq(qitem, doc_freq)
        qvec = vector_product(qvec, idf)
        
        # Calculate vectors and scores
        results = queries[query]
        feat = {}
        for x in results:
            # title
            title = features[query][x]['title']
            title_vec = vector_from_text(qitem, title)
            title_win = compute_window(qitem, title.split())
            # url
            url = re.sub(r'\W+', ' ', x)
            url_vec = vector_from_text(qitem, url)
            url_win = compute_window(qitem, url.split())
            # header
            header_vec = [0] * len(qitem)
            header_win = float("inf")
            if 'header' in features[query][x]:
                header_arr = features[query][x]['header']
                for header in header_arr:
                    header_vec = vector_sum(header_vec, vector_from_text(qitem, header))
                    header_win = min([header_win, compute_window(qitem, header.split())])
            # body
            body_vec = [0] * len(qitem)
            body_win = float("inf")
            if 'body_hits' in features[query][x]:
                body = features[query][x]['body_hits']
                body_vec = [len(body.setdefault(item, [])) for item in qitem]
                body_win = compute_body_window(qitem, body)
            # achors
            anchor_vec = [0] * len(qitem)
            anchor_win = float("inf")
            if 'anchors' in features[query][x]:
                anchor = features[query][x]['anchors']
                for key in anchor:
                    anchor_vec = vector_sum(anchor_vec, [anchor[key] * u for u in vector_from_text(qitem, key)])
                    anchor_win = min([anchor_win, compute_window(qitem, key.split())])
            # window size
            WIN = min([title_win, header_win, url_win, body_win, anchor_win])
            # length normalization
            norm = int(features[query][x]['body_length']) + 500
            # Boosted score
            feat[x] = boosted_weighted_score([title_vec, header_vec, url_vec, body_vec, anchor_vec], norm, qvec, WIN)
            
        rankedQueries[query] = [u[0] for u in sorted(feat.items(), key=lambda x:x[1], reverse=True)]
    return rankedQueries

#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)

def printToFileRankedResults(queries, outputFile):
    with open(outputFile, 'w') as ff:
        for query in queries:
            ff.write("query: " + query + "\n")
            for res in queries[query]:
                ff.write("  url: " + res + "\n")

#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFile = "ranked.txt" #Please don't change this!

    #populate map with features from file
    (queries, features) = extractFeatures(featureFile)

    #load document frequency
    term_doc_freq = load_doc_freq("term_doc_freq")

    #calling baseline ranking system, replace with yours
    #rankedQueries = baseline(queries, features)
    rankedQueries = task1(queries, features, term_doc_freq)
    #rankedQueries = task2(queries, features, term_doc_freq)
    #rankedQueries = task3(queries, features, term_doc_freq)
    print >> sys.stderr, sum([len(rankedQueries[u]) for u in rankedQueries])
    
    #print ranked results to file
    #printRankedResults(rankedQueries)
    printToFileRankedResults(rankedQueries, outputFile)
       
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    #main(sys.argv[1])
    main("queryDocTrainData")


