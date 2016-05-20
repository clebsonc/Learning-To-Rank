# coding=utf-8

import numpy as np
import sys

def find_max_relevance(r, rj="higher"):
    """
    Find the relevant judgments labels of a ranking list containing the associated relevance judgments.
    Date: 28/04/2016
    :param r: list containing the ranking with the related relevance judgment
    :param k: position to be computed
    :param rj: we can use the three following criteria to find the relevance judgment:
    1) higher: given a relevance judgment list with multi relevance judgments, consider only the higher  relevant document
    as the relevant. (g.e: considering the relevance judgments [4, 3, 2, 1, 0] only 4 is considered relevant)
    2) majority: considers middle relevance judgment as relevant as well (g.e: relevance [4, 3, 2, 1, 0] than 4 and 3 will
    be considered as relevant while the others are irrelevant
    :return: a list containing the elements that are relevant
    """
    max_relevance=np.nan
    if rj == 'higher':
        max_relevance = [max(r)]
    elif rj == "majority":
        max_relevance = np.sort(np.unique(r))
        max_relevance = max_relevance[-(len(max_relevance) / 2):]
    return max_relevance


def precision_at_k(r, k, rj="higher"):
    """
    Compute the precision at position given a relevance judgment ranking.
    Date: 28/04/2016
    :param r: list containing the ranking with the related relevance judgment
    :param k: position to be computed
    :param rj: since the precision at position considers only binary relevance, we can use the three following criteria to
    overcome this problem by defining which elements are relevant when considering multi relevance judgment.
    1) higher: given a relevance judgment list with multi relevance judgments, consider only the higher  relevant document
    as the relevant. (g.e: considering the relevance judgments [4, 3, 2, 1, 0] only 4 is considered relevant)
    2) majority: considers middle relevance judgment as relevant as well (g.e: relevance [4, 3, 2, 1, 0] than 4 and 3 will
    be considered as relevant while the others are irrelevant
    3) This is the recommended way, considering for examples, if a given query does not contain the most relevant documents
    in a set of queries, than it will consider the given relevant judgments, in case the other two are used, thant it the
    relevant documents are going to be the higher elements on the rank list.
    Just pass a list containing the labels to be considered relevant (g.e: relevance [4, 3, 2, 1, 0] and the desired
    relevant labels are 4 and 3, than just pass a list containing those labels [4, 3]
    :return: the precision at position @k
    """
    if k < 1:
        raise ValueError("the @k position must be greater than zero and the ranking must be a list")
    r = np.asarray(r)
    kr = r[0:k]
    if isinstance(rj, list):
        kr = [x for x in kr if x in rj]
    elif isinstance(rj, str):
        max_relevance = find_max_relevance(r, rj)
        kr = [x for x in kr if x in max_relevance]
    return len(kr)/float(k)


def average_precision(r, rj="higher"):
    """
    Compute the average precision
    Date: 28/04/2016
    :param r: ranking of the documents with its associated relevance judgment
    :param rj: since the precision at position considers only binary relevance, we can use the three following criteria to
    overcome this problem by defining which elements are relevant when considering multi relevance judgment.
    1) higher: given a relevance judgment list with multi relevance judgments, consider only the higher  relevant document
    as the relevant. (g.e: considering the relevance judgments [4, 3, 2, 1, 0] only 4 is considered relevant)
    2) majority: considers middle relevance judgment as relevant as well (g.e: relevance [4, 3, 2, 1, 0] than 4 and 3 will
    be considered as relevant while the others are irrelevant
    3) list containing the labels to be considered relevant (g.e: relevance [4, 3, 2, 1, 0] and the desired relevant labels
     are 4 and 3, than just pass a list containing those labels [4, 3]
    :return: average precision
    """
    if not isinstance(r, list):
        raise ValueError("The relevance judgment must be a list")
    precision_at_k_values = [precision_at_k(list(r), k=k, rj=rj) for k in xrange(1, len(r)+1, 1)]

    if not isinstance(rj, list):
        max_relevance = find_max_relevance(r, rj)
    else:  # rj is a list of the labels that are considered relevant
        max_relevance = rj
    ap = [precision_at_k_values[x] for x in xrange(len(r)) if r[x] in max_relevance]
    return sum(ap)/len(ap) if len(ap)>0 else 0


def mean_average_precision(qrl, rj="higher"):
    """
    Computes the Mean Average Precision
    Date: 28/04/2016
    :param qrl: Set of ranking for each query. Considering the example with multi relevance judgment:
        [[4, 0, 1, 3, 0, 0, 0], [3, 4, 2, 0, 0, 0, 0], ...] \n
        Each sublist of the list represents a query with 7 documents, where the relevance judgment for each documents are:
        doc1 -> rel 4
        doc2 -> rel 0
        doc3 -> rel 1
        ...
        doc7 -> rel 0
    :param rj: since the precision at position considers only binary relevance, we can use the three following criteria to
        overcome this problem by defining which elements are relevant when considering multi relevance judgment.
        1) higher: given a relevance judgment list with multi relevance judgments, consider only the higher  relevant document
           as the relevant. For the relevance judgments [4, 3, 2, 1, 0] only 4 is considered relevant. Considering
           the same set of relevance judgments [4, 3, 2, 1, 0], if a retrieved set of documents contains only [2, 1, 0], then
           only the document with relevance 2 will be considered relevant, even though all others queries contains docs with all 5 relevances.
        2) majority: considers middle relevance judgment as relevant as well (g.e: relevance [4, 3, 2, 1, 0] than 4 and 3 will
           be considered as relevant while the others are irrelevant. If a retrieved query contains docs with [3, 2, 1, 0], then only [3, 2]
           will considered relevant, even though all other queries contains docs with all other queries.
        3) list containing the labels to be considered relevant (g.e: relevance [4, 3, 2, 1, 0] and the desired relevant labels
           are 4 and 3, than just pass a list containing those labels [4, 3]. On the Letor and MS tools the default relevant judgment
           for a multi-relevance judgment with 5 level [4, 3, 2, 1, 0], the docs considered relevant are [4, 3, 2] while all others are considered
           to be non-relevant.
    :return: The Mean Average Precision, which is the mean of the average precision of each query
    """
    return np.mean([average_precision(r, rj) for r in qrl])


def dcg_at_k(r, k, equation=2):
    """
    Computes dcg@ following the equation:\n
    Date 29/04/2016
    1) $ rel_{i}+\sum_{i}^{m} \frac{rel_{i}}{log_{2}(i)}$ \n
    2) $\sum_{i}^{m} \frac{2**rel_{i} - 1}{log_{2}(i+1)}$. \n
    References:
    (1) - Learning to Rank for Information Retrieval (Liu, Tie-Yan - 2011)
    (2) - Tune and mix: learning to rank using ensembles of calibrated multi-class classifiers (Busa-Fekete et-al - 2013)
    (3) - Wikipedia page: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    :param r: list containing the relevance judgment sorted considering the predicted score for each document
    :param k: position to compute the relevance judgement
    :param method: which equation use to compute the ndcg: 1 for the first equation and 2 for the second equation.
    :return: the discounte cumulative gain at position k
    """
    if k>len(r):
        return 0.0;
    r = np.asfarray(r)[:k]  # select the documents (relevance judgments) up to the k^{th} position
    if r.size:  # r an k must not be zero
        if equation == 1:
            rel1 = r[0]
            num = r[1:]
            dem = np.log2(np.arange(2, k+1, 1))
            return rel1 + np.sum(num/dem)
        else:   # this is the default equation
            num = 2**r-1
            dem = np.log2(np.arange(1, k+1, 1)+1)
            return np.sum(num/dem)
    else:
        raise ValueError("The relevance judgments list must not be empty and k must be greater than zero")


def ndcg_at_k(r, k, equation=2, miss_relevant=0.0):
    """
    Computes the Normalized Discounted Cumulative Gain at position k (NDCG@k), which is the Discounted Cumulative Gain at position k (DCG@k)
    normalized by the the ideal DCG@k
    Date: 29/04/2016
    :param r: list containing the relevance judgments sorted in accordance with the documents score.
    :param k: the postion to compute the DNCG
    :param equation: Which equation use to compute the DCG@k: \n
    1) $ rel_{i}+\sum_{i}^{m} \frac{rel_{i}}{log_{2}(i)}$ \n
    2) $\sum_{i}^{m} \frac{2**rel_{i} - 1}{log_{2}(i+1)}$ \n
    :param miss_relevant: Considering the following reference, some tools attribute the score of 1.0 while other attribute the value of 0.0
    for queries that does not contain the relevant documents. Amongst the tools the considers 0.0 are RankLib, TREC and Letor3.0. The tools
    that considers 1.0 to empty queries are the YAHOO and the Kaggle script.
    :param rj: considering that we use the parameter miss_relevant to attribute a score of 1.0 or 0.0 to empty queries, it's necessary
    to consider one of the 3 following euristics to find the elements that are relevant:
    1) higher: given a relevance judgment list with multi relevance judgments, the higher relevant document is considered
    as the only judgment that really matters to be relevant the relevant.
    (g.e: considering the relevance judgments [4, 3, 2, 1, 0] only 4 is considered relevant) \n
    2) majority: considers middle relevance judgment as relevant as well (g.e: relevance [4, 3, 2, 1, 0] than 4 and 3 will
    be considered as relevant while the others are irrelevant \n
    3) list containing the labels to be considered relevant (g.e: relevance [4, 3, 2, 1, 0] and the desired relevant labels
     are 4 and 3, than just pass a list containing those labels [4, 3]
    Refereces:
        1 - Tune and mix: learning to rank using ensembles of calibrated multi-class classifiers (Busa-Fekete et-al - 2013)
        2 - https://www.kaggle.com/wiki/NormalizedDiscountedCumulativeGain
    :return: the NDCG@k
    """
    if k>len(r) or max(r)==0:
        if miss_relevant==1.0:
            return 1.0
        return 0.0

    rk = r[:k]

    # if the query contains any of the relevant documents on the maxrel list, then compute the ndcg metric
    # idcg = dcg_at_k(r=sorted(rk, reverse=True), k=k, equation=equation)
    idcg = dcg_at_k(r=sorted(r, reverse=True), k=k, equation=equation)
    dcg = dcg_at_k(r=rk, k=k, equation=equation)
    return dcg/idcg


def read_score(path):
    """
    Reads the output score from the model
    Date: December/2015
    :param path: file that contains the score for each document. First score refers to the first doc, second to the second
    doc and so on.
    :return: a list containing the score for each document
    """
    path = open(path, 'r')
    lines = path.readlines()
    for idx in xrange(len(lines)):
        lines[idx] = float(lines[idx].strip())
    return lines


def readfile(file=None, query_id=True):
    """
    Reads a training or test file in svmlight, ranklib, ranksvm format
    Date: December/2015
    :param trainfile: Training data set
    :param testfile: Test data set
    :param query_id: if true, returns the query id array for each document
    :return: a tuple of dense matrix containing the train and test file, each element of the tuple are made out of:
            [0]: sparse matrix containing the docs and its features
            [1]: ndarray of shape (n_samples) containing the labels (relevance judgement for each document)
            [2]: the query id
    """
    from sklearn.datasets import load_svmlight_file
    try:
        if file == None:
            raise ValueError("File can not be empty.")
        else:
            train = load_svmlight_file(f=file, query_id=query_id)
            if query_id:
                td = (train[0].toarray(), train[1], train[2])
            else:
                td = (train[0].toarray(), train[1])
            return td
    except ValueError as err:
        print err.args


def count_el_region(y):  # private method
    """
    Count how many distinct documents there are on y. (g.e: If y is a list of relevance judgment, it will count how many
    documents there are for each relevance judgment. If y is the list of queries, it will count how many queries there are
    and how many documents per query on the list). \n
    Date: December 2015
    :param y: parameter containing the set of values in a np.array format. Example:
                y = np.asarray([0,1,2,2,2,1,0,2,3,1,1,2,2,2,3,2,2,3,4,1,0,2,1,2,2,3,1,2,4,2])

    :return: a dictionary whose keys are the distinct elements and the values are the amount of element per keys. For the
             above example, the return will be:
                {0: 3, 1: 7, 2: 14, 3: 4, 4: 2}
    """
    region = {}  # dictionary
    for val in y:
        if not region.has_key(val):
            region[val] = 1
        else:
            region[val] += 1
    return region


def find_buckets_query(q_train):
    """
    Find the bucket of elements belonging to each query.\n
    Date: December/2015
    For instance, on TD2003: \n
    q: (range of documents) \n
    1: (0, 999) \n
    2: (1000, 1999) \n
    ... \n
    30: (28058, 29057) \n\n

    :param q_train: np.array containing the set of queries, where each position of the vector refers to a single
    document on the training set

    :return: a dictionary whose keys are query id's and values are a tuple containing the range (start, end) on the list
             of values.
    """
    region = count_el_region(q_train)  # region contains only the amount of documents in each query
    set_of_keys = sorted(region.keys(), reverse=False)  # sort the keys, in order to get the lowest key as the first
    # KEY, this is necessary, since dictionaries does not hold an order
    sum = 0
    for key in set_of_keys:
        if key == set_of_keys[0]:  # since the set of keys list is sorted on the first loop, this is always valid on
            # the first time
            sum += region[key]
            region[key] = [0, sum-1]
        else:  # after the first iteration, this will always execute
            kv = region[key]
            region[key] = [sum, sum+kv-1]
            sum += kv
    return region


def get_rank_list_per_query(feature_file, prediction_file):
    """
    Return a list containing sublists of relevance judgment sorted accordingly with the prediction file.
    :param feature_file: The original file containing the feature_file (g.e: test file used during the predicted task)
    :param prediction_file: the score for each document on the feature_file
    :return: a list containing list of sorted relevance judgment, where the elements are sorted accordingly with the prediction file
    """
    score = read_score(prediction_file)
    x, y, q = readfile(feature_file)
    queries = find_buckets_query(q)

    rank = [ [score[idx], y[idx], q[idx]] for idx in xrange(len(y))]

    # sort the scores by queries
    for q in queries:
        start, end = queries[q]
        # print q, start, end
        rank[start:(end+1)] = sorted(rank[start:(end+1)], reverse=True)

    rank_by_query = list()
    for q in sorted(queries.keys()):
        start, end = queries[q]
        rank_by_query.append(list(np.array(rank[start:(end+1)])[:, 1]))
    return rank_by_query


"""def get_rank_list_per_query_score(y, prediction_file, queries, relevant = None):
    if relevant is None:
        relevant = [max(y)]
    rank = list()
    for idx in xrange(len(y)):
        if y[idx] not in relevant:
            rank.append([prediction_file[idx], y[idx]])
        else:
            rank.append([prediction_file[idx], min(relevant)])

    # sort the scores by queriesby Query:
    for q in queries:
        start, end = queries[q]
        rank[start:(end+1)] = sorted(rank[start:(end+1)], reverse=True)

    rank_by_query = {}
    for q in queries:
        start, end = queries[q]
        rank_by_query[q] = np.array(rank[start:(end+1)])[:,1]
    return rank_by_query
"""

def print_metrics(rbq, rj="higher", file=False, equation=2, miss_relevant=0.0):
    ap=list()
    for idxr in xrange(len(rbq)):
        ap.append(average_precision(rbq[idxr]))
    map = mean_average_precision(rbq, rj)

    ndcg = ["NDCG@" + str(x) for x in xrange(1, 11, 1)]
    ndcg_at = list()
    for idxr in xrange(len(rbq)):
        #  compute NDCG@K, here only until the 10th position
        ndcg_at.append([ndcg_at_k(rbq[idxr], k=k, equation=equation, miss_relevant=miss_relevant) for k in xrange(1, 11, 1)])

    if file == False:
        for idxr in xrange(len(ap)):
            print "Ap-q:"+str(idxr), ap[idxr]
        print "MAP", mean_average_precision(rbq, rj)
        print
        for v in ndcg:
            sys.stdout.write(v+" ")
        sys.stdout.write("\n")
        for idxr in xrange(len(ndcg_at)):
            for v in ndcg_at[idxr]:
                sys.stdout.write(str(v)+" ")
            print
        print
    else:   # recond data on file
        file = open(file, "w")
        for idxr in xrange(len(ap)):
            file.write("Ap-q:"+str(idxr)+" "+str(ap[idxr])+"\n")
        file.write("MAP: "+str(map)+"\n\n")
        for idxr in xrange(len(ndcg)):
            file.write(ndcg[idxr] + " ")
        file.write("\n")
        for idxr in xrange(len(ndcg_at)):
            for v in ndcg_at[idxr]:
                file.write(str(v)+" ")
            file.write("\n")
        file.close()


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print "Parameters are: "
        print "argv[1]: Feature File"
        print "argv[2]: Prediction File"
        print "argv[3]: Output File"
        print "argv[4]: Miss Relevant: 1 or 0 (default is 0)"
        print "argv[5]: Equation: 1 or 2 (default is 2)"
    else:
        feature_file = sys.argv[1]
        prediction_file = sys.argv[2]
        out_file = sys.argv[3] if sys.argv[3]!="False" else False
        miss_relevant = int(sys.argv[4])
        equation = int(sys.argv[5])
        print "Feature File: ", feature_file
        print "Prediction File: ", prediction_file
        print "Output File: ", out_file
        print "Miss Relevant: ", miss_relevant
        print "Equation: ", equation
        # ndcg_at = int(sys.argv[3])
        rank_by_query = get_rank_list_per_query(feature_file=feature_file, prediction_file=prediction_file)
        print_metrics(rbq=rank_by_query, rj="higher", file=out_file, miss_relevant=miss_relevant, equation=equation)

