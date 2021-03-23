import sys
import math
import numpy as np
from collections import Counter
import copy
import time
from matplotlib import pyplot as plt

# consts
SUBMITTER_NAME = ""
SUBMITTER_ID = ""
NUM_OF_CLUSTERS = 9
EPSILON = 0.001 #sys.float_info.epsilon
WORD_COUNT_FILTER = 3
LAMBDA = 0.08
EM_THRESHOLD = 0.01
K = 10
DEBUG_MODE = False # True
TEST_AFTER_EACH_ITER = True
CREATE_ACCURACY_GRAPH = True
CREATE_CONFUSION_MATRIX_CSV = True




def DEBUG_PRINT(msg):
    if DEBUG_MODE:
        print(msg)



# a class for reading an input file of the exercise format
class Reader:
    def __init__(self, input_dataset_filename, topics_filename):
        """
        Constructor
        :param input_dataset_filename: the input dataset file name
        :param topics_filename: the topics file names
        """

        self._input_dataset_filename = input_dataset_filename
        self._topics_filename = topics_filename
        self._parsed_data = []
        self._topics_per_data = []
        self._topics_data = []


    def parse(self):
        """
        Parse data into list of string-lists (each string-list represent the words of the article in that index, in order)
        Also parses the topics per each article and a list of topics.
        """

        # dataset
        with open(self._input_dataset_filename) as input_dataset_file:
            for line in input_dataset_file:
                if line[:6] != '<TRAIN' and line[:5] != '<TEST' and line != '\n':
                    self._parsed_data.append(line.split())
                elif line[:6] == '<TRAIN':
                    topics = line.split()[2:]
                    topics[-1] = topics[-1].split('>')[0]
                    self._topics_per_data.append(topics)


        # topics
        with open(self._topics_filename) as topics_file:
            for line in topics_file:
                if line != "\n":
                    self._topics_data.append(line.strip())


    def get(self):
        """
        Get parsed data
        :return: parsed data (list of string-lists), topics for each data, topics
        """

        return self._parsed_data, self._topics_per_data, self._topics_data



# The main class for all data processing (EM algorithm and all around it)
class DataProcessor:
    def __init__(self, dev_set_data, topics_per_data, topics_data):
        """
        Constructor - initializes the EM algorithm
        :param: dev_set_data - the dev set data
        :param: topics_per_data - topics for each data
        :param: topics_data - a list of the topics
        """

        self._dev_set_data = dev_set_data
        self._topics_per_data = topics_per_data
        self._topics_data = topics_data

        #self._dev_set_data_filtered = [""]*len(self._dev_set_data)
        self._corpus = []
        # create the n_t and n_t_k
        self._n_t = []
        self._n_t_k = []
        self._n_k_t = {}

        # number of words in the dataset - will be init later
        self._num_of_words_in_dataset = 0

        # number of articles
        self._N = len(dev_set_data)

        # pre process - tune the n_t and n_t_k correctly after filtering rare words
        self._pre_process()
        DEBUG_PRINT("finished pre-process")

        # calculate initial w_t_i
        self._w_t_i = np.zeros((self._N, NUM_OF_CLUSTERS)) #[[0.0]*NUM_OF_CLUSTERS]*self._N
        for t in range(self._N):
            self._w_t_i[t, t % NUM_OF_CLUSTERS] = 1.0
        DEBUG_PRINT("finished init w_t_i")

        # calculate the initial alpha[i] and  P[i][k]
        self._alpha_i = np.zeros(NUM_OF_CLUSTERS)
        self._p_i_k = np.zeros((NUM_OF_CLUSTERS, len(self._corpus)))#[] #np.zeros(NUM_OF_CLUSTERS)
        self._calc_m_step()
        DEBUG_PRINT("finished init m step")

        # for final testing
        self._confusion_matrix = None
        self._accuracy = 0

    def _pre_process(self):
        """
        Pre process : filter rare words and apply it to create n_t, n_t_k and corpus
        """

        # 1. create the concatenation of all articles and word count per article
        # 2. put initial word frequencies into n_t_k (pre filtering)
        # 3. put initial length into the n_k (pre filtering)
        dev_set_all_words = []
        for entry in self._dev_set_data:
            dev_set_all_words += entry
            self._n_t_k.append(Counter(entry))
            self._n_t.append(len(entry))

        # create corpus counting
        dev_set_all_words_count = Counter(dev_set_all_words)
        self._corpus = copy.deepcopy(dev_set_all_words_count)

        # filter rare words - adjust n_t_k and n_t accordingly, also affect in corpus
        for word in dev_set_all_words_count.keys():
            if dev_set_all_words_count[word] <= WORD_COUNT_FILTER:
                for t in range(len(self._n_t_k)):
                    if word in self._n_t_k[t].keys():
                        self._n_t[t] -= self._n_t_k[t][word]
                        del(self._n_t_k[t][word])
                        del(self._corpus[word])
                        #self._dev_set_data_filtered[t] = list(filter(lambda w: w != word, self._dev_set_data[t]))

        # the corpus will only be the list of words appearing the articles
        self._corpus = list(self._corpus.keys())
        DEBUG_PRINT("corpus size: " + str(len(self._corpus)))

        # create an n_k_t and n_k_t matrix for more efficient calculation later
        self._n_k_t_mat = np.zeros((len(self._corpus), self._N))
        for k in self._corpus:
            self._n_k_t[k] = np.array([self._n_t_k[t][k] for t in range(self._N)])
            #print(self._corpus.index(k))
            self._n_k_t_mat[self._corpus.index(k)] = self._n_k_t[k]

        # number of words in the dataset
        self._num_of_words_in_dataset = np.sum(np.array(self._n_t))

    def _calc_m_step(self):
        """
        EM algorithm M step
        """

        # calculate alpha_i
        for i in range(NUM_OF_CLUSTERS):
            self._alpha_i[i] = float(np.sum(self._w_t_i[:, i]))/self._N

        self._alpha_i[self._alpha_i < EPSILON] = EPSILON
        self._alpha_i = self._alpha_i/np.sum(self._alpha_i)
        # print("alpha")
        # print(self._alpha_i)

        # calculate p_i_k
        # self._p_i_k = []
        # for i in range(NUM_OF_CLUSTERS):
        #     self._p_i_k.append({})
        #     for k in self._corpus:
        #         #t0 = time.time()
        #         self._p_i_k[i][k] = (np.sum(self._w_t_i[:, i] * self._n_k_t[k]) + LAMBDA) / (np.sum(self._w_t_i[:, i] * np.array(self._n_t)) + LAMBDA)
        #         #self._p_i_k[i][k] = (sum([self._w_t_i[t][i] * self._n_t_k[t][k] for t in range(self._N)]) + LAMBDA) / (sum([self._w_t_i[t][i] * self._n_t[t] for t in range(self._N)]) + LAMBDA)
        #         #t1 = time.time()
        #         #DEBUG_PRINT("time : " + str(t1-t0))
        #         #DEBUG_PRINT("finished p_i_k for i = " + str(i) + ", k = " + str(k))
        #     DEBUG_PRINT("finished p_i_k for i = " + str(i))

        for i in range(NUM_OF_CLUSTERS):
            #t0 = time.time()
            self._p_i_k[i] = np.transpose((np.matmul(self._n_k_t_mat, self._w_t_i[:, i].reshape(self._N,1)) + LAMBDA) / (np.dot(self._w_t_i[:, i],np.array(self._n_t)) + len(self._corpus)*LAMBDA))
            #self._p_i_k[i][k] = (sum([self._w_t_i[t][i] * self._n_t_k[t][k] for t in range(self._N)]) + LAMBDA) / (sum([self._w_t_i[t][i] * self._n_t[t] for t in range(self._N)]) + LAMBDA)
            #t1 = time.time()
            #DEBUG_PRINT("time : " + str(t1-t0))
            #DEBUG_PRINT("finished p_i_k for i = " + str(i) + ", k = " + str(k))
            DEBUG_PRINT("finished p_i_k for i = " + str(i))


    def _calc_e_step(self):
        """
        EM algorithm E step
        """

        # calculate z_i
        z_i = self._calc_all_z_i()
        #print(z_i)
        DEBUG_PRINT("z_i calculated")

        # calculate m value
        # import pdb
        # pdb.set_trace()
        m = np.max(z_i, axis=0)
        # update w_t_i
        for t in range(self._N):
            for i in range(NUM_OF_CLUSTERS):
                if z_i[i, t] - m[t] < -K:
                    self._w_t_i[t, i] = 0
                else:
                    # calculate the sum of exponents
                    denum = 0
                    for j in range(NUM_OF_CLUSTERS):
                        #print(z_i[j, t] - m[t])
                        if z_i[j, t] - m[t] >= -K:
                            denum += np.e**(z_i[j, t]-m[t])
                    # calculate the w_t_i in [t, i] indexes
                    self._w_t_i[t, i] = float(np.e**(z_i[i, t]-m[t]))/denum

        # print("wti")
        # print(self._w_t_i)
        # print("")
        #
        # print("pik")
        # print(self._p_i_k)
        # print("")

    def calc_em_iteration(self):
        """
        Calculate EM algorithm
        """

        self._calc_e_step()
        DEBUG_PRINT("finished E step")
        self._calc_m_step()
        DEBUG_PRINT("finished M step")


    def calc_ln_likelihood_and_mean_perplexity(self):
        """
        Calculate the ln of the likelihood and mean perplexity
        :return: [ln of the likelihood, mean perplexity]
        """

        # calculate z_i (for all articles)
        z_i = self._calc_all_z_i()
        #print(z_i)
        DEBUG_PRINT("z_i calculated")
        # calculate m value (for all articles)
        m = np.max(z_i, axis=0)

        # ln likelihood calculation
        ln_likelihood = 0
        for t in range(self._N):
            # calculate the sum of exponents
            sum_of_exponents = 0
            for j in range(NUM_OF_CLUSTERS):
                if z_i[j, t] - m[t] >= -K:
                    sum_of_exponents += np.e ** (z_i[j, t] - m[t])
            # add to the ln likelihood
            ln_likelihood += m[t] + np.log(sum_of_exponents)

        mean_perplexity = np.e**(-float(ln_likelihood)/self._num_of_words_in_dataset)
        return [ln_likelihood, mean_perplexity]


    def _calc_all_z_i(self):
        """
        Calculate z_i values, for all articles
        :return: the z_i vectors (for all articles)
        """

        # create p_i_k matrix for performing fast matrix multiplication instead of multiplying items one by one
        # p_i_k_mat = np.zeros((NUM_OF_CLUSTERS, len(self._corpus)))
        # for i in range(NUM_OF_CLUSTERS):
        #     p_i_k_mat[i] = np.array([self._p_i_k[i][k] for k in self._corpus])

        # calculate z_i for all articles (a matrix) according to the algorithm (stable) using matrix mult (in order to do it fast)
        z_i = np.log(np.repeat(self._alpha_i.reshape(NUM_OF_CLUSTERS, 1), self._N, axis=1)) + np.matmul(np.log(self._p_i_k), self._n_k_t_mat) # p_i_k_mat

        # import pdb
        # pdb.set_trace()

        return z_i


    def test(self):
        """
        Perform testing to the EM algorithm result - calculate confusion matrix and accuracy for the model
        """

        # --------------------------
        # Calculate confusion matrix

        self._confusion_matrix = np.zeros((NUM_OF_CLUSTERS, len(self._topics_data)+2)) # [[0]*(len(self._topics_data)+1)]*NUM_OF_CLUSTERS
        for i in range(NUM_OF_CLUSTERS):
            self._confusion_matrix[i, 0] = i
        for t in range(self._N):
            # check which cluster article t fits most and mark in the matrix, in the columns of the topics specified for this article
            most_likely_cluster = np.argmax(self._w_t_i[t, :])
            for topic in self._topics_per_data[t]:
                self._confusion_matrix[most_likely_cluster, self._topics_data.index(topic)+1] += 1
            # update the size of the cluster
            self._confusion_matrix[most_likely_cluster, -1] += 1



        # --------------------------
        # Calculate accuracy
        correct_assignments = 0
        for t in range(self._N):
            # classify the article
            # check which cluster article t fits most and mark in the matrix, in the columns of the topics specified for this article
            most_likely_cluster = np.argmax(self._w_t_i[t, :])
            # get the dominant topic for chosen cluster
            dominant_topic = np.argmax(self._confusion_matrix[most_likely_cluster, 1:-1])
            # check if there is a correct assignments
            for topic in self._topics_per_data[t]:
                if dominant_topic == self._topics_data.index(topic):
                    # there was a hit in one of the possible topics - this is considered as a correct assignment
                    correct_assignments += 1
                    break

        # order the matrix rows according to cluster size column
        self._confusion_matrix = np.flip(self._confusion_matrix[np.argsort(self._confusion_matrix[:, -1])], axis=0)

        self._accuracy = float(correct_assignments)/self._N

    def get_confusion_matrix(self):
        """
        Get confusion matrix
        :return: the confusion matrix
        """
        return self._confusion_matrix

    def get_accuracy(self):
        """
        Get accuracy
        :return: the accuracy
        """
        return self._accuracy

    def get_vocab_size(self):
        """
        Get vocabulary size
        :return: the vocabulary size
        """
        return len(self._corpus)



# main method of the program
def main(argv):
    # command line arguments
    dev_set_filename = argv[0]
    topics_filename = argv[1]

    # read data
    r = Reader(dev_set_filename, topics_filename)

    # parse data
    r.parse()
    dev_set_data, topics_per_data, topics_data = r.get()

    # init processor
    d = DataProcessor(dev_set_data, topics_per_data, topics_data)

    # ---------------------
    #   TRAINING PROCESS
    # ---------------------

    print("Running EM algorithm training")

    # ln likelihood values list init
    ln_likelihood = []

    # mean perplexity values list init
    mean_perplexity = []

    accuracy = []

    # calculate ln likelihood
    l0, mp = d.calc_ln_likelihood_and_mean_perplexity()
    if TEST_AFTER_EACH_ITER:
        d.test()
        accuracy.append(d.get_accuracy())
    ln_likelihood.append(l0)
    mean_perplexity.append(mp)
    print("initial log likelihood: " + str(l0))
    # em iteration
    d.calc_em_iteration()
    # calculate ln likelihood
    l1, mp = d.calc_ln_likelihood_and_mean_perplexity()
    if TEST_AFTER_EACH_ITER:
        d.test()
        accuracy.append(d.get_accuracy())
    mean_perplexity.append(mp)
    print("log likelihood: " + str(l1))
    diff10 = l1 - l0
    # em iteration
    d.calc_em_iteration()
    # calculate ln likelihood
    l2, mp = d.calc_ln_likelihood_and_mean_perplexity()
    if TEST_AFTER_EACH_ITER:
        d.test()
        accuracy.append(d.get_accuracy())
    ln_likelihood.append(l2)
    mean_perplexity.append(mp)
    print("log likelihood: " + str(l2))
    diff21 = l2 - l1

    # we stop when the difference between ln likelihhood in the last 3 iterations is small enough, i.e. the log likelihood is converging to  a limit
    while abs(diff10) > EM_THRESHOLD or abs(diff21) > EM_THRESHOLD:
        # em iteration
        d.calc_em_iteration()
        l0 = l1
        l1 = l2
        l2, mp = d.calc_ln_likelihood_and_mean_perplexity()
        if TEST_AFTER_EACH_ITER:
            d.test()
            accuracy.append(d.get_accuracy())
        ln_likelihood.append(l2)
        mean_perplexity.append(mp)
        print("log likelihood: " + str(l2))
        diff10 = l1 - l0
        diff21 = l2 - l1

    # plot
    plt.plot(np.array(ln_likelihood))
    plt.savefig('ln_likelihood.png')
    plt.clf()
    plt.plot(np.array(mean_perplexity))
    plt.savefig('mean_perplexity.png')
    if CREATE_ACCURACY_GRAPH:
        plt.clf()
        plt.plot(np.array(accuracy))
        plt.savefig('accuracy.png')


    # ---------------------
    #   Final Testing PROCESS
    # ---------------------

    # run testing
    print("Testing - creating confusion matrix and calculating accuracy")
    d.test()

    # print all
    print("K = " + str(K))
    print("LAMBDA = " + str(LAMBDA))
    print("Vocabulary size = " + str(d.get_vocab_size()))
    print("confusion matrix:")
    print(d.get_confusion_matrix())
    print("accuracy:")
    print(d.get_accuracy())

    # writing confusion matrix into csv file
    if CREATE_CONFUSION_MATRIX_CSV:
        np.savetxt("confusion_matrix.csv", d.get_confusion_matrix(), delimiter=",")


# In order to run the main method
if __name__ == "__main__":
    main(sys.argv[1:])









