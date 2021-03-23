import sys
import math
from collections import Counter

# consts
SUBMITTER_NAME = ""
SUBMITTER_ID = ""
VOCAB_SIZE = 300000
OUTPUT_LINES = 28
VERIFY_MODELS = True

# a class for reading an input file of the exercise format
class Reader:
    def __init__(self, input_filename):
        """
        Constructor
        :param input_filename: the input file name
        """
        self._input_filename = input_filename
        self._parsed_data = []

    def parse(self):
        """
        Parse data into list of string-lists (each string-list represent the words of the article in that index, in order)
        """
        with open(self._input_filename) as input_file:
            for line in input_file:
                if line[:6] != '<TRAIN' and line[:5] != '<TEST' and line != '\n':
                    self._parsed_data.append(line.split())

    def get(self):
        """
        Get parsed data
        :return: parsed data (list of string-lists)
        """
        return self._parsed_data


# The main class for all data processing
class DataProcessor:
    def __init__(self, dev_set_data, test_set_data, input_word):
        """
        Constructor
        :param dev_set_data:
        """
        self._dev_set_data = dev_set_data
        self._test_set_data = test_set_data
        self._input_word = input_word
        self._num_of_events_in_dev_set = 0
        self._training_set = []
        self._validation_set = []
        self._num_of_events_in_train_set = 0
        self._num_of_events_in_valid_set = 0
        self._num_of_different_events_in_train_set = 0
        self._p_mle_input_word = 0
        self._p_mle_unseen_word = 0
        self._p_lidstone_input_word = 0
        self._p_lidstone_unseen_word = 0
        self._perplexity_lidstone_lambda_0_01 = 0
        self._perplexity_lidstone_lambda_0_10 = 0
        self._perplexity_lidstone_lambda_1_00 = 0
        self._min_lidstone_perplexity = 0
        self._selected_lidstone_lambda = 0
        self._training_set_for_heldout = []
        self._heldout_set = []
        self._num_of_events_in_heldout_train_set = 0
        self._num_of_events_in_heldout_set = 0
        self._p_est_heldout = {}
        self._heldout_train_set_counting = {}
        self._perplexity_lidstone_on_test_set_lidstone_with_best_lambda = 0
        self._perplexity_lidstone_on_test_heldout = 0
        self._best_model_select = ''
        self._tr = {}
        self._Nr_values = {}
        self._f_lambda = {}
        self._f_h = {}
        self._development_set = []

    def process(self):
        """
        Main processing method, executes the flow of models training and calculations
        """

        # calculate number of events in dev set
        dev_set_all_words = []
        for entry in self._dev_set_data:
            dev_set_all_words += entry
        self._num_of_events_in_dev_set = len(dev_set_all_words)

        # calculate training set & validation set for lidstone
        self._development_set = dev_set_all_words
        self._training_set = dev_set_all_words[:int(round(0.9*self._num_of_events_in_dev_set))]
        self._validation_set = dev_set_all_words[int(round(0.9*self._num_of_events_in_dev_set)):]
        self._num_of_events_in_train_set = len(self._training_set)
        self._num_of_events_in_valid_set = len(self._validation_set)

        # calculate number of different events in the training set for lidstone
        self._num_of_different_events_in_train_set = len(list(set(self._training_set)))

        # the number of times input_word appears in the training set for lidstone
        self._num_of_input_word_in_train_set = self._training_set.count(self._input_word)

        # P(Event = input_word) the MLE of input_word based on the training set for lidstone
        self._training_set_counting = Counter(self._training_set)
        self._p_mle_input_word = self._mle_prob_est_calc(self._input_word)

        # P(Event = 'unseen-word') the MLE based on the lidstone training set for 'unseen-word' (it is what MLE
        # gives for unseen events in case 'unseen-word' is not in the training set)
        self._p_mle_unseen_word = self._mle_prob_est_calc('unseen-word')

        # P(Event = input_word) as estimated by lidstone model using lambda=0.10
        self._p_lidstone_input_word = self._lidstone_prob_est_calc(self._input_word, 0.1)

        # P(Event = 'unseen_word') as estimated by lidstone model using lambda=0.10
        self._p_lidstone_unseen_word = self._lidstone_prob_est_calc('unseen_word', 0.1)

        # The perplexity on the lidstone validation set using lambda=0.01
        self._perplexity_lidstone_lambda_0_01 = self._lidstone_perplexity_calc(0.01, self._validation_set)

        # The perplexity on the lidstone validation set using lambda=0.10
        self._perplexity_lidstone_lambda_0_10 = self._lidstone_perplexity_calc(0.1, self._validation_set)

        # The perplexity on the lidstone validation set using lambda=1.00
        self._perplexity_lidstone_lambda_1_00 = self._lidstone_perplexity_calc(1.0, self._validation_set)

        # The value of lambda that minimizes the lidstone validation set + The minimized perplexity on the validation set using the best value found for lambda
        self._min_lidstone_perplexity, self._selected_lidstone_lambda = self._lidstone_train()

        # for Held-out model training:
        # splitting the development set into a training set and a held-out set
        self._training_set_for_heldout = dev_set_all_words[:int(round(0.5 * self._num_of_events_in_dev_set))]
        self._heldout_set = dev_set_all_words[int(round(0.5 * self._num_of_events_in_dev_set)):]

        # number of events in held-out training set
        self._num_of_events_in_heldout_train_set = len(self._training_set_for_heldout)
        self._num_of_events_in_heldout_set = len(self._heldout_set)

        # create the heldout probability database (entry per r - number of event appearance)
        self._heldout_warmup()

        # calculate P(Event=input_word) as estimated by the held-out model
        self._p_heldout_input_word = self._heldout_prob_est_calc(self._input_word)

        # calculate P(Event='unseen-word') as estimated by the held-out model
        self._p_heldout_unseen_word = self._heldout_prob_est_calc('unseen-word')

        # calculate number of events in the test set
        test_set_all_words = []
        for entry in self._test_set_data:
            test_set_all_words += entry
        self._num_of_events_in_test_set = len(test_set_all_words)
        self._test_set = test_set_all_words

        # the perplexity of the test set according to the Lidstone model with the best lambda
        self._perplexity_lidstone_on_test_set_lidstone_with_best_lambda = self._lidstone_perplexity_calc(self._selected_lidstone_lambda, self._test_set)

        # the perplexity of the test set according to the heldout model
        self._perplexity_lidstone_on_test_heldout = self._heldout_perplexity_calc(self._test_set)

        # selecting the best model from lidstone with best lambda and heldout
        if self._perplexity_lidstone_on_test_set_lidstone_with_best_lambda > self._perplexity_lidstone_on_test_heldout:
            self._best_model_select = 'H'
        else:
            self._best_model_select = 'L'

        # prepare what's left to have the full table needed for output29 : f_lambda, f_h
        for i in range(10):
            self._f_lambda[i] = round(self._lidstone_prob_est_calc_by_freq(i, self._selected_lidstone_lambda)*len(self._training_set), 5)
            self._f_h[i] = round(self._heldout_prob_est_calc_by_freq(i)*len(self._training_set_for_heldout), 5)

    def _mle_prob_est_calc(self, event):
        """
        MLE probability estimator
        :param event: an event
        :return: the MLE estimated probability
        """
        return float(self._training_set_counting[event]) / len(self._training_set)

    def _lidstone_prob_est_calc(self, event, lambda_p):
        """
        Lidstone probability estimator by event
        :param event: an event
        :param lambda_p: the lambda parameter for the model
        :return: the Lidstone estimated probability
        """
        return self._lidstone_prob_est_calc_by_freq(self._training_set_counting[event], lambda_p)

    def _lidstone_prob_est_calc_by_freq(self, freq, lambda_p):
        """
        Lidstone probability estimator by frequency (counting of event)
        :param freq: the frequency for which to calculate
        :param lambda_p: the lambda parameter for the model
        :return: the Lidstone estimated probability
        """
        prob = float(freq + lambda_p) / (len(self._training_set) + lambda_p*VOCAB_SIZE)
        if prob:
            return prob
        else:
            return sys.float_info.epsilon

    def _lidstone_perplexity_calc(self, lambda_p, data_set):
        """
        Lidstone perplexity calculator
        :param lambda_p: the lambda parameter for the model
        :param data_set: the data set to calculate for
        :return: the perplexity for the Lidstone model
        """
        return 2**(-sum([math.log(self._lidstone_prob_est_calc(event, lambda_p), 2) for event in data_set])/len(data_set))

    def _lidstone_train(self):
        """
        Lidstone trainer - trains the Lidstone model - finds the lambda that minimizes the perplexity
        :return: the minimal perplexity and the lambda that gives it
        """
        lambda_p = 0
        min_perplexity = self._lidstone_perplexity_calc(lambda_p, self._validation_set)
        selected_lambda = 0
        lambda_p_index = 0
        # scan all lambda values in range [0,2] in resolution of 0.01 and search for the value that minized the perplexity
        while lambda_p <= 2:
            cur_perplexiy = self._lidstone_perplexity_calc(lambda_p, self._validation_set)
            if cur_perplexiy < min_perplexity:
                min_perplexity = cur_perplexiy
                selected_lambda = lambda_p
            lambda_p_index += 1
            lambda_p = 0.01*lambda_p_index

        return min_perplexity, selected_lambda

    def _heldout_warmup(self):
        """
        Heldout "warmup" - training the held-out model (creates all data structures and calculate probability estimation per r value)
        """
        # this is a dictionary that maps events in the train set to the number they appear
        self._heldout_train_set_counting = Counter(self._training_set_for_heldout)
        # the unique r values (each value here is an amount one or more events appears in the train set)
        r_values = list(set(self._heldout_train_set_counting.values()))
        # the Nr - number of events appearing r times per r - is simply the counting amount of each r in the heldout_train_set_counting.values()
        # since the number a particular r appears there is number of entries that gave r in heldout_train_set_counting and these entries are the events
        self._Nr_values = Counter(self._heldout_train_set_counting.values())

        # this is a dictionary that maps events in the heldout set to the number they appear
        heldout_set_counting = Counter(self._heldout_set)

        # run for all r values and calculate the probability estimation for event appearing r times in train set
        for r in r_values:
            # first calculate tr
            self._tr[r] = 0
            for event in self._heldout_train_set_counting.keys():
                # summing the counting in heldout set, of events happening r times in heldout train set
                if self._heldout_train_set_counting[event] == r:
                   try:
                       c_h_event = heldout_set_counting[event]
                   except KeyError:
                       c_h_event = 0
                   self._tr[r] += c_h_event
            self._p_est_heldout[r] = float(self._tr[r])/(self._Nr_values[r]*self._num_of_events_in_heldout_set)

        # add entry for r = 0 (events not seen in the heldout train set)
        self._Nr_values[0] = VOCAB_SIZE - len(list(set(self._training_set_for_heldout)))
        # calculting t0 by summing all events counting that appeared in the heldout set and not in it's train set
        self._tr[0] = 0
        for event in list(set(self._heldout_set)):
            if event not in self._training_set_for_heldout:
                self._tr[0] += heldout_set_counting[event]
        self._p_est_heldout[0] = float(self._tr[0])/(self._Nr_values[0]*self._num_of_events_in_heldout_set)

    def _heldout_prob_est_calc(self, event):
        """
        Held-out probability estimator by event
        :param event: an event
        :return: the Held-out estimated probability
        """
        return self._heldout_prob_est_calc_by_freq(self._heldout_train_set_counting[event])

    def _heldout_prob_est_calc_by_freq(self, freq):
        """
        Held-out probability estimator by frequency (counting of event)
        :param freq: the frequency for which to calculate
        :return: the Held-out estimated probability
        """
        return self._p_est_heldout[freq]

    def _heldout_perplexity_calc(self, data_set):
        """
        Held-out perplexity calculator
        :param data_set: the data set to calculate for
        :return: the perplexity for the Held-out model
        """
        return 2**(-sum([math.log(self._heldout_prob_est_calc(event), 2) for event in data_set])/len(data_set))

    def verify_models(self):
        """
        Verify that the model (Lidstone, Held-out) are valid (probabilities sum to 1.0)
        """

        # unique events - number of events that appeared from the vocabulary
        unique_events_list = list(set(self._development_set))

        # number of unseen words in the vocabulary
        n0 = VOCAB_SIZE - len(unique_events_list)

        #--------------------------------
        # verify Lidstone model
        #--------------------------------

        # probability of an unseen-word multiplied by n0
        px_star_mult_n0 = self._lidstone_prob_est_calc('unseen_word', self._selected_lidstone_lambda) * n0

        # sum of all probabilities of words that appeared from the vocabulary
        sum_appeared_words = sum([self._lidstone_prob_est_calc(event, self._selected_lidstone_lambda) for event in unique_events_list])

        # verify the two above measures are summed to 1
        if round(px_star_mult_n0 + sum_appeared_words, 5) != 1.0:
            print("Lidstone model error : probabilities do not sum to 1")
            exit()
        # else:
        #     print("Lidstone model is valid - probabilities sum to 1")

        # --------------------------------
        # verify Lidstone model
        # --------------------------------

        # probability of an unseen-word multiplied by n0
        px_star_mult_n0 = self._p_heldout_unseen_word * n0

        # sum of all probabilities of words that appeared from the vocabulary
        sum_appeared_words = sum([self._heldout_prob_est_calc(event) for event in unique_events_list])

        # verify the two above measures are summed to 1
        if round(px_star_mult_n0 + sum_appeared_words, 5) != 1.0:
            print("Heldout model error : probabilities do not sum to 1")
            exit()
        # else:
        #     print("Heldout model is valid - probabilities sum to 1")

    def get(self, item):
        """
        Get an item from the data processor
        :param item: name of item of the data processor
        :return: the item's value
        """
        try:
            return eval("self._" + item)
        except NameError:
            print("no such item")
            return None

    def get_table(self):
        """
        Get the table as needed for output29
        :return: the table
        """
        return self._f_lambda, self._f_h, self._Nr_values, self._tr





# main method of the program
def main(argv):
    # command line arguments
    dev_set_filename = argv[0]
    test_set_filename = argv[1]
    input_word = argv[2]
    output_filename = argv[3]

    # open output file
    output_file = open(output_filename, 'w')

    # -----------------------------------------
    # output generation

    # init output array
    output_lines = [None]*OUTPUT_LINES

    # pre-outputX
    output_file.write("#Students\t" + SUBMITTER_NAME + '\t' + SUBMITTER_ID + '\n')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Init

    # output 1 : development set file name
    output_lines[0] = [dev_set_filename]

    # output 2 : test file name
    output_lines[1] = [test_set_filename]

    # output 3 : INPUT WORD
    output_lines[2] = [input_word]

    # output 4 : output file name
    output_lines[3] = [output_filename]

    # output 5 : language vocabulary size
    output_lines[4] = [str(VOCAB_SIZE)]

    # output 6 : Puniform(Event = INPUT WORD)
    output_lines[5] = [str(1.0/VOCAB_SIZE)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Development set preprocessing

    # reading dev set
    dev_reader = Reader(dev_set_filename)
    dev_reader.parse()
    dev_set_data = dev_reader.get()

    # reading test set
    test_reader = Reader(test_set_filename)
    test_reader.parse()
    test_set_data = test_reader.get()

    # init data processor
    data_processor = DataProcessor(dev_set_data=dev_set_data, test_set_data=test_set_data, input_word=input_word)

    # process
    data_processor.process()

    # output 7 : num of events in dev set
    output_lines[6] = [str(data_processor.get("num_of_events_in_dev_set"))]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Lidstone model training

    # output 8 : num of events in validation set
    output_lines[7] = [str(data_processor.get("num_of_events_in_valid_set"))]

    # output 9 : num of events in training set
    output_lines[8] = [str(data_processor.get("num_of_events_in_train_set"))]

    # output 10 : num of different events in the lidstone training set
    output_lines[9] = [str(data_processor.get("num_of_different_events_in_train_set"))]

    # outout 11 : the number of times the event input_word appears in the lidstone training set
    output_lines[10] = [str(data_processor.get("num_of_input_word_in_train_set"))]

    # output 12 : P(Event = input_word) the MLE of input_word based on the lidstone training set
    output_lines[11] = [str(data_processor.get("p_mle_input_word"))]

    # output 13 : P(Event = 'unseen-word') the MLE based on the lidstone training set for 'unseen-word' (it is what MLE
    # gives for unseen events in case 'unseen-word' is not in the training set)
    output_lines[12] = [str(data_processor.get("p_mle_unseen_word"))]

    # output 14 : P(Event = input_word) as estimated by lidstone model using lambda=0.10
    output_lines[13] = [str(data_processor.get("p_lidstone_input_word"))]

    # output 15 : P(Event = 'unseen_word') as estimated by lidstone model using lambda=0.10
    output_lines[14] = [str(data_processor.get("p_lidstone_unseen_word"))]

    # output 16 : The perplexity on the lidstone validation set using lambda=0.01
    output_lines[15] = [str(data_processor.get("perplexity_lidstone_lambda_0_01"))]

    # output 17 : The perplexity on the lidstone validation set using lambda=0.10
    output_lines[16] = [str(data_processor.get("perplexity_lidstone_lambda_0_10"))]

    # output 18 : The perplexity on the lidstone validation set using lambda=1.00
    output_lines[17] = [str(data_processor.get("perplexity_lidstone_lambda_1_00"))]

    # output 19 : The value of lambda that minimizes the lidstone validation set
    output_lines[18] = [str(data_processor.get("selected_lidstone_lambda"))]

    # output 20 : The minimized perplexity on the lidstone validation set using the best value found for lambda
    output_lines[19] = [str(data_processor.get("min_lidstone_perplexity"))]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Heldout model training

    # output 21 : number of events in the held-out's training set
    output_lines[20] = [str(data_processor.get("num_of_events_in_heldout_train_set"))]

    # output 22 : number of events in the held-out set
    output_lines[21] = [str(data_processor.get("num_of_events_in_heldout_set"))]

    # output 23 : P(Event=input_word) as estimated by the held-out model
    output_lines[22] = [str(data_processor.get("p_heldout_input_word"))]

    # output 24 : P(Event='unseen-word') as estimated by the held-out model
    output_lines[23] = [str(data_processor.get("p_heldout_unseen_word"))]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 6. Models evaluation on test set
    if VERIFY_MODELS:
        data_processor.verify_models()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. Code debugging (model verification)

    # output 25 : total number of events in the test set
    output_lines[24] = [str(data_processor.get("num_of_events_in_test_set"))]

    # output 26 : the perplexity of the test set according to the Lidstone model with the best lambda
    output_lines[25] = [str(data_processor.get("perplexity_lidstone_on_test_set_lidstone_with_best_lambda"))]

    # output 27 : the perplexity of the test set according to the heldout model
    output_lines[26] = [str(data_processor.get("perplexity_lidstone_on_test_heldout"))]

    # output 28 : best model from lidstone with best lambda and heldout
    output_lines[27] = [str(data_processor.get("best_model_select"))]




    # print all outputs except from last (the table)
    for i, output_line in enumerate(output_lines):
        output_file.write("#Output" + str(i+1) + "\t" + "\t".join(output_lines[i]) + '\n')

    # prepare what's left to have the full table needed for output29 : f_lambda, f_h
    output_file.write("#Output29\n")
    table = data_processor.get_table()
    for i in range(10):
        output_file.write(str(i) + '\t' + str(table[0][i]) + '\t' + str(table[1][i]) + '\t' + str(table[2][i]) + '\t' + str(table[3][i]) + '\n')




# In order to run the main method
if __name__ == "__main__":
    main(sys.argv[1:])
