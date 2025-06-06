import math, os, pickle, re
from typing import Tuple, List, Dict
from datetime import datetime, timedelta

class BayesClassifier:
    """A simple BayesClassifier implementation

    Attributes:
        pos_freqs - dictionary of frequencies of positive words
        neg_freqs - dictionary of frequencies of negative words
        pos_filename - name of positive dictionary cache file
        neg_filename - name of positive dictionary cache file
        training_data_directory - relative path to training directory
        neg_file_prefix - prefix of negative reviews
        pos_file_prefix - prefix of positive reviews
    """

    def __init__(self, stock_data, ticker):
        """Constructor initializes and trains the Naive Bayes Sentiment Classifier. If a
        cache of a trained classifier is stored in the current folder it is loaded,
        otherwise the system will proceed through training.  Once constructed the
        classifier is ready to classify input text."""
        # initialize attributes
        self.pos_freqs: Dict[str, int] = {}
        self.neg_freqs: Dict[str, int] = {}
        self.net_freqs: Dict[str, int] = {}

        if not os.path.exists(f"ai/{ticker}"):
            os.makedirs(f"ai/{ticker}")

        self.pos_filename: str = f"ai/{ticker}/pos.dat"
        self.neg_filename: str = f"ai/{ticker}/neg.dat"
        self.net_filename: str = f"ai/{ticker}/net.dat"
        self.training_data_directory: str = "wiki_data/"

        # check if both cached classifiers exist within the current directory
        if os.path.isfile(self.pos_filename) and os.path.isfile(self.neg_filename) and os.path.isfile(self.net_filename):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(self.pos_filename)
            self.neg_freqs = self.load_dict(self.neg_filename)
            self.net_freqs = self.load_dict(self.net_filename)
        else:
            print("Data files not found - running training...")
            self.train(stock_data)

    def train(self, stock_data) -> None:
        """Trains the Naive Bayes Sentiment Classifier

        Train here means generates `pos_freq/neg_freq` dictionaries with frequencies of
        words in corresponding positive/negative reviews
        """
            

        # get the list of file names from the training data directory
        # os.walk returns a generator (feel free to Google "python generators" if you're
        # curious to learn more, next gets the first value from this generator or the
        # provided default `(None, None, [])` if the generator has no values)
        _, __, files = next(os.walk(self.training_data_directory), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {self.training_data_directory}")
        
        for file in files:
            date = datetime.strptime(file.split("_")[1].replace(".txt", ""), "%Y-%m-%d") + timedelta(days=1)

            try:
                day1 = stock_data.loc[file.split("_")[1].replace(".txt", ""), "Close"]
                day2 = stock_data.loc[date.strftime("%Y-%m-%d"), "Close"]
            except:
                continue

            percent_increase = ((day2 - day1) / day1) * 100

            file_data = open("wiki_data/" + file, "r", encoding="latin1").read()

            tokens = self.tokenize(file_data)

            if percent_increase >= 1:
                self.update_dict(tokens, self.pos_freqs)
            elif -1 < percent_increase < 1:
                self.update_dict(tokens, self.net_freqs)
            else:
                self.update_dict(tokens, self.neg_freqs)


        self.save_dict(self.pos_freqs, self.pos_filename)
        self.save_dict(self.neg_freqs, self.neg_filename)
        self.save_dict(self.net_freqs, self.net_filename)

        # files now holds a list of the filenames
        # self.training_data_directory holds the folder name where these files are
        

        # stored below is how you would load a file with filename given by `fName`
        # `text` here will be the literal text of the file (i.e. what you would see
        # if you opened the file in a text editor
        # text = self.load_file(os.path.join(self.training_data_directory, fName))


        # *Tip:* training can take a while, to make it more transparent, we can use the
        # enumerate function, which loops over something and has an automatic counter.
        # write something like this to track progress (note the `# type: ignore` comment
        # which tells mypy we know better and it shouldn't complain at us on this line):
        # for index, filename in enumerate(files, 1): # type: ignore
        #     print(f"Training on file {index} of {len(files)}")
        #     <the rest of your code for updating frequencies here>


        # we want to fill pos_freqs and neg_freqs with the correct counts of words from
        # their respective reviews
        
        # for each file, if it is a negative file, update (see the Updating frequencies
        # set of comments for what we mean by update) the frequencies in the negative
        # frequency dictionary. If it is a positive file, update (again see the Updating
        # frequencies set of comments for what we mean by update) the frequencies in the
        # positive frequency dictionary. If it is neither a postive or negative file,
        # ignore it and move to the next file (this is more just to be safe; we won't
        # test your code with neutral reviews)
        

        # Updating frequences: to update the frequencies for each file, you need to get
        # the text of the file, tokenize it, then update the appropriate dictionary for
        # those tokens. We've asked you to write a function `update_dict` that will make
        # your life easier here. Write that function first then pass it your list of
        # tokens from the file and the appropriate dictionary
        

        # for debugging purposes, it might be useful to print out the tokens and their
        # frequencies for both the positive and negative dictionaries
        

        # once you have gone through all the files, save the frequency dictionaries to
        # avoid extra work in the future (using the save_dict method). The objects you
        # are saving are self.pos_freqs and self.neg_freqs and the filepaths to save to
        # are self.pos_filename and self.neg_filename

    def classify(self, text: str) -> str:
        """Classifies given text as positive, negative or neutral from calculating the
        most likely document class to which the target string belongs

        Args:
            text - text to classify

        Returns:
            classification, either positive, negative or neutral
        """
        # TODO: fill me out

        
        # get a list of the individual tokens that occur in text
        tokens = self.tokenize(text)

        # create some variables to store the positive and negative probability. since
        # we will be adding logs of probabilities, the initial values for the positive
        # and negative probabilities are set to 0
        pp = 0
        pn = 0
        pne = 0

        pl = sum(self.pos_freqs.values())
        nl = sum(self.neg_freqs.values())
        pnl = sum(self.net_freqs.values())

        for token in tokens:
            pp += math.log((self.pos_freqs.get(token, 0) + 1) / pl)
            pn += math.log((self.neg_freqs.get(token, 0) + 1) / nl)
            pne += math.log((self.net_freqs.get(token, 0) + 1) / pnl)

        if pp > pn and pp > pne: return (1, 0, 0)
        elif pne >= pp and pne >= pn: return (0, 1, 0)
        else: return (0, 0, 1)

        # get the sum of all of the frequencies of the features in each document class
        # (i.e. how many words occurred in all documents for the given class) - this
        # will be used in calculating the probability of each document class given each
        # individual feature
        

        # for each token in the text, calculate the probability of it occurring in a
        # postive document and in a negative document and add the logs of those to the
        # running sums. when calculating the probabilities, always add 1 to the numerator
        # of each probability for add one smoothing (so that we never have a probability
        # of 0)


        # for debugging purposes, it may help to print the overall positive and negative
        # probabilities
        

        # determine whether positive or negative was more probable (i.e. which one was
        # larger)
        

        # return a string of "positive" or "negative"

    def load_file(self, filepath: str) -> str:
        """Loads text of given file

        Args:
            filepath - relative path to file to load

        Returns:
            text of the given file
        """
        with open(filepath, "r", encoding='utf8') as f:
            return f.read()

    def save_dict(self, dict: Dict, filepath: str) -> None:
        """Pickles given dictionary to a file with the given name

        Args:
            dict - a dictionary to pickle
            filepath - relative path to file to save
        """
        print(f"Dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> Dict:
        """Loads pickled dictionary stored in given file

        Args:
            filepath - relative path to file to load

        Returns:
            dictionary stored in given file
        """
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def tokenize(self, text: str) -> List[str]:
        """Splits given text into a list of the individual tokens in order

        Args:
            text - text to tokenize

        Returns:
            tokens of given text in order
        """
        tokens = []
        token = ""
        for c in text:
            if (
                re.match("[a-zA-Z0-9]", str(c)) != None
                or c == "'"
                or c == "_"
                or c == "-"
            ):
                token += c
            else:
                if token != "":
                    tokens.append(token.lower())
                    token = ""
                if c.strip() != "":
                    tokens.append(str(c.strip()))

        if token != "":
            tokens.append(token.lower())
        return tokens

    def update_dict(self, words: List[str], freqs: Dict[str, int]) -> None:
        """Updates given (word -> frequency) dictionary with given words list

        By updating we mean increment the count of each word in words in the dictionary.
        If any word in words is not currently in the dictionary add it with a count of 1.
        (if a word is in words multiple times you'll increment it as many times
        as it appears)

        Args:
            words - list of tokens to update frequencies of
            freqs - dictionary of frequencies to update
        """
        # TODO: your work here

        for word in words: freqs[word] = freqs.get(word, 0) + 1