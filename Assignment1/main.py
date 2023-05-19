import numpy as np
from scipy.stats import spearmanr as spear
import time

# Vocabulary paths
VOCAB_PATH = './cs402_assn1/vocab-3k.txt'
WORDSIM_PATH = './cs402_assn1/vocab-wordsim.txt'

# Corpus path
CORPUS_PATH = 'wiki-0.1percent.txt'

# Similarity dataset paths
MEN_PATH = './cs402_assn1/men.txt'
SIMLEX_PATH = './cs402_assn1/simlex-999.txt'


def load_vocab(wordsim=True):
    try:
        PATH = WORDSIM_PATH if wordsim else VOCAB_PATH
        vocab = np.loadtxt(PATH, dtype=str)
        return vocab
    except Exception as e:
        raise FileNotFoundError(e)


def load_simdataset(men=True):
    try:
        PATH = MEN_PATH if men else SIMLEX_PATH
        pairs = np.loadtxt(PATH, dtype=str, skiprows=1, usecols=(0, 1))
        corr = np.loadtxt(PATH, dtype=float, skiprows=1, usecols=(2))
        return pairs, corr
    except Exception as e:
        raise FileNotFoundError(e)


def cosine_similarity(C, V, men=True):
    pairs, corr = load_simdataset(men)
    M = np.zeros_like(corr, dtype=float)
    sqr_len = C.shape[1]**2
    for i, pair in enumerate(pairs):
        if pair[0] in V and pair[1] in V:
            idx_1 = np.where(V == pair[0])[0][0]
            idx_2 = np.where(V == pair[1])[0][0]
            M[i] = np.dot(C[idx_1], C[idx_2])/sqr_len
    return M


def EVALWS(M):
    VX = load_vocab(True)
    VY = load_vocab(False)
    men = cosine_similarity(M, VX, True)
    simlex = cosine_similarity(M, VX, False)
    _, men_corr = load_simdataset(True)
    _, simlex_corr = load_simdataset(False)
    print("Correlation for MEN")
    print(spear(men_corr, men))
    print("Correlation for SIMLEX")
    print(spear(simlex_corr, simlex))


def nearest_neighbors(C, word, num=10):
    V = load_vocab(False)
    check = np.where(V == word)[0]
    if check.shape[0] == 0:
        raise AssertionError("No such word in vocabulary")
    idx = check[0]
    length = V.shape[0]
    sqr_len = length**2
    L = np.empty_like(V, dtype=float)
    for i in range(length):
        L[i] = np.dot(C[idx], C[i])/sqr_len
    D = dict()
    max_idx = L.argmax(0)
    L[max_idx] = -np.inf
    for i in range(num):
        max_idx = L.argmax(0)
        D[V[max_idx]] = L[max_idx]
        L[max_idx] = -np.inf
    return D


def print_dict(D):
    for key in D:
        tab = '\t' if len(key) > 8 else '\t\t'
        print(f"{str(key)}{tab}: {float(D[key])}")


class CorpusDataset:
    def __init__(self, chunk_size):
        self.corpus_size = 0
        self.chunk_size = chunk_size
        self.skip = 0
        self.empty = False
        self.switch = True
        self.left = None
        self.right = self._load_data()

    def is_empty(self):
        return self.empty

    def size(self):
        return self.corpus_size

    def load_chunk(self):
        if self.is_empty():
            raise RuntimeError("No chunks to load")
        else:
            if self.switch:
                self.left = self._load_data()
                self.switch = False
                return self.right
            else:
                self.right = self._load_data()
                self.switch = True
                return self.left

    def _load_data(self):
        try:
            corpus = np.genfromtxt(
                CORPUS_PATH, dtype=str, delimiter='\n', skip_header=self.skip, max_rows=self.chunk_size, encoding='utf8')
            if corpus.shape[0] == 0:
                self.empty = True
            self.skip += self.chunk_size
            self.corpus_size += corpus.shape[0]
            return corpus
        except Exception as e:
            raise FileNotFoundError(e)


class Counter:
    def __init__(self, window=3, chunk_size=100000, flag=True):
        self.START = 0
        self.END = 1
        self.WINDOW = window
        self.CORPUS_SIZE = 0
        self.CHUNK_SIZE = chunk_size
        self.CORPUS = CorpusDataset(self.CHUNK_SIZE)
        self.VX = load_vocab(flag)
        self.VY = load_vocab(False)
        self.C = np.zeros((self.VX.shape[0], self.VY.shape[0]), dtype=float)
        self.C_PMI = None
        self.VX_COUNT = np.zeros_like(self.VX, dtype=int)
        self.VY_COUNT = np.zeros_like(self.VY, dtype=int)
        self.check_pmi = False

    def count_raw(self):
        self.check_pmi = True
        chunk_num = 0
        while not self.CORPUS.is_empty():
            chunk = self.CORPUS.load_chunk()
            chunk_num += 1
            print(f"Chunk {chunk_num} is loaded")
            x = 0
            for sen in chunk:
                sentence = sen.split(' ')
                """for pos, word in enumerate(sentence):
                    found = np.where(self.VX == word)[0]
                    if found.size > 0:
                        self.VX_COUNT[found[0]] += 1
                        row = self.C[found[0]]
                        for i in range(self.WINDOW):
                            c_pos = pos + i + 1
                            if c_pos >= length:
                                row[self.END] += 1
                                break
                            c_idx = np.where(self.VY == sentence[c_pos])
                            if len(c_idx) != 0:
                                prev.append(c_idx[0])
                                self.VY_COUNT[c_idx[0]] += 1
                                row[c_idx[0]] += 1
                        for i in range(self.WINDOW + 2, 2 * self.WINDOW + 2):
                            if i > len(prev):
                                row[self.START] += 1
                                break
                            self.VY_COUNT[prev[-i]] += 1
                            row[prev[-i]] += 1"""
                """XI = []
                YI = []
                for word in sentence:
                    xi = np.where(self.VX == word)[0]
                    yi = np.where(self.VY == word)[0]
                    if xi.size > 0:
                        XI.append(xi[0])
                    if yi.size > 0:
                        YI.append(yi[0])"""

                if (x % 10000 == 0):
                    size = chunk.shape[0]
                    print(f"Chunk {chunk_num} progress: {x/size*100:0.0f}%")
                x += 1
        self.CORPUS_SIZE = self.CORPUS.size()
        return self.C

    def calculate_pmi(self):
        if self.check_pmi:
            VX_PROB = np.expand_dims(self.VX_COUNT/self.CORPUS_SIZE, 1)
            VY_PROB = np.expand_dims(self.VY_COUNT/self.CORPUS_SIZE, 0)
            VX_VY_PROB = np.matmul(VX_PROB, VY_PROB)
            VX_VY_PROB[VX_VY_PROB == 0] = 1
            C_PROB = self.C/self.CORPUS_SIZE
            self.C_PMI = np.log2(np.divide(C_PROB, VX_VY_PROB))
            self.C_PMI[self.C_PMI < 0] = 0
            return self.C_PMI
        else:
            raise AssertionError("Counting is not performed")


# 1.1 Distributional Counting (takes around 30 minutes to run)
counter_3 = Counter(3)
tic = time.perf_counter()
C_3 = counter_3.count_raw()
toc = time.perf_counter()
print(f"Timer: {toc - tic:0.0f} seconds")
EVALWS(C_3)

# 1.2 Computing PMIs (runs relativelly fast)
C_PMI_3 = counter_3.calculate_pmi()
EVALWS(C_PMI_3)
del counter_3

"""# 1.3 Experimentation (takes around 70 minutes to run)
counter_1 = Counter(1)
C_1 = counter_1.count_raw()
EVALWS(C_1)
C_PMI_1 = counter_1.calculate_pmi()
EVALWS(C_PMI_1)
del counter_1

counter_6 = Counter(6)
C_6 = counter_6.count_raw()
EVALWS(C_6)
C_PMI_6 = counter_6.calculate_pmi()
EVALWS(C_PMI_6)
del counter_6

# 1.4.1 Warm-up: Printing nearest neighbors (takes around a week to run)

counter_1 = Counter(1, flag=False)
CN_1 = counter_1.count_raw()
CN_PMI_1 = counter_1.calculate_pmi()
D_1 = nearest_neighbors(CN_PMI_1, 'monster', 10)
print_dict(D_1)
del counter_1

counter_6 = Counter(6, flag=False)
CN_6 = counter_6.count_raw()
CN_PMI_6 = counter_6.calculate_pmi()
D_6 = nearest_neighbors(CN_PMI_6, 'monster', 10)
print_dict(D_6)
del counter_6
"""
