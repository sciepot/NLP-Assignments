#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

typedef double Data;
const std::string VOCAB_PATH = "./cs402_assn1/vocab-25k.txt";
const std::string WORDSIM_PATH = "./cs402_assn1/vocab-wordsim.txt";
const std::string CORPUS_PATH = "wiki-1percent.txt";
const std::string MEN_PATH = "./cs402_assn1/men.txt";
const std::string SIMLEX_PATH = "./cs402_assn1/simlex-999.txt";

void load_vocab(std::vector<std::string> *vocab, bool wordsim = true)
{
    std::string PATH = wordsim ? WORDSIM_PATH : VOCAB_PATH;
    std::ifstream vocab_file(PATH);
    if (!vocab_file.is_open())
    {
        std::cout << PATH << " not found" << std::endl;
        exit(1);
    }
    std::string buffer;
    while (getline(vocab_file, buffer))
    {
        vocab->push_back(buffer);
    }
    vocab_file.close();
    std::sort(vocab->begin(), vocab->end());
    return;
}

std::vector<std::string> split(const std::string &buffer)
{
    std::vector<std::string> buffer_split;
    buffer_split.push_back("");
    for (int i = 0; i < buffer.length(); i++)
    {
        if (buffer[i] == '\t' || buffer[i] == ' ')
        {
            buffer_split.push_back("");
            continue;
        }
        buffer_split.back().push_back(buffer[i]);
    }
    return buffer_split;
}

int find_index(const std::string &word, const std::vector<std::string> &vocab)
{
    int low = 0;
    int high = vocab.size();
    int mid = (low + high) / 2;
    while (low + 1 < high)
    {
        if (word == vocab[mid])
        {
            return mid;
        }
        else if (word < vocab[mid])
        {
            high = mid;
            mid = (low + high) / 2;
        }
        else
        {
            low = mid;
            mid = (low + high) / 2;
        }
    }
    return -1;
}

void load_simdataset(std::vector<std::vector<std::string>> *pairs, std::vector<Data> *corrs, bool men = true, bool skip = true)
{
    std::string PATH = men ? MEN_PATH : SIMLEX_PATH;
    std::ifstream sim_file(PATH);

    if (!sim_file.is_open())
    {
        std::cout << PATH << " not found" << std::endl;
        exit(1);
    }
    std::string buffer;
    std::vector<std::string> buffer_split;
    std::vector<std::string> pair;
    if (skip)
        std::getline(sim_file, buffer);
    while (std::getline(sim_file, buffer))
    {

        buffer_split = split(buffer);
        pair.push_back(buffer_split[0]);
        pair.push_back(buffer_split[1]);

        pairs->push_back(pair);
        corrs->push_back(std::stod(buffer_split[2]));
    }
    sim_file.close();
    return;
}

std::vector<Data> cosine_similarity(const std::vector<std::vector<Data>> &C, const std::vector<std::string> &V, bool men = true)
{
    std::vector<std::vector<std::string>> pairs;
    std::vector<Data> corrs;
    load_simdataset(&pairs, &corrs);
    std::vector<Data> my_coors;
    Data sqr_len = static_cast<Data>(std::sqrt(C[0].size()));
    int idx_1;
    int idx_2;
    for (int i = 0; i < pairs.size(); i++)
    {
        idx_1 = find_index(pairs[i][0], V);
        idx_2 = find_index(pairs[i][1], V);
        if (idx_1 > 0 && idx_2 > 0)
        {
            for (int x = 0; x < pairs.size(); x++)
            {
                my_coors.push_back(C[idx_1][x] * C[idx_2][x] / sqr_len);
            }
        }
    }
    return my_coors;
}

class Counter
{
private:
    const int START = 0;
    const int END = 1;
    int WINDOW;
    Data CORPUS;
    std::vector<std::string> VX;
    std::vector<std::string> VY;
    std::vector<std::vector<Data>> C;
    std::vector<std::vector<Data>> C_PMI;
    std::vector<Data> VX_COUNT;
    std::vector<Data> VY_COUNT;
    bool check_pmi = false;
    int num_rows;
    int num_cols;

public:
    Counter(int window, bool flag = true) : WINDOW(window)
    {
        load_vocab(&VX, flag);
        load_vocab(&VY, false);
        num_rows = VX.size();
        num_cols = VY.size();
        C.resize(num_rows);
        for (auto &row : C)
            row.resize(num_cols);
        C_PMI.resize(num_rows);
        for (auto &row : C_PMI)
            row.resize(num_cols);
        VX_COUNT.resize(num_rows);
        VY_COUNT.resize(num_cols);
    }

    std::vector<std::vector<Data>> *count_raw()
    {
        check_pmi = true;
        std::ifstream corpus_file(CORPUS_PATH);
        if (!corpus_file.is_open())
        {
            std::cout << CORPUS_PATH << " not found" << std::endl;
            exit(1);
        }
        std::string buffer;
        int count = 0;
        while (getline(corpus_file, buffer))
        {
            CORPUS++;
            std::vector<std::string> sentence = split(buffer);
            int pos = 0;
            int size = sentence.size();
            for (std::string &word : sentence)
            {
                int found = find_index(word, VX);
                if (found > 0)
                {
                    VX_COUNT[found]++;
                    std::vector<Data> &row = C[found];
                    for (int i = (pos <= WINDOW ? 0 : pos - WINDOW); i < (size <= pos + WINDOW ? size : pos + WINDOW); i++)
                    {
                        if (i == pos)
                        {
                            continue;
                        }
                        int next = find_index(sentence[i], VY);
                        if (next > 0)
                        {
                            VY_COUNT[next]++;
                            row[next]++;
                        }
                    }
                }
                pos++;
            }

            count++;
            if (count % 100000 == 0)
                std::cout << "Progress bar: " << count / 10000 << "%" << std::endl;
        }
        corpus_file.close();
        return &C;
    }

    std::vector<std::vector<Data>> *calculate_pmi()
    {
        Data temp;
        if (check_pmi)
        {
            for (int x = 0; x < num_rows; x++)
            {
                for (int y = 0; y < num_cols; y++)
                {
                    temp = (VX_COUNT[x] * VY_COUNT[y]) / CORPUS;
                    if (temp == 0)
                        temp = 1;
                    C_PMI[x][y] = std::log2(C[x][y] / temp);
                }
            }
            return &C_PMI;
        }
        else
        {
            std::cout << "Counting is performed\n";
            return nullptr;
        }
    }
};

int main()
{
    Counter counter(1, false);
    std::vector<std::vector<Data>> *C = counter.count_raw();
    std::vector<std::vector<Data>> *C_PMI = counter.calculate_pmi();

    return 0;
}
