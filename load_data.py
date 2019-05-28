import csv
import numpy as np
from keras.preprocessing import sequence

from my_models import build_model_1, build_model_2, build_model_3, build_model_4

PAD, START, END = '<PAD>', '<START>', '<END>'


def find_wrong_word(good_sentence, bad_sentence):
    wrong_word = None

    good_sentence_split = good_sentence.split(" ")
    bad_sentence_split = bad_sentence.split(" ")

    for word in bad_sentence_split:
        if word not in good_sentence_split:
            wrong_word = word
            break

    return wrong_word


def add_special_symbols(sequences, special_symbols):
    new_sequences = []
    for i in range(len(sequences)):
        new_sequences.append(np.concatenate(
            (np.concatenate(([special_symbols[START]], np.array(sequences[i])), axis=0), [special_symbols[END]]),
            axis=0))

    return np.array(new_sequences)


def load_dataset(filename):
    """
    Character encoding.
    :param filename:
    :return:
    """
    print("\n####### DATA LOADING #######")

    good_sentences = []
    bad_sentences = []
    char_to_id_enc = {}
    id_to_char_enc = {}
    id = 1
    max_len = -1

    with open(filename, encoding='utf-8') as csv_data_file:
        csv_reader = csv.reader(csv_data_file)

        for row in csv_reader:
            good_sentence = row[0].lower()
            bad_sentence = row[1].lower()

            for char in good_sentence + bad_sentence:
                if char not in char_to_id_enc:
                    char_to_id_enc[char] = id
                    id_to_char_enc[id] = char
                    id += 1

            good_sentence_enc = [char_to_id_enc[char] for char in good_sentence]
            bad_sentence_enc = [char_to_id_enc[char] for char in bad_sentence]

            max_len = max(max_len, max(len(good_sentence_enc), len(bad_sentence_enc)))

            good_sentences.append(good_sentence_enc)
            bad_sentences.append(bad_sentence_enc)

    print("There are %d rows to process in this dataset.\n" % len(good_sentences))

    print("Id:", id)

    special_sym = {START: id, END: id + 1}

    print("Good sentences:\n%s" % [id_to_char_enc[idd] for idd in good_sentences[:1][0]])
    print("Bad sentences:\n%s" % [id_to_char_enc[idd] for idd in bad_sentences[:1][0]])

    print("Maximum length of all sequences:", max_len)

    good_sentences = sequence.pad_sequences(good_sentences, maxlen=max_len)
    bad_sentences = sequence.pad_sequences(bad_sentences, maxlen=max_len)

    good_sentences = add_special_symbols(good_sentences, special_sym)
    bad_sentences = add_special_symbols(bad_sentences, special_sym)

    # print("Padded Good sentences:\n%s" % good_sentences[:3])
    # print("Padded Bad sentences:\n%s" % bad_sentences[:3])

    # update the encoding
    char_to_id_enc[PAD] = 0
    id_to_char_enc[0] = PAD

    char_to_id_enc[START] = special_sym[START]
    id_to_char_enc[special_sym[START]] = START

    char_to_id_enc[END] = special_sym[END]
    id_to_char_enc[special_sym[END]] = END

    print('good_sentences shape:', good_sentences.shape)
    print('bad_sentences shape:', bad_sentences.shape)

    print("Alphabet:", char_to_id_enc)
    print("Alphabet:", id_to_char_enc)

    print("\n-> DONE [DATA LOADING].\n")

    return good_sentences, bad_sentences, char_to_id_enc, id_to_char_enc


def load_dataset_1(filename):
    """
    Word encoding.
    :param filename:
    :return:
    """
    print("\n####### DATA LOADING #######")

    good_sentences = []
    bad_sentences = []
    word_to_id_enc = {}
    id_to_word_enc = {}
    id = 1
    max_len = -1

    with open(filename, encoding='utf-8') as csv_data_file:
        csv_reader = csv.reader(csv_data_file)

        for row in csv_reader:
            good_sentence = row[0].lower()
            bad_sentence = row[1].lower()

            good_sentence_words = good_sentence.split()
            bad_sentence_words = bad_sentence.split()

            for word in good_sentence_words + bad_sentence_words:
                if word not in word_to_id_enc:
                    word_to_id_enc[word] = id
                    id_to_word_enc[id] = word
                    id += 1

            good_sentence_enc = [word_to_id_enc[word] for word in good_sentence_words]
            bad_sentence_enc = [word_to_id_enc[word] for word in bad_sentence_words]

            max_len = max(max_len, max(len(good_sentence_enc), len(bad_sentence_enc)))

            good_sentences.append(good_sentence_enc)
            bad_sentences.append(bad_sentence_enc)

    print("There are %d rows to process in this dataset.\n" % len(good_sentences))

    print("Good sentences:\n%s" % good_sentences[:3])
    print("Bad sentences:\n%s" % bad_sentences[:3])

    print("Maximum length of all sequences:", max_len)

    good_sentences = sequence.pad_sequences(good_sentences, maxlen=max_len)
    bad_sentences = sequence.pad_sequences(bad_sentences, maxlen=max_len)

    print('good_sentences shape:', good_sentences.shape)
    print('bad_sentences shape:', bad_sentences.shape)

    # update the encoding
    word_to_id_enc[PAD] = 0
    id_to_word_enc[0] = PAD

    print("\n-> DONE [DATA LOADING].\n")

    return good_sentences, bad_sentences, word_to_id_enc, id_to_word_enc


def process_dataset(dataset):
    print("\n####### DATA PROCESSING #######")

    good_sentences, bad_sentences, char_to_id_enc, id_to_char_enc = dataset

    print("\n-> DONE [DATA PROCESSING].\n")

    return good_sentences, bad_sentences, char_to_id_enc, id_to_char_enc


def main():
    filename = "data/typos.csv"
    dataset = load_dataset(filename)
    build_model_1(dataset)


if __name__ == '__main__':
    main()
