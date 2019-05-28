import sys
import pandas as pd
import codecs
import random
import string
import nltk.data


def process_csv(doc):
    print("->Processing [%s]...\n" % doc)
    pass


def process_txt(doc):
    print("-> Processing [%s]...\n" % doc)

    file = codecs.open(doc, encoding='utf-8')

    text = file.read()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    # print(len(sentences))
    # print('\n********************\n'.join(sentences))

    return sentences


def alter_sentences(sentences):
    ids = ["sentence_%s" % str(i) for i in range(1, len(sentences) + 1)]
    dataset = pd.DataFrame()
    dataset["id"] = ids
    dataset["good_sentence"] = sentences

    bad_sentences = []
    # generate one mistake per sentence
    for sentence in sentences:
        choice = random.uniform(0, 1)

        new_sentence = []

        altered = False
        # a word can be misspelled with chance choice
        if choice <= 0.25 and not altered:
            p = 0.1
            words = sentence.split(' ')

            for word in words:
                if word:
                    outcome = random.random()
                    if outcome <= p:
                        altered = True
                        ix = random.choice(range(len(word)))
                        new_word = ''.join(
                            [word[w] if w != ix else random.choice(string.ascii_letters) for w in range(len(word))])
                        new_sentence.append(new_word)
                    else:
                        new_sentence.append(word)

            new_sentence = ' '.join([w for w in new_sentence])

        # remove word "pe" if "pe care" exists
        if not altered:
            p = random.uniform(0, 1)
            if "pe care" in sentence:
                if p <= 0.3:
                    new_sentence = sentence.replace("pe care", "care")

        # replace word "decât" with "ca" and viceversa
        if not altered:
            p = random.uniform(0, 1)
            if "decat" in sentence:
                if p <= 0.4:
                    new_sentence = sentence.replace("decât", "ca")
            elif "ca" in sentence:
                if p <= 0.4:
                    new_sentence = sentence.replace("ca", "decât")

        # remove char "."
        if not altered:
            p = random.uniform(0, 1)
            if "." in sentence:
                if p <= 0.1:
                    new_sentence = sentence.replace(".", "")

        # if still not altered, remove few chars
        if not altered:
            ix = random.choice(range(len(sentence)))
            new_sentence = sentence[:ix] + sentence[ix:]

        bad_sentences.append(new_sentence)

    dataset["bad_sentence"] = bad_sentences

    dataset.to_csv("dataset.csv", index=False, encoding='utf-8')

    return dataset


def main():
    doc = "data/doc1.txt"

    ext = doc.split(".")[1]

    print(ext)

    if ext == "csv":
        sentences = process_csv(doc)
    elif ext == "txt":
        sentences = process_txt(doc)
        alter_sentences(sentences)


if __name__ == '__main__':
    main()
