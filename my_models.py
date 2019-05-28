from keras.layers import Activation, Dense, Conv1D, Conv2D, Flatten, Reshape, \
    Conv2DTranspose, Dropout, Embedding, LSTM, \
    Bidirectional, RepeatVector, TimeDistributed
from keras.models import Sequential, Input, Model
from keras.utils import np_utils
from keras import backend as K

import numpy as np


def decode_sentence(table, sentence):
    decoded_sentence = [table[int(id)] for id in sentence if int(id) != 0]

    return decoded_sentence


def split_dataset(good_sentences, bad_sentences):
    train_idx = int(good_sentences.shape[0] * 0.99)
    train_mask = range(train_idx)
    test_mask = range(train_idx, good_sentences.shape[0])
    good_sentences_train = good_sentences[train_mask]
    bad_sentences_train = bad_sentences[train_mask]
    good_sentences_test = good_sentences[test_mask]
    bad_sentences_test = bad_sentences[test_mask]

    return good_sentences_train, good_sentences_test, bad_sentences_train, bad_sentences_test


def print_results(model, bad_sentences, bad_sentences_test, good_sentences_test, id_to_char_enc, batch_size):
    corrected_sentences = model.predict(bad_sentences_test, batch_size=batch_size)

    for j in range(10):
        orig = ""
        wrong = ""
        corr = ""
        print()
        for i in range(bad_sentences.shape[1]):
            # orig += id_to_char_enc[np.argmax(good_sentences_test[0][i * 75:(i + 1) * 75])]
            # corr += id_to_char_enc[np.argmax(corrected_sentences[0][i * 75:(i + 1) * 75])]

            orig_c = id_to_char_enc[np.argmax(good_sentences_test[j][i][:])]
            # wrong_c = id_to_char_enc[np.ceil(bad_sentences_test[j][i][0] * float(len(char_to_id_enc)))]
            wrong_c = id_to_char_enc[bad_sentences_test[j][i]]
            corr_c = id_to_char_enc[np.argmax(corrected_sentences[j][i][:])]

            orig += orig_c if orig_c != '<PAD>' else ""
            wrong += wrong_c if wrong_c != '<PAD>' else ""
            corr += corr_c if corr_c != '<PAD>' else ""

        print("Original sequence:", orig)
        print("Wrong sequence:", wrong)
        print("Corrected sequence:", corr)
        print()


def build_model_1(dataset):
    good_sentences, bad_sentences, char_to_id_enc, id_to_char_enc = dataset
    good_sentences_train, good_sentences_test, bad_sentences_train, bad_sentences_test = split_dataset(good_sentences,
                                                                                                       bad_sentences)

    good_sentences_train = np_utils.to_categorical(good_sentences_train, num_classes=len(char_to_id_enc))
    good_sentences_test = np_utils.to_categorical(good_sentences_test, num_classes=len(char_to_id_enc))

    """bad_sentences_train = np.reshape(bad_sentences_train,
                                     (bad_sentences_train.shape[0], bad_sentences_train.shape[1], 1))
    bad_sentences_test = np.reshape(bad_sentences_test,
                                    (bad_sentences_test.shape[0], bad_sentences_test.shape[1], 1))

    # normalize
    bad_sentences_train = bad_sentences_train / float(len(char_to_id_enc))
    bad_sentences_test = bad_sentences_test / float(len(char_to_id_enc))"""

    print("Shapes after split...")
    print(bad_sentences_train.shape)
    print(good_sentences_train.shape)
    print(bad_sentences_test.shape)
    print(good_sentences_test.shape)

    batch_size = 32

    model = Sequential()
    model.add(Embedding(input_dim=len(char_to_id_enc) + 1, output_dim=128, input_length=bad_sentences_train.shape[1]))
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh')))
    model.add(Dropout(0.1))
    model.add(Dense(good_sentences_train.shape[2]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(good_sentences_train.shape[2]))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # comment this if use Embedding
    """good_sentences_train = np.reshape(good_sentences_train,
                                      (good_sentences_train.shape[0],
                                       good_sentences_train.shape[1] * good_sentences_train.shape[2]))

    good_sentences_test = np.reshape(good_sentences_test,
                                     (good_sentences_test.shape[0],
                                      good_sentences_test.shape[1] * good_sentences_test.shape[2]))"""

    print('Train...')
    model.fit(bad_sentences_train, good_sentences_train,
              batch_size=batch_size,
              epochs=30,
              validation_data=[bad_sentences_test, good_sentences_test])

    """score, acc = model.evaluate(bad_sentences_test, good_sentences_test, batch_size=batch_size)
    print("Score:" + str(score))
    print("Accuracy: " + str(acc))

    corrected_sentences = model.predict(bad_sentences_test, batch_size=batch_size)

    # print(decode_sentence(id_to_char_enc, corrected_sentences[0]))

    print(corrected_sentences[0])
    print(good_sentences_test[0])"""

    # print some results
    print_results(model, bad_sentences, bad_sentences_test, good_sentences_test, id_to_char_enc, batch_size)


def build_model_2(dataset):
    good_sentences, bad_sentences, char_to_id_enc, id_to_char_enc = dataset

    good_sentences_reshaped = np.empty((good_sentences.shape[0], good_sentences.shape[1], 1, 1)).astype('float32')

    for i in range(good_sentences.shape[0]):
        good_sentences_reshaped[i, :, :, 0] = np.reshape(good_sentences[i], (good_sentences.shape[1], 1))

    bad_sentences_reshaped = np.empty((bad_sentences.shape[0], bad_sentences.shape[1], 1, 1)).astype('float32')

    for i in range(bad_sentences.shape[0]):
        bad_sentences_reshaped[i, :, :, 0] = np.reshape(bad_sentences[i], (bad_sentences.shape[1], 1))

    print(good_sentences_reshaped.shape)
    print(bad_sentences_reshaped.shape)

    good_sentences_train, good_sentences_test, bad_sentences_train, bad_sentences_test = split_dataset(
        good_sentences_reshaped,
        bad_sentences_reshaped)

    good_sentences_train = np_utils.to_categorical(good_sentences_train, num_classes=len(char_to_id_enc))
    good_sentences_test = np_utils.to_categorical(good_sentences_test, num_classes=len(char_to_id_enc))

    len_seq = good_sentences.shape[1]

    input_shape = (len_seq, 1, 1)
    batch_size = 128
    kernel_size = 3
    latent_dim = 64
    layer_filters = [32, 64]

    # Build the Autoencoder Model
    # First build the Encoder Model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    # Stack of Conv2D blocks
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation='relu',
                   padding='same')(x)

    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)

    print("Shape:", shape)

    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    # Build the Decoder Model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # Stack of Transposed Conv2D blocks
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=1,
                            activation='relu',
                            padding='same')(x)

    x = Conv2DTranspose(filters=good_sentences_train.shape[3],
                        kernel_size=kernel_size,
                        padding='same')(x)

    outputs = Activation('softmax', name='decoder_output')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # Autoencoder = Encoder + Decoder
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()

    autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    autoencoder.fit(bad_sentences_train,
                    good_sentences_train,
                    validation_data=(bad_sentences_test, good_sentences_test),
                    epochs=4,
                    batch_size=batch_size)

    score, acc = autoencoder.evaluate(bad_sentences_test, good_sentences_test, batch_size=batch_size)

    print("Results...")
    print("Score:" + str(score))
    print("Accuracy: " + str(acc))

    corrected_sentences = autoencoder.predict(bad_sentences_test, batch_size=batch_size)

    # print some results
    for j in range(10):
        orig = ""
        wrong = ""
        corr = ""
        print()
        for i in range(bad_sentences.shape[1]):
            # orig += id_to_char_enc[np.argmax(good_sentences_test[0][i * 75:(i + 1) * 75])]
            # corr += id_to_char_enc[np.argmax(corrected_sentences[0][i * 75:(i + 1) * 75])]

            orig_c = id_to_char_enc[np.argmax(good_sentences_test[j][i][:])]
            # wrong_c = id_to_char_enc[np.ceil(bad_sentences_test[j][i][0] * float(len(char_to_id_enc)))]
            wrong_c = id_to_char_enc[int(bad_sentences_test[j][i][0][0])]
            corr_c = id_to_char_enc[np.argmax(corrected_sentences[j][i][:])]

            orig += orig_c if orig_c != '<PAD>' else ""
            wrong += wrong_c if wrong_c != '<PAD>' else ""
            corr += corr_c if corr_c != '<PAD>' else ""

        print("Original sequence:", orig)
        print("Wrong sequence:", wrong)
        print("Corrected sequence:", corr)
        print()


def build_model_3(dataset):
    good_sentences, bad_sentences, char_to_id_enc, id_to_char_enc = dataset
    good_sentences_train, good_sentences_test, bad_sentences_train, bad_sentences_test = split_dataset(good_sentences,
                                                                                                       bad_sentences)

    good_sentences_train = np_utils.to_categorical(good_sentences_train, num_classes=len(char_to_id_enc))
    good_sentences_test = np_utils.to_categorical(good_sentences_test, num_classes=len(char_to_id_enc))

    batch_size = 64

    model = Sequential()
    model.add(Embedding(len(char_to_id_enc) + 1, 128, input_length=bad_sentences_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Conv1D(bad_sentences_train.shape[1], 3, padding='same', activation='relu', strides=1))
    model.add(Dense(good_sentences_train.shape[2]))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(bad_sentences_train, good_sentences_train,
              batch_size=batch_size,
              epochs=10,
              validation_data=(bad_sentences_test, good_sentences_test))

    score, acc = model.evaluate(bad_sentences_test, good_sentences_test, batch_size=batch_size)

    print("Results...")
    print("Score:" + str(score))
    print("Accuracy: " + str(acc))

    # print some results
    print_results(model, bad_sentences, bad_sentences_test, good_sentences_test, id_to_char_enc, batch_size)


def build_model_4(dataset):
    good_sentences, bad_sentences, char_to_id_enc, id_to_char_enc = dataset
    good_sentences_train, good_sentences_test, bad_sentences_train, bad_sentences_test = split_dataset(good_sentences,
                                                                                                       bad_sentences)

    good_sentences_train = np_utils.to_categorical(good_sentences_train, num_classes=len(char_to_id_enc))
    good_sentences_test = np_utils.to_categorical(good_sentences_test, num_classes=len(char_to_id_enc))

    # comment this if use Embedding
    """bad_sentences_train = np.reshape(bad_sentences_train,
                                     (bad_sentences_train.shape[0], bad_sentences_train.shape[1], 1))
    bad_sentences_test = np.reshape(bad_sentences_test,
                                    (bad_sentences_test.shape[0], bad_sentences_test.shape[1], 1))

    # normalize
    bad_sentences_train = bad_sentences_train / float(len(char_to_id_enc))
    bad_sentences_test = bad_sentences_test / float(len(char_to_id_enc))"""

    print(bad_sentences_train.shape)
    print(good_sentences_train.shape)

    batch_size = 32

    model = Sequential()
    model.add(Embedding(input_dim=len(char_to_id_enc) + 1, output_dim=128, input_length=bad_sentences_train.shape[1]))
    model.add(LSTM(128, activation='tanh'))
    model.add(RepeatVector(bad_sentences.shape[1]))
    model.add(LSTM(128, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(good_sentences_train.shape[2])))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    print('Train...')
    model.fit(bad_sentences_train, good_sentences_train,
              batch_size=batch_size,
              epochs=10, validation_data=[bad_sentences_test, good_sentences_test])

    # print some results
    print_results(model, bad_sentences, bad_sentences_test, good_sentences_test, id_to_char_enc, batch_size)
