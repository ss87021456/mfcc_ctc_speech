#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time

import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range

try:
    from python_speech_features import mfcc
except ImportError:
    print("Failed to import python_speech_features.\n Try pip install python_speech_features.")
    raise ImportError
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences

'''
def fake_data(num_examples, num_features, num_labels, min_size = 10, max_size=100):

    # Generating different timesteps for each fake data
    timesteps = np.random.randint(min_size, max_size, (num_examples,))

    # Generating random input
    inputs = np.asarray([np.random.randn(t, num_features).astype(np.float32) for t in timesteps])
    #print(inputs[0].shape)
    # Generating random label, the size must be less or equal than timestep in order to achieve the end of the lattice in max timestep
    labels = np.asarray([np.random.randint(0, num_labels, np.random.randint(1, inputs[i].shape[0], (1,))).astype(np.int64) for i, _ in enumerate(timesteps)])

    return inputs, labels
'''




# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 13
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 500
num_hidden = 100
num_layers = 1
batch_size = 10
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 50
num_batches_per_epoch = int(num_examples/batch_size)

print("===============train inputs generate==============")

temp_inputs = []
temp_labels = []
test_temp_inputs = []

dir_path = '../cut_wav/'
label_path = '../label.txt'

with open(label_path) as file:
    my_list = file.readlines()
    my_list = [x.strip('\n') for x in my_list]
    for line in my_list:
        original = ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '')
        targets = original.replace(' ', '  ')
        targets = targets.split(' ')

        # Adding blank label
        targets = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in targets])

        # Transform char into index
        targets = np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX
                              for x in targets])
        temp_labels.append(targets)


for i in range(1,51,1):
    file_path = dir_path + "cut_" + str(i) + ".wav"
    fs, audio = wav.read(file_path)
    print(audio.shape)
    #print("sec:",audio.shape[0]/16000.0,"s")
    audio = np.resize(audio,[len(audio),1])
    inputs = mfcc(audio, samplerate=fs)

    inputs = (inputs - np.mean(inputs))/np.std(inputs)
    print(inputs.shape)
    temp_inputs.append(inputs)
    print (i,"/",50)

#exit()

for i in range(51,71,1):
    file_path = dir_path + "cut_" + str(i) + ".wav"
    fs, audio = wav.read(file_path)
    audio = np.resize(audio,[len(audio),1])
    test_inputs = mfcc(audio, samplerate=fs)

    test_inputs = (test_inputs - np.mean(test_inputs))/np.std(test_inputs)
    test_temp_inputs.append(test_inputs)
    print (i,"/",21)

train_inputs = np.asarray([element for element in temp_inputs])
train_targets = np.asarray([element for element in temp_labels])

test_inputs = np.asarray([element for element in test_temp_inputs])
test_targets = train_targets[51:]
train_targets = train_targets[:50]

print(train_inputs.shape)
print(train_targets.shape)
print(test_inputs.shape)
print(test_targets.shape)

print("===============train inputs generate done ==============")


#inputs_fake, labels_fake = fake_data(num_examples, num_features, num_classes - 1)

#print(labels_fake.shape)
#print(type(labels_fake[0]))
#print(type(train_targets[0]))

# THE MAIN CODE!

graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell4
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    #cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    #cell = tf.contrib.rnn.GRUCell(num_hidden, activation=tf.tanh)
    cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
    # Stacking rnn cells
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
                                        state_is_tuple=True)

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.random_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

    saver = tf.train.Saver()


with tf.Session(graph=graph) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()


    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for batch in range(num_batches_per_epoch):

            # Getting the index
            indexes = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
            #print("indexes!!!",indexes)
            batch_train_inputs = train_inputs[indexes]
            # Padding input to max_time_step of this batch
            batch_train_inputs, batch_train_seq_len = pad_sequences(batch_train_inputs)
            print(batch_train_inputs.shape)
            print(batch_train_seq_len)
            # Converting to sparse representation so as to to feed SparseTensor input
            batch_train_targets = sparse_tuple_from(train_targets[indexes])
           # indices, values, shape = sparse_tuple_from(train_targets[indexes])

           # print("indices",indices)
           # print("values",values)
           # print("shape",shape)

           # exit()

            feed = {inputs: batch_train_inputs,
                    targets: batch_train_targets,
                    seq_len: batch_train_seq_len}

            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        # Shuffle the data
        shuffled_indexes = np.random.permutation(num_examples)
        train_inputs = train_inputs[shuffled_indexes]
        train_targets = train_targets[shuffled_indexes]
    
        # Metrics mean
        train_cost /= num_examples
        train_ler /= num_examples
    
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler, time.time() - start))

        # Decoding all at once. Note that this isn't the best way
        
        # Padding input to max_time_step of this batch
        batch_train_inputs, batch_train_seq_len = pad_sequences(train_inputs)
    
        # Converting to sparse representation so as to to feed SparseTensor input
        batch_train_targets = sparse_tuple_from(train_targets)
    
        feed = {inputs: batch_train_inputs,
                targets: batch_train_targets,
                seq_len: batch_train_seq_len
                }
    
        # Decoding
        if curr_epoch % 25 == 0:
            save_path = saver.save(session, "../checkpoint/model.ckpt")
            print("Model saved in file: %s" % save_path)
            d = session.run(decoded[0], feed_dict=feed)
            dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)
        
            for i, seq in enumerate(dense_decoded):
        
                if i == 0:
                    seq = [s for s in seq if s != -1]
            
                    str_decoded = ''.join([chr(x) for x in np.asarray(seq) + FIRST_INDEX])
                    # Replacing blank label to none
                    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
                    # Replacing space label to space
                    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
            
                    temp = ''.join([chr(x) for x in np.asarray(train_targets[i]) + FIRST_INDEX])
                    # Replacing blank label to none
                    temp = temp.replace(chr(ord('z') + 1), '')
                    # Replacing space label to space
                    temp = temp.replace(chr(ord('a') - 1), ' ')
            
                    #print('Sequence %d' % i)
                    print('Original:%s' % temp)
                    print('Decoded: %s' % str_decoded)

    batch_test_inputs, batch_test_seq_len = pad_sequences(test_inputs)
    
    # Converting to sparse representation so as to to feed SparseTensor input
    batch_test_targets = sparse_tuple_from(test_targets)
    
    feed_test = {inputs: batch_test_inputs,
            targets: batch_test_targets,
            seq_len: batch_test_seq_len
            }
    # Final testing
    d = session.run(decoded[0], feed_dict=feed)
    dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=session)

    for i, seq in enumerate(dense_decoded):

        seq = [s for s in seq if s != -1]
    
        str_decoded = ''.join([chr(x) for x in np.asarray(seq) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
    
        temp = ''.join([chr(x) for x in np.asarray(train_targets[i]) + FIRST_INDEX])
        # Replacing blank label to none
        temp = temp.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        temp = temp.replace(chr(ord('a') - 1), ' ')
    
        #print('Sequence %d' % i)
        print('Original:%s' % temp)
        print('Decoded: %s' % str_decoded)

