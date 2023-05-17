# coding=utf-8

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class PolicyGradient_chatbot():
    #Creating policy gradient as reward system for RL model
    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size, n_encode_lstm_step, n_decode_lstm_step, bias_init_vector=None, lr=0.0001):
        self.dim_wordvec = dim_wordvec
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_words = n_words
        self.n_encode_lstm_step = n_encode_lstm_step
        self.n_decode_lstm_step = n_decode_lstm_step
        self.lr = lr

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random.uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

            #Defininf the LSTM Layers
            self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
            self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        #Defining the encoding vectors
        self.encode_vector_W = tf.Variable(tf.random.uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='encode_vector_W')
        self.encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b')

        #Defining the word embedding
        self.embed_word_W = tf.Variable(tf.random.uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        #Word vector
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])

        #Caption
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_decode_lstm_step+1])
        #Caption mask
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step+1])

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        #Word embedding vectors
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b ) 
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        entropies = []
        loss = 0.
        pg_loss = 0.  # Calculating the policy gradient loss

        '''Encoding Stage'''
        for i in range(0, self.n_encode_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)      #Output of 1st LSTM Layer in encoding

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)      #Output of 2nd LSTM Layer in encoding

        '''Decoding Stage'''
        for i in range(0, self.n_decode_lstm_step):
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)               #Output of 1st LSTM Layer in decoding

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)        #Output of 2nd LSTM Layer in decoding

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)       #One hot encoding

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)

            #Calculating the cross entropy loss
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)       
            cross_entropy = cross_entropy * caption_mask[:, i]
            entropies.append(cross_entropy)
            pg_cross_entropy = cross_entropy * reward[:, i]

            #Updating the total PG loss
            pg_current_loss = tf.reduce_sum(pg_cross_entropy) / self.batch_size
            pg_loss = pg_loss + pg_current_loss

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(pg_loss)

        input_tensors = {
            'word_vectors': word_vectors,
            'caption': caption,
            'caption_mask': caption_mask,
            'reward': reward
        }

        feats = {
            'entropies': entropies
        }

        return train_op, pg_loss, input_tensors, feats

    def build_generator(self):

        word_vectors = tf.Variable(tf.zeros([self.batch_size, self.n_encode_lstm_step, self.dim_wordvec]), dtype=tf.float32)

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.matmul(word_vectors_flat, self.encode_vector_W) + self.encode_vector_b
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []
        states = []

        for i in range(0, self.n_encode_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)          #Output of 1st LSTM Layer in encoding
                states.append(state1)   

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)      #Output of 2nd LSTM Layer in encoding

        for i in range(0, self.n_decode_lstm_step):

            tf.compat.v1.variable_scope().reuse_variables()

            if i == 0:
                # <bos>
                with tf.device('/cpu:0'):
                    current_embed = tf.keras.layers.Embedding(1, self.dim_wordvec)(tf.ones([self.batch_size], dtype=tf.int64))

            with tf.compat.v1.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)           #Output of 1st LSTM Layer in decoding

            with tf.compat.v1.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)            #Output of 2nd LSTM Layer in decoding

            logit_words = tf.keras.backend.dot(output2, self.embed_word_W) + self.embed_word_b
            max_prob_index = tf.argmax(logit_words, axis=1)
            generated_words.append(max_prob_index)
            probs.append(logit_words)


            with tf.device("/cpu:0"):
                current_embed = tf.keras.layers.Embedding(self.vocab_size, self.dim_wordvec)(max_prob_index)

            embeds.append(current_embed)

        feats = {
            'probs': probs,
            'embeds': embeds,
            'states': states
        }

        return word_vectors, generated_words, feats
