import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import tensorflow as tf
import tensorflow.keras as keras
from model import BertModel
from dataset import Graph_Bert_Dataset



keras.backend.clear_session()

optimizer = tf.keras.optimizers.Adam(1e-4)

medium_balanced = {'name':'Medium','num_layers': 6, 'num_heads': 8, 'd_model': 256,'path':'mgbert_hs_weights_balanced','addH':True}

arch = medium_balanced      ## small 3 4 128   medium: 6 6  256     large:  12 8 516
num_layers = arch['num_layers']
num_heads =  arch['num_heads']
d_model =  arch['d_model']
addH = arch['addH']


dff = d_model*2
vocab_size =17
dropout_rate = 0.1

model = BertModel(num_layers=num_layers,d_model=d_model,dff=dff,num_heads=num_heads,vocab_size=vocab_size)

train_dataset, test_dataset = Graph_Bert_Dataset(path='data/pubchem_200w.txt', smiles_field='CAN_SMILES', addH=addH).get_data()

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None,None), dtype=tf.float32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_step(x, adjoin_matrix,y, char_weight):
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    with tf.GradientTape() as tape:
        predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=True)
        loss = loss_function(y,predictions,sample_weight=char_weight)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss.update_state(loss)
    train_accuracy.update_state(y,predictions,sample_weight=char_weight)


# @tf.function(input_signature=train_step_signature)
# def test_step(x, adjoin_matrix,y, char_weight):
#     seq = tf.cast(tf.math.equal(x, 0), tf.float32)
#     mask = seq[:, tf.newaxis, tf.newaxis, :]
#     predictions = model(x,adjoin_matrix=adjoin_matrix,mask=mask,training=False)
#     test_accuracy.update_state(y,predictions,sample_weight=char_weight)


for epoch in range(3):
    start = time.time()
    train_loss.reset_states()

    for (batch, (x, adjoin_matrix ,y , char_weight)) in enumerate(train_dataset):
        train_step(x, adjoin_matrix, y , char_weight)

        if batch % 128 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.result()))
            print('Accuracy: {:.4f}'.format(train_accuracy.result()))
            #
            # for x, adjoin_matrix ,y , char_weight in test_dataset:
            #     test_step(x, adjoin_matrix, y , char_weight)
            # print('Test Accuracy: {:.4f}'.format(test_accuracy.result()))
            # test_accuracy.reset_states()
            train_accuracy.reset_states()

    print(arch['path'] + '/bert_weights_{}_{}.h5'.format(arch['name'], epoch+1))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Accuracy: {:.4f}'.format(train_accuracy.result()))
    model.save_weights(arch['path']+'/bert_weights_{}_{}.h5'.format(arch['name'],epoch+1))
    print('Saving checkpoint')


