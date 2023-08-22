import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pandas as pd
import numpy as np

from dataset import M2VGraph_Classification_Dataset
from sklearn.metrics import r2_score, roc_auc_score

import os
from model import PredictModel, BertModel

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
keras.backend.clear_session()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(seed):

    task = 'nr-ahr'
    print(task)

    medium_balanced = {'name': 'Medium', 'num_layers': 6, 'num_heads': 8, 'd_model': 256, 'path': 'weights_mfgb_hs_balanced', 'addH': True}

    arch = medium_balanced  ## small 3 4 128   medium: 6 6  256     large:  12 8 516
    pretraining = True
    pretraining_str = 'pretraining' if pretraining else ''

    # trained_epoch = 12

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    addH = arch['addH']

    dff = d_model * 2
    vocab_size = 717
    dropout_rate = 0.1

    seed = seed
    np.random.seed(seed=seed)
    tf.random.set_seed(seed=seed)
    train_dataset, test_dataset , val_dataset = M2VGraph_Classification_Dataset('data/clf/{}.csv'.format(task), smiles_field='SMILES',
                                                               label_field='Label',addH=arch['addH']).get_data()



    x, adjoin_matrix, y = next(iter(train_dataset.take(1)))
    seq = tf.cast(tf.math.equal(x, 0), tf.float32)
    mask = seq[:, tf.newaxis, tf.newaxis, :]
    model = PredictModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                         dense_dropout=0.5)

    if pretraining:
        temp = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size)
        pred = temp(x, mask=mask, training=True, adjoin_matrix=adjoin_matrix)
        temp.load_weights(arch['path']+'/bert_weights_{}.h5'.format(arch['name']))
        temp.encoder.save_weights(arch['path']+'/bert_weights_encoder_{}_{}_{}.h5'.format(arch['name'], task, seed))
        del temp

        pred = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
        model.encoder.load_weights(arch['path']+'/bert_weights_encoder_{}_{}_{}.h5'.format(arch['name'], task, seed))
        print('load_wieghts')


    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    auc= -10
    stopping_monitor = 0

    # loss_train_writer = tf.summary.create_file_writer('./logs/train_log/nohloss-{}-{}'.format(str(seed), task))
    # acc_train_writer = tf.summary.create_file_writer('./logs/train_log/nohacc-{}-{}'.format(str(seed), task))
    #
    # auc_val_writer = tf.summary.create_file_writer('./logs/val_log/nohauc-{}-{}'.format(str(seed), task))
    # acc_val_writer = tf.summary.create_file_writer('./logs/val_log/nohacc-{}-{}'.format(str(seed), task))
    #
    # auc_test_writer = tf.summary.create_file_writer('./logs/test_log/nohauc-{}-{}'.format(str(seed), task))
    # acc_test_writer = tf.summary.create_file_writer('./logs/test_log/nohacc-{}-{}'.format(str(seed), task))

    for epoch in range(256):
        accuracy_object = tf.keras.metrics.BinaryAccuracy()
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        for x,adjoin_matrix,y in train_dataset:
            with tf.GradientTape() as tape:
                seq = tf.cast(tf.math.equal(x, 0), tf.float32)
                mask = seq[:, tf.newaxis, tf.newaxis, :]
                preds = model(x,mask=mask,training=True,adjoin_matrix=adjoin_matrix)
                loss = loss_object(y,preds)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                accuracy_object.update_state(y,preds)
        # with acc_train_writer.as_default():
        #     tf.summary.scalar(name="train_acc", data=accuracy_object.result().numpy().item(), step=epoch)
        # with loss_train_writer.as_default():
        #     tf.summary.scalar(name="train_loss", data=loss.numpy().item(), step=epoch)
        print('epoch: ',epoch,'loss: {:.4f}'.format(loss.numpy().item()),'accuracy: {:.4f}'.format(accuracy_object.result().numpy().item()))

        y_true = []
        y_preds = []

        for x, adjoin_matrix, y in val_dataset:
            seq = tf.cast(tf.math.equal(x, 0), tf.float32)
            mask = seq[:, tf.newaxis, tf.newaxis, :]
            preds = model(x,mask=mask,adjoin_matrix=adjoin_matrix,training=False)
            y_true.append(y.numpy())
            y_preds.append(preds.numpy())
        y_true = np.concatenate(y_true,axis=0).reshape(-1)
        y_preds = np.concatenate(y_preds,axis=0).reshape(-1)
        y_preds = tf.sigmoid(y_preds).numpy()
        auc_new = roc_auc_score(y_true,y_preds)
        val_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
        # with auc_val_writer.as_default():
        #     tf.summary.scalar(name="val_auc", data=auc_new, step=epoch)
        # with acc_val_writer.as_default():
        #     tf.summary.scalar(name="val_acc", data=val_accuracy, step=epoch)
        print('val auc:{:.4f}'.format(auc_new), 'val accuracy:{:.4f}'.format(val_accuracy))

        if auc_new > auc:
            auc = auc_new
            stopping_monitor = 0
            # np.save('{}/{}{}{}{}{}'.format(arch['path'], task, seed, arch['name'], trained_epoch, trained_epoch, pretraining_str),
            #         [y_true, y_preds])
            model.save_weights('weights_mfgb_hs_balanced/{}_{}.h5'.format(task,seed))
            print('save model weights')
        else:
            stopping_monitor += 1
        print('best val auc: {:.4f}'.format(auc))
        if stopping_monitor > 0:
            print('stopping_monitor:',stopping_monitor)
        if stopping_monitor >= 100:
            break

    y_true = []
    y_preds = []
    model.load_weights('weights_mfgb_hs_balanced/{}_{}.h5'.format(task, seed))
    for x, adjoin_matrix, y in test_dataset:
        seq = tf.cast(tf.math.equal(x, 0), tf.float32)
        mask = seq[:, tf.newaxis, tf.newaxis, :]
        preds = model(x, mask=mask, adjoin_matrix=adjoin_matrix, training=False)
        y_true.append(y.numpy())
        y_preds.append(preds.numpy())
    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
    y_preds = tf.sigmoid(y_preds).numpy()
    test_auc = roc_auc_score(y_true, y_preds)
    test_accuracy = keras.metrics.binary_accuracy(y_true.reshape(-1), y_preds.reshape(-1)).numpy()
    # with auc_test_writer.as_default():
    #     tf.summary.scalar(name="test_auc", data=test_auc, step=epoch)
    # with acc_test_writer.as_default():
    #     tf.summary.scalar(name="test_acc", data=test_accuracy, step=epoch)
    print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_accuracy))

    return test_auc

if __name__ == '__main__':
    auc_list = []
    for seed in [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]:
        print(seed)
        auc = main(seed)
        auc_list.append(auc)
    print(auc_list)



