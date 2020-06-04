# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import shutil
from deeplabv3plus import model
from dataset import Dataset
from config import cfg


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

log = cfg.TRAIN.LOGDIR
EPOCHS = cfg.TRAIN.EPOCHS
save_every_n_epoch = cfg.TRAIN.SAVE_EPOCH

if os.path.exists(log): shutil.rmtree(log)

if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    TrainSet = Dataset('train')
    model = model(depthwise=True, backbone='mobilenetv2')

    if os.listdir('./saved_weights'):
        latest_weight = tf.train.latest_checkpoint('./saved_weights')
        # latest_weight = r"./saved_model/epoch-14"
        model.load_weights(latest_weight)

    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr=1e-5)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    miou = tf.keras.metrics.MeanIoU(num_classes=21, name='miou')

    summary_writer = tf.summary.create_file_writer(logdir='tensorboard')  # 实例化记录器
    tf.summary.trace_on(profiler=True)

    # @tf.function
    def train_step(image_batch, label_batch):
        with tf.GradientTape() as tape:
            predictions = model(image_batch)
            loss = loss_object(y_true=label_batch, y_pre=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)
        miou.update_state(y_true=label_batch, y_pred=tf.argmax(predictions, axis=-1))
    # start training
    step = 0
    for epoch in range(EPOCHS):

        for img, labels in TrainSet:
            train_step(img, labels)
            print("Epoch: {}/{}, step:{}, loss: {:.5f}, accuracy: {:.5f}, miou: {:.5f}".format(epoch + 1,
                                                                                               EPOCHS,
                                                                                               step,
                                                                                               train_loss.result().numpy(),
                                                                                               train_accuracy.result().numpy(),
                                                                                               miou.result().numpy()))
            with summary_writer.as_default():
                tf.summary.scalar("loss", train_loss.result().numpy(), step=step)
            step += 1

        if (epoch+1) % save_every_n_epoch == 0:
            model.save_weights(filepath='./saved_model' + "/epoch-{}".format(epoch), save_format='tf')

    tf.saved_model.save(model, 'FCN8s.h5')


'''

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                           tf.keras.metrics.MeanIoU(num_classes=21, name="meanIoU")],
                  experimental_run_tf_function=False
    )

    model.fit_generator(TrainSet, steps_per_epoch=416, epochs=10)
    model.save_weights('./deeplabv3plus')
    '''
