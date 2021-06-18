import datetime
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from Settings import settings as S
import math
from Settings import utils as U

BATCH_SIZE = 32
IMG_SIZE = 224
BUFFER_SIZE = 2048
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
log_dir = "Data/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
AUTOTUNE = tf.data.experimental.AUTOTUNE

plt.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc', curve='ROC'),  # num_thresholds=2000, , multi_label=True
    tf.keras.metrics.AUC(name='prc', curve='PR'),
    tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
]


def data_frame_extractor(df, labels):
    files = df['Path'].values.tolist()  # create list
    df = df[S.CLASSES]
    labels = df.to_numpy()
    return files, labels


def imagenet_Normalize(x):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    x_standardized = (x - imagenet_mean) / imagenet_std
    return x_standardized


def parse_data(files, labels):
    # Read an image
    image = tf.io.read_file(files)
    # Decode to dense vector
    image_decoded = tf.image.decode_jpeg(image, channels=3)
    # change dtype to float32
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_standardized = imagenet_Normalize(image_decoded)
    return image_standardized, labels


def create_dataset(files, labels):
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    # parallelize
    dataset = dataset.map(parse_data, num_parallel_calls=AUTOTUNE)
    # Shuffle the data
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def prepare_TTA_valid_eval():
    def flip(x, label):
        image = tf.image.random_flip_up_down(x)
        return image, label

    def sat(x, label):
        image = tf.image.adjust_saturation(x, 2)
        return image, label

    def brgh(x, label):
        image = tf.image.adjust_brightness(x, 0.2)
        return image, label

    def rot_2(x, label):
        image = tfa.image.rotate(x, math.radians(-2), interpolation='BILINEAR')
        return image, label

    def rot_4(x, label):
        image = tfa.image.rotate(x, math.radians(-4), interpolation='BILINEAR')
        return image, label

    def rot_6(x, label):
        image = tfa.image.rotate(x, math.radians(-6), interpolation='BILINEAR')
        return image, label

    def rot2(x, label):
        image = tfa.image.rotate(x, math.radians(2), interpolation='BILINEAR')
        return image, label

    def rot4(x, label):
        image = tfa.image.rotate(x, math.radians(4), interpolation='BILINEAR')
        return image, label

    def rot6(x, label):
        image = tfa.image.rotate(x, math.radians(6), interpolation='BILINEAR')
        return image, label

    def prepare_ds(x=None, aug=False):
        test_df = pd.read_csv('Valid.csv')
        X_test, y_test = data_frame_extractor(test_df, [S.CLASSES])
        test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_set = test_set.map(parse_data, num_parallel_calls=AUTOTUNE)
        if aug:
            test_set = test_set.map(x)

        test_set = test_set.batch(234)
        return test_set

    aug = [flip, sat, brgh, rot2, rot4, rot6, rot_2, rot_4, rot_6]
    dt_list = []
    for i in aug:
        dt_list.append(prepare_ds(i, aug=True))
    a = prepare_ds().unbatch()
    c = a
    for i in range(9):
        c = c.concatenate(dt_list[i].unbatch())
    c = c.batch(234)
    c = c.prefetch(buffer_size=AUTOTUNE)
    return c


def validate_sub_score():
    test_df = pd.read_csv('Valid.csv')
    X_test, y_test = data_frame_extractor(test_df, [S.CLASSES])
    sub_score_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    sub_score_set = sub_score_set.map(parse_data)
    sub_score_set = sub_score_set.batch(234)
    return sub_score_set


def prepare_training_data_sets(a):
    # print_examples(img_path)
    train_df = pd.read_csv(a)
    valid_df = pd.read_csv('Valid.csv')

    X_train, y_train = data_frame_extractor(train_df, S.CLASSES)
    X_valid, y_valid = data_frame_extractor(valid_df, S.CLASSES)
    train_set = create_dataset(X_train, y_train)
    valid_set = create_dataset(X_valid, y_valid)
    return train_set, valid_set


def get_compiled_model(tune_at, path, metrics=METRICS, use_base_weights=True, fine_tune=False):
    '''
    Build architecture based on pretrained DenseNet
    :return: compiled model
    '''
    if use_base_weights is True:
        base_weights = "imagenet"
    else:
        base_weights = None

    base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights=base_weights)
    if fine_tune:
        base_model.trainable = True
        fine_tune_at = tune_at

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        initial_learning_rate = 0.00001
    else:
        base_model.trainable = False
        initial_learning_rate = 0.0001

    inputs = base_model.inputs
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions = tf.keras.layers.Dense(len(S.CLASSES), activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    '''
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=6896,  # 6896
        decay_rate=0.1,
        staircase=True)
    
    boundaries = [6982, 2*6982, 3*6982,4*6982]
    values = [0.0001, 0.00009, 0.000081, 0.000072, 0.000065]
    lr_schedule =tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
    '''

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[metrics])
    # model.summary()

    if fine_tune:
        latest = path  # 'Data/Weights/U_zeros_v1/model/tune/attempt1.ckpt'
        model.load_weights(latest)

    return model


def init_training(data_sets, model_checkpoint, model_weights, path = None, tune = False):
    '''
    Initialize training process and tensorboard.
    :param data_sets: csv file with paths and labels.
    :param model_checkpoint: path to directory, where checkpoints are stored.
    :param model_weights: path to directory, where best weights are stored.
    :return:
    '''
    train_set, valid_set = prepare_training_data_sets(data_sets)
    print("Done prep...\n")

    if tune is False:
        model = get_compiled_model()
    else:
        model = get_compiled_model(80, path, metrics=METRICS, use_base_weights=False, fine_tune=True)
    print('Training...\n')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",  # 'val_auc'
        verbose=1,
        min_delta=0.001,
        patience=4,
        mode='min',
        restore_best_weights=True)
    checkpoint_filepath = model_checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='loss',
        mode='min',
        save_weights_only=True,
        # save_freq= 3448,
        save_best_only=False)

    history = model.fit(train_set,
                        epochs=12,
                        validation_data=valid_set,
                        callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback],
                        shuffle=True,
                        )

    model.save_weights(model_weights, save_format="tf")
    print('Done :)\n')


def eval(model_weights, image_val_tta, image_val, image_all):
    model = get_compiled_model(use_base_weights=False)
    latest = model_weights

    model.load_weights(latest)
    val_tta = prepare_TTA_valid_eval()
    valid_set = validate_sub_score()

    predictions = model.predict(val_tta, verbose=1)
    evals = model.evaluate(val_tta, verbose=1)
    print(evals, '\n')
    roc = U.get_roc_curve(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'], predictions,
                           val_tta, image_val_tta)
    print(roc)
    print("5 res:", sum(roc) / 5)
    print('\n')

    print("Results: \n")
    predictions1 = model.predict(valid_set, verbose=1)
    a = U.get_roc_curve(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'], predictions1,
                         valid_set, image_val)
    print(a)
    print("5 res:", sum(a) / 5)

    d = U.get_roc_curve(S.CLASSES, predictions1, valid_set,image_all, roc_5 = False)
    print(d)

    evals = model.evaluate(valid_set, verbose=1)
    print(evals)

data_sets = 'Train_U_zeros_LSR_0_03.csv'
model_checkpoint = r'Data/Weights/U_zeros_LSR_0_03/checkpoints/weights.{epoch:02d}-{loss:.3f}.ckpt'
model_weights = r'Data/Weights/U_zeros_LSR_0_03/model/weights/attempt1.ckpt'
init_training(data_sets, model_checkpoint, model_weights)

image_val_tta = r'Data/Weights/U_zeros_LSR_0_03/image_val_tta.png'
image_val = r'Data/Weights/U_zeros_LSR_0_03/image_val.png'
image_all = r'Data/Weights/U_zeros_LSR_0_03/image_all.png'
eval(model_weights, image_val_tta, image_val, image_all)

model_checkpoint = r'Data/Weights/U_zeros_LSR_0_03/checkpoint_tune/weights.{epoch:02d}-{loss:.3f}.ckpt'
path = r'Data/Weights/U_zeros_LSR_0_03/model/weights/attempt1.ckpt'
model_weights =  r'Data/Weights/U_zeros_LSR_0_03/model/weights_after_tune/attempt1.ckpt'
init_training(data_sets, model_checkpoint, model_weights, path = path, tune = True )

image_val_tta = r'Data/Weights/U_zeros_LSR_0_03/tune_image_val_tta.png'
image_val = r'Data/Weights/U_zeros_LSR_0_03/tune_image_val.png'
image_all = r'Data/Weights/U_zeros_LSR_0_03/tune_image_all.png'
eval(model_weights, image_val_tta, image_val, image_all)