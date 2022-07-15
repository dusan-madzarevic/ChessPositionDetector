import numpy
import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Dropout, RandomRotation, RandomFlip
from tensorflow import keras
from keras import layers, Sequential
import matplotlib.pyplot as plt
import keras_tuner as tuner
from tensorflow.python.keras import models

from chess_util import generate_FEN

image_size = (180, 180)
batch_size = 25
epoch_num = 10

labels = ["empty","bp","bb","bn","br","bq","bk","wp","wb","wn","wr","wq","wk"]
labels_base = ["empty","base_black","base_white"]
def load_train_data(directory, train_val_split):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=train_val_split,
        labels="inferred",
        label_mode="categorical",
        class_names=labels,
        shuffle=True,
        subset="training",
        color_mode="rgb",
        seed=2109,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=train_val_split,
        labels="inferred",
        class_names=labels,
        shuffle=True,
        label_mode="categorical",
        subset="validation",
        color_mode="rgb",
        seed=2109,
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_ds, val_ds

def load_train_data_base(directory, train_val_split):

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=train_val_split,
        labels="inferred",
        label_mode="categorical",
        class_names=labels_base,
        shuffle=True,
        subset="training",
        color_mode="rgb",
        seed=2109,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        validation_split=train_val_split,
        labels="inferred",
        class_names=labels_base,
        shuffle=True,
        label_mode="categorical",
        subset="validation",
        color_mode="rgb",
        seed=2109,
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_ds, val_ds

def visualize_data(dataset):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()

def construct_model_pretrained():

    resnet_model = Sequential()

   # resnet_model.add(RandomRotation(0.2))
    #resnet_model.add(RandomFlip("horizontal"))

    pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                      input_shape=(180, 180, 3),
                                                      pooling='avg', classes=13,
                                                      weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = False

    resnet_model.add(pretrained_model)

    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dropout(0.5))
    resnet_model.add(Dense(13, activation='softmax'))

    return resnet_model

def construct_model():

    inputs = keras.Input(shape=image_size + (3,), batch_size=batch_size)
    # Image augmentation block

    rescale_pixels = layers.Rescaling(1.0 / 255)(inputs)
    conv_1 = layers.Conv2D(32, 3, input_shape=(180, 180, 3), strides=2, padding="same")(rescale_pixels)
    bn_1 = layers.BatchNormalization()(conv_1)
    relu_1 = layers.Activation("relu")(bn_1)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(relu_1)
    x = layers.Dropout(0.2)(x)

    conv_2 = layers.Conv2D(64, 3, padding="same")(x)
    bn_2 = layers.BatchNormalization()(conv_2)
    relu_2 = layers.Activation("relu")(bn_2)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(relu_2)
    flatten = layers.Flatten()(x)
    x = layers.Dropout(0.2)(flatten)

    dens = layers.Dense(units=512, activation="relu")(x)
    drop_final = layers.Dropout(0.5)(dens)
    outputs = layers.Dense(units=13, activation="softmax")(dens)
    kerasModel = keras.Model(inputs, outputs)
    kerasModel.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return kerasModel

def construct_model_base():

    inputs = keras.Input(shape=image_size + (3,), batch_size=batch_size)
    # Image augmentation block

    rescale_pixels = layers.Rescaling(1.0 / 255)(inputs)
    conv_1 = layers.Conv2D(32, 3, input_shape=(180, 180, 3), strides=2, padding="same")(rescale_pixels)
    bn_1 = layers.BatchNormalization()(conv_1)
    relu_1 = layers.Activation("relu")(bn_1)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(relu_1)

    flatten = layers.Flatten()(x)

    outputs = layers.Dense(units=3, activation="softmax")(flatten)
    kerasModel = keras.Model(inputs, outputs)
    kerasModel.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return kerasModel

def train_model(model, train_dataset, validation_dataset):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(
        train_dataset, epochs=epoch_num, validation_data=validation_dataset,
    )

    model.save("model_resnet50_10.h5")

def write_final(final_results, field_scores, k, count):
    lab = labels[np.argmax(field_scores[k])]
    if lab == "wk" or lab == "bk" or lab == "wq" or lab == "bq":
        field_scores[k][0][np.argmax(field_scores[k])] = 0
        write_final(final_results, field_scores, k, count)
    if count.get(lab) is not None:
        count[lab] += 1
        if lab == "wp" or lab == "bp":
            if count[lab] <= 8:
                final_results[k + 1] = labels[np.argmax(field_scores[k])]
            else:
                print(field_scores[k])
                field_scores[k][0][np.argmax(field_scores[k])] = 0
                write_final(final_results, field_scores, k, count)
        else:
            if count[lab] <= 2:
                final_results[k + 1] = labels[np.argmax(field_scores[k])]
            else:
                print(field_scores[k])
                field_scores[k][0][np.argmax(field_scores[k])] = 0
                write_final(final_results, field_scores, k, count)
    else:
        final_results[k + 1] = labels[np.argmax(field_scores[k])]

def predict_board(model, model_base):

    count = {"wn": 0, "wb": 0, "wr": 0,"bn": 0,"bb": 0, "br": 0, "wp":0, "bp":0}

    prediction_labels = []
    scores = {}
    scores_crop = {}
    field_scores = []
    for label in range(13):
        scores[label] = []
        scores_crop[label] = []


    print(scores)
    final_results = {}
    for j in range(1, 65):
        final_results[j] = ''
    for i in range(1, 65):
        print(i)
        pos = ""
        if i < 10:
            pos = "0"+str(i)
        else:
            pos = str(i)
        img = keras.preprocessing.image.load_img(
            "Data/raw_data/alpha_data_image200"+pos+".jpeg", target_size=image_size
        )
        img_base = keras.preprocessing.image.load_img(
            "Data/raw_data_base/alpha_data_image200"+pos+".jpeg", target_size=image_size
        )
        img_crop = keras.preprocessing.image.load_img(
            "Data/raw_data_crop/alpha_data_image200"+pos+".jpeg", target_size=image_size
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        img_array_base = keras.preprocessing.image.img_to_array(img_base)
        img_array_base = tf.expand_dims(img_array_base, 0)  # Create batch axis

        img_array_crop = keras.preprocessing.image.img_to_array(img_crop)
        img_array_crop = tf.expand_dims(img_array_crop, 0)  # Create batch axis

        predicts = model.predict(img_array)
        score = predicts
        print(score)

        predicts_base = model_base.predict(img_array_base)
        score_base = predicts_base
        print(score_base)

        predicts_crop = model.predict(img_array_crop)
        score_crop = predicts_crop
        print(score_crop)
        field_scores.append(score_crop)
        for label in range(13):
            scores[label].append([score[0][label],score_base])
            scores_crop[label].append([score_crop[0][label], score_base])

    max_bk = max(scores[6])
    max_bk_index = scores[6].index(max_bk)
    scores[6][max_bk_index][0] = 0
    max_bk_2 = max(scores[6])
    max_bk_2_index = scores[6].index(max_bk_2)

    if (scores[6][max_bk_2_index][1][0][2] > scores[6][max_bk_index][1][0][2]):
        final_results[max_bk_2_index+1] = 'bk'
    else:
        final_results[max_bk_index+1] = 'bk'

    max_wk = max(scores[12])
    max_wk_index = scores[12].index(max_wk)
    scores[12][max_wk_index][0] = 0
    max_wk_2 = max(scores[12])
    max_wk_2_index = scores[12].index(max_wk_2)

    if(scores[12][max_wk_2_index][1][0][2] > scores[12][max_wk_index][1][0][2]):
        final_results[max_wk_2_index+1] = 'wk'
    else:
        final_results[max_wk_index+1] = 'wk'

    max_bq = max(scores[5])
    max_bq_val = max_bq[0]
    max_bq_index = scores[5].index(max_bq)
    scores[5][max_bq_index][0] = 0
    max_bq_2 = max(scores[5])
    max_bq_2_val = max_bq_2[0]
    max_bq_2_index = scores[5].index(max_bq_2)

    if max_bq_val > 0.95 and max_bq_2_val > 0.95:
        final_results[max_bq_index + 1] = 'bq'
        final_results[max_bq_2_index + 1] = 'bq'
    else:
        if (scores[5][max_bq_2_index][1][0][2] > scores[5][max_bq_index][1][0][2]):
            final_results[max_bk_2_index+1] = 'bq'
        else:
            final_results[max_bk_index+1] = 'bq'

    max_wq = max(scores[11])
    max_wq_val = max_wq[0]
    max_wq_index = scores[11].index(max_wq)
    scores[5][max_wq_index][0] = 0
    max_wq_2 = max(scores[11])
    max_wq2_val = max_wq_2[0]
    max_wq_2_index = scores[11].index(max_wq_2)

    if max_wq_val > 0.95 and max_wq2_val > 0.95:
        final_results[max_wq_index + 1] = 'wq'
        final_results[max_wq_2_index + 1] = 'wq'
    else:
        if (scores[11][max_wq_2_index][1][0][2] > scores[11][max_wq_index][1][0][2]):
            final_results[max_wk_2_index+1] = 'wq'
        else:
            final_results[max_wk_index+1] = 'wq'

    print(final_results)
    for k in range(len(field_scores)):
        if final_results[k + 1] == '':
            write_final(final_results,field_scores,k,count)



    print(scores_crop)
    print(final_results)
    l = list(final_results.values())
    l_rev = l[::-1]
    print(l)
    return l_rev


def load_model(model_path):
    model = keras.models.load_model(model_path)

    return model


if __name__ == '__main__':
    train_dataset, val_dataset = load_train_data("Train_Data", 0.2)
    print(train_dataset)
    #model = construct_model_pretrained()
    model = load_model("model_resnet50_10.h5")
    model_base = load_model("model_base.h5")
    #model.summary()

    #tuner = tuner.RandomSearch(
    #    hypermodel=construct_model,
    #    objective="val_accuracy",
    #    max_trials=10,
    #    executions_per_trial=2,
    #    overwrite=True,
    #    directory="my_dir",
    #    project_name="helloworld",
    #)

    predictions = predict_board(model, model_base)
    FEN = generate_FEN(predictions)

    print("The predicted FEN notation for this game is as follows:\n\n"+FEN+"\n\nInput it into your favorite chess engine to evaluate the postiion.")

    #tuner.search(train_dataset, epochs=30, validation_data=val_dataset, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

    #train_model(model, train_dataset, val_dataset)
    #keras.utils.plot_model(model, show_shapes=True)

    #test_model(model)
