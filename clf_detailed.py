import numpy as np
from tensorflow.keras import layers, models, optimizers,callbacks, utils




data = np.load('/home/lasii/Research/dataset/alz/processed/data.npy')
label = np.load('/home/lasii/Research/dataset/alz/processed/labels.npy')

data_aug = np.load('/home/lasii/Research/dataset/alz/processed/aug_data.npy')
label_aug = np.load('/home/lasii/Research/dataset/alz/processed/aug_labels.npy')



batch_size = 32
patience=3
epochs = 50




def mlp_clf():
    model = models.Sequential()
    # model.add(layers.BatchNormalization())

    model.add(layers.Dense(1024, activation='selu'))
    model.add(layers.Dense(1024, activation='selu'))
    model.add(layers.Dense(1024, activation='selu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    return model

earlystop_callback = callbacks.EarlyStopping(monitor='loss', patience=patience)


nb_classes = 3


#TSTO

TOTS = mlp_clf()
TOTS.fit(
    data,
    utils.to_categorical(label, num_classes=nb_classes),
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    callbacks=[earlystop_callback],
    validation_data=(data_aug, utils.to_categorical(label_aug, num_classes=nb_classes))
    )

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


# TOTS

TSTO = mlp_clf()
TSTO.fit(
    data_aug,
    utils.to_categorical(label_aug, num_classes=nb_classes),
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    callbacks=[earlystop_callback],
    validation_data=(data, utils.to_categorical(label, num_classes=nb_classes))
    )


TOTS.evaluate(data_aug, utils.to_categorical(label_aug, num_classes=nb_classes))
TSTO.evaluate(data, utils.to_categorical(label, num_classes=nb_classes))

