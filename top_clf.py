import numpy as np
from tensorflow.keras import layers, models, optimizers,callbacks, utils
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix




# ddpm_data_aug = np.load('/home/lasii/Research/jbhi/ddpm_aug_data.npy')
# ddpm_label_aug = np.load('/home/lasii/Research/jbhi/ddpm_aug_labels.npy')

# vae_data_aug = np.load('/home/lasii/Research/jbhi/vae_aug_data.npy')
# vae_label_aug = np.load('/home/lasii/Research/jbhi/vae_aug_labels.npy')



data = np.load('/home/lasii/Research/jbhi/test.npy')
label = np.load('/home/lasii/Research/jbhi/test_y.npy')

data_aug = np.load('/home/lasii/Research/jbhi/vae_aug_data.npy')
label_aug = np.load('/home/lasii/Research/jbhi/vae_aug_labels.npy')



# data_aug = np.load('/home/lasii/Research/jbhi/ddpm_aug_data.npy')
# label_aug = np.load('/home/lasii/Research/jbhi/ddpm_aug_labels.npy')



nb_classes = 3
batch_size = 128
patience=5
epochs = 10




def mlp_clf():
    model = models.Sequential()
    # model.add(layers.BatchNormalization())

    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer = optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    return model

earlystop_callback = callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)


# #TSTO

# TOTS = mlp_clf()
# TOTS.fit(
#     data,
#     utils.to_categorical(label, num_classes=nb_classes),
#     batch_size=batch_size,
#     epochs=epochs,
#     shuffle=True,
#     callbacks=[earlystop_callback],
#     validation_data=(data_aug, utils.to_categorical(label_aug, num_classes=nb_classes))
#     )

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


# TOTS
nb_classes = 3
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



##########################################################################################################33


ind_0 = np.where(label == 0)
ind_0_a = np.where(label_aug == 0)

ind_1 = np.where(label == 1)
ind_1_a = np.where(label_aug == 1)

ind_2 = np.where(label == 2)
ind_2_a = np.where(label_aug == 2)

##########################################################################################################33


data_aug_01 = np.concatenate((data_aug[ind_0_a], data_aug[ind_1_a]))
label_aug_01 = np.concatenate((label_aug[ind_0_a], label_aug[ind_1_a]))

data_01 = np.concatenate((data[ind_0], data[ind_1]))
label_01 = np.concatenate((label[ind_0], label[ind_1]))


nb_classes = 2

TSTO_01 = mlp_clf()
TSTO_01.fit(
    data_aug_01,
    utils.to_categorical(label_aug_01, num_classes=nb_classes),
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    callbacks=[earlystop_callback],
    validation_data=(data_01, utils.to_categorical(label_01, num_classes=nb_classes))
    )


##########################################################################################################33


data_aug_02 = np.concatenate((data_aug[ind_0_a], data_aug[ind_2_a]))
label_aug_02 = np.concatenate((label_aug[ind_0_a], label_aug[ind_2_a]))
label_aug_02[label_aug_02==2]=1

data_02 = np.concatenate((data[ind_0], data[ind_2]))
label_02 = np.concatenate((label[ind_0], label[ind_2]))
label_02[label_02==2]=1


nb_classes = 2

TSTO_02 = mlp_clf()
TSTO_02.fit(
    data_aug_02,
    utils.to_categorical(label_aug_02, num_classes=nb_classes),
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    callbacks=[earlystop_callback],
    validation_data=(data_02, utils.to_categorical(label_02, num_classes=nb_classes))
    )

##########################################################################################################33


data_aug_12 = np.concatenate((data_aug[ind_1_a], data_aug[ind_2_a]))
label_aug_12 = np.concatenate((label_aug[ind_1_a], label_aug[ind_2_a]))
label_aug_12[label_aug_12==1]=0
label_aug_12[label_aug_12==2]=1


data_12 = np.concatenate((data[ind_1], data[ind_2]))
label_12 = np.concatenate((label[ind_1], label[ind_2]))
label_12[label_12==1]=0
label_12[label_12==2]=1

nb_classes = 2

TSTO_12 = mlp_clf()
TSTO_12.fit(
    data_aug_12,
    utils.to_categorical(label_aug_12, num_classes=nb_classes),
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    callbacks=[earlystop_callback],
    validation_data=(data_12, utils.to_categorical(label_12, num_classes=nb_classes))
    )


##########################################################################################################33




# # TOTS.evaluate(data_aug, utils.to_categorical(label_aug, num_classes=nb_classes))
# TSTO.evaluate(data, utils.to_categorical(label, num_classes=3))
# print(precision_recall_fscore_support(label, np.argmax(TSTO.predict(data), axis = 1), average='macro'))
# conf_matrix = confusion_matrix(label, np.argmax(TSTO.predict(data), axis = 1))


# print(precision_recall_fscore_support(label, np.argmax(TSTO.predict(data), axis = 1), average='micro'))
# print(precision_recall_fscore_support(label, np.argmax(TSTO.predict(data), axis = 1), average='weighted'))



TSTO_01.evaluate(data_01, utils.to_categorical(label_01, num_classes=nb_classes))
print(precision_recall_fscore_support(label_01, np.argmax(TSTO_01.predict(data_01), axis = 1), average='macro'))
conf_matrix_01 = confusion_matrix(label_01, np.argmax(TSTO_01.predict(data_01), axis = 1))

# print(precision_recall_fscore_support(label_01, np.argmax(TSTO_01.predict(data_01), axis = 1), average='micro'))
# print(precision_recall_fscore_support(label_01, np.argmax(TSTO_01.predict(data_01), axis = 1), average='weighted'))



# TSTO_02.evaluate(data_02, utils.to_categorical(label_02, num_classes=nb_classes))
# print(precision_recall_fscore_support(label_02, np.argmax(TSTO_01.predict(data_02), axis = 1), average='macro'))
# conf_matrix_02 = confusion_matrix(label_02, np.argmax(TSTO_02.predict(data_02), axis = 1))

# print(precision_recall_fscore_support(label_02, np.argmax(TSTO_01.predict(data_02), axis = 1), average='micro'))
# print(precision_recall_fscore_support(label_02, np.argmax(TSTO_01.predict(data_02), axis = 1), average='weighted'))



TSTO_12.evaluate(data_12, utils.to_categorical(label_12, num_classes=nb_classes))
print(precision_recall_fscore_support(label_12, np.argmax(TSTO_12.predict(data_12), axis = 1), average='macro'))
conf_matrix_12 = confusion_matrix(label_12, np.argmax(TSTO_12.predict(data_12), axis = 1))

# print(precision_recall_fscore_support(label_12, np.argmax(TSTO_01.predict(data_12), axis = 1), average='micro'))
# print(precision_recall_fscore_support(label_12, np.argmax(TSTO_01.predict(data_12), axis = 1), average='weighted'))




print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5')












