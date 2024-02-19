import numpy as np
from tensorflow.keras import layers, models, optimizers, callbacks, utils, constraints, Model

    
    
data = np.load('/home/lasii/Research/dataset/alz/processed/data.npy')
label = np.load('/home/lasii/Research/dataset/alz/processed/label.npy')


data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)



nb_classes = 3
epochs = 60
patience=20
batch_size = 32
input_shape=(data.shape[1],data.shape[2],data.shape[3])
chans = 19
fs = 500
time = 30
samples = fs*time



    
model_eegnet = models.Sequential()

model_eegnet.add(layers.Input(shape=(chans,samples,1)))

model_eegnet.add(layers.Convolution2D(8, (1, 32), padding='same', data_format='channels_last', use_bias=False))   
model_eegnet.add(layers.BatchNormalization(axis=3))

model_eegnet.add(layers.DepthwiseConv2D((chans, 1), use_bias=False, depth_multiplier=2, depthwise_constraint=constraints.max_norm(1.)))
model_eegnet.add(layers.BatchNormalization(axis=3))
model_eegnet.add(layers.Activation('elu'))
model_eegnet.add(layers.AveragePooling2D(pool_size=(1, 4)))
model_eegnet.add(layers.Dropout(0.25))

model_eegnet.add(layers.SeparableConv2D(16, (1, 16), use_bias=False, padding='same'))
model_eegnet.add(layers.BatchNormalization(axis=3))
model_eegnet.add(layers.Activation('elu'))
model_eegnet.add(layers.AveragePooling2D(pool_size=(1, 8)))
model_eegnet.add(layers.Dropout(0.25))
 
model_eegnet.add(layers.Flatten())

# model_eegnet.add(layers.Dense(256,activation='elu'))
model_eegnet.add(layers.Dense(784,activation='elu'))
model_eegnet.add(layers.Dense(784,activation='elu'))
model_eegnet.add(layers.Dense(784,activation='elu', name='feat_layer'))
model_eegnet.add(layers.Dense(nb_classes,activation='softmax'))

model_eegnet.summary()
model_eegnet.compile(loss='categorical_crossentropy', optimizer = optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])

earlystop_callback = callbacks.EarlyStopping(monitor='loss', patience=patience)

model_eegnet.fit(
     data,
     utils.to_categorical(label, num_classes=nb_classes),
     batch_size=batch_size,
     epochs=epochs,
     callbacks=[earlystop_callback],
     shuffle=True)


model_eegnet_feat_extractor = Model(inputs=model_eegnet.inputs, outputs=model_eegnet.get_layer(name="feat_layer").output)

eegNet_late_feat = model_eegnet_feat_extractor.predict(data[:10])

np.save('/home/lasii/Research/dataset/alz/processed/data_emb.npy', eegNet_late_feat, fix_imports=True)




