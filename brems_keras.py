import matplotlib.pyplot as plt
from keras import layers, models, callbacks
import utils

plt.ion()
plt.close('all')

BATCH_SZ, D_in, H, D_out = 128, 2, 800, 1
EPOCHS = 5000


def model_fn():

    input_ = layers.Input((D_in,))
    # x = BatchNormalization()(input_)
    x = layers.Dense(H, activation='sigmoid')(input_)
    x = layers.Dense(H, activation='sigmoid')(x)
    x = layers.Dense(D_out)(x)

    return models.Model(input_, x)


if __name__ == '__main__':

    model = model_fn()
    model.compile('adam', 'mse')

    xtrain, xtest, ytrain, ytest = utils.get_data()

    utils.plot_data((xtrain, ytrain), (xtest, ytest))

    history = model.fit(
        xtrain, ytrain,
        batch_size=BATCH_SZ,
        epochs=EPOCHS,
        callbacks=[callbacks.ReduceLROnPlateau(min_lr=1e-6)],
        validation_data=(xtest, ytest)
    )

    utils.plot_loss(history.history['loss'], history.history['val_loss'])

    fig = plt.figure('test', dpi=100, figsize=(5, 4))
    I_ = model.predict(xtest)
    hu, _ = list(zip(*xtest))
    plt.plot(hu, ytest, 'o')
    plt.plot(hu, I_, 'o')

    fig = plt.figure('train', dpi=100, figsize=(5, 4))
    I_ = model.predict(xtrain)
    hu, _ = list(zip(*xtrain))
    plt.plot(hu, ytrain, 'o')
    plt.plot(hu, I_, 'o')
