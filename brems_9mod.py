import numpy as np
import matplotlib.pyplot as plt
import utils_6
import torch
from datetime import datetime
from logger import Logger

torch.set_printoptions(precision=3)  # this doesn't seem to do anything
np.set_printoptions(precision=3, suppress=True)

USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')

BATCH_SZ, D_in_1, H, L, Q,  D_out = 32, 100, 50, 50, 10, 2
EPOCHS = 50_000


if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(D_in_1),
        torch.nn.Linear(D_in_1, H),
        #torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(H, L),
        #torch.nn.Sigmoid(),[1;2B
        torch.nn.Tanh(),
        torch.nn.Linear(L, Q),
        torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        #torch.nn.Linear(H, H),
        #torch.nn.Tanh(),
        #torch.nn.LeakyReLU(),
        torch.nn.Linear(Q, D_out),
    )

    if USE_GPU:
        model.cuda()

    return model


if __name__ == '__main__':
    plt.ion()
    plt.close('all')

    model = build_model()

    loss_fn = torch.nn.MSELoss()
    best_loss = float('inf')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-9, patience=100, factor=0.5, verbose=True,
    )

    # Logging of train and test loss
    experiment_id = datetime.now().isoformat()
    print('Logging experiment as: ', experiment_id)
    logger = Logger(f'/hdd/bahammel/tensorboard/{experiment_id}')

    # Load data
    trainer, tester = utils_6.get_data_4()
    train_loader = torch.utils.data.DataLoader(trainer, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(tester, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):

        # Train the model
        train_losses = []
        model.train()
        for x_batch, y_batch in train_loader:

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Every 10 Epochs, log the training loss and run the test set
        if epoch % 10 == 0:
            test_losses = []
            model.eval()
            for x_batch, y_batch in test_loader:
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                test_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            test_loss = np.mean(test_losses)

            print(f"epoch: {epoch}, train losses: {train_losses}")
            print(f"epoch: {epoch}, mean train loss: {train_loss}")
            print(f"epoch: {epoch}, mean test loss: {test_loss}")

            logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], epoch)
            logger.scalar_summary('train-loss', train_loss, epoch)
            logger.scalar_summary('test-loss', test_loss, epoch)

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

            # Update learning rate and save the model
            lr_scheduler.step(test_loss)

            if best_loss > test_loss:
                torch.save(model, f'/hdd/bahammel/checkpoint/{experiment_id}')
                best_loss = test_loss
