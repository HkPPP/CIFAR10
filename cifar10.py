import torch
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import CIFAR10_dataset, count_parameters, ICCNN


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Data augmentation
    # train_transform = transforms.Compose([
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  
    #     transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    # ])

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])


    
    mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
    # These values are mostly used by researchers as found to very useful in fast convergence

    # https://pytorch.org/vision/stable/transforms.html
    train_transform = transforms.Compose([
        transforms.RandomRotation(20), # Randomly rotate some images by 20 degrees
        transforms.RandomHorizontalFlip(0.1), # Randomly horizontal flip the images
        transforms.ColorJitter(brightness = 0.1, # Randomly adjust color jitter of the images
                              contrast = 0.1, 
                              saturation = 0.1), 
        transforms.RandomAdjustSharpness(sharpness_factor = 2,
                                        p = 0.1), # Randomly adjust sharpness
        transforms.ToTensor(),   # Converting image to tensor
        transforms.Normalize(mean, std), # Normalizing with standard mean and standard deviation
        # transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)
        ])


    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)
                                        ])


    train_dataset = CIFAR10_dataset(partition="train", transform=train_transform)
    test_dataset = CIFAR10_dataset(partition="test", transform=test_transform)

    batch_size = 100
    num_workers = multiprocessing.cpu_count()-1
    print("Num workers", num_workers)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)


    # Instantiating the network and printing its architecture
    num_classes = 10
    net = ICCNN(num_classes)
    print(net)


    print("Params: ", count_parameters(net))

    # Training hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, min_lr=0.00001)


    # Load model in GPU
    net.to(device)

    print("\n---- Start Training ----")
    epochs = 100
    best_accuracy = -1
    best_epoch = 0
    for epoch in range(epochs):
        # TRAIN NETWORK
        train_loss, train_correct = 0, 0
        net.train()
        with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for batch in tepoch:

                # Returned values of Dataset Class
                images = batch["img"].to(device)
                labels = batch["label"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                outputs = net(images)
                loss = criterion(outputs, labels)

                # Calculate gradients
                loss.backward()

                # Update gradients
                optimizer.step()

                # one hot -> labels
                labels = torch.argmax(labels, dim=1)
                pred = torch.argmax(outputs, dim=1)
                train_correct += pred.eq(labels).sum().item()

                # print statistics
                train_loss += loss.item()

        train_loss /= (len(train_dataloader.dataset) / batch_size)

        # TEST NETWORK
        test_loss, test_correct = 0, 0
        net.eval()
        with torch.no_grad():
            with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
                for batch in tepoch:

                    images = batch["img"].to(device)
                    labels = batch["label"].to(device)

                    # Forward
                    outputs = net(images)
                    test_loss += criterion(outputs, labels)

                    # one hot -> labels
                    labels = torch.argmax(labels, dim=1)
                    pred = torch.argmax(outputs, dim=1)

                    test_correct += pred.eq(labels).sum().item()

        lr_scheduler.step(test_loss)

        test_loss /= (len(test_dataloader.dataset) / batch_size)
        test_accuracy = 100. * test_correct / len(test_dataloader.dataset)

        print("[Epoch {}] Train Loss: {:.6f} - Test Loss: {:.6f} - Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%".format(
            epoch + 1, train_loss, test_loss, 100. * train_correct / len(train_dataloader.dataset), test_accuracy
        ))

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch

            # Save best weights
            torch.save(net.state_dict(), "models/best_model.pt")

    print("\nBEST TEST ACCURACY: ", best_accuracy, " in epoch ", best_epoch)