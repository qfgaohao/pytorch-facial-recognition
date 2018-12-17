import torch
from .transforms import test_transform

import logging
import os
from PIL import Image


def name2path(lfw_image_dir, name, id):
    path = os.path.join(lfw_image_dir, name, f"{name}_{int(id):04d}.png")
    return path


def read_test_file(lfw_image_dir, lfw_test_file):
    images1 = []
    images2 = []
    labels = []
    with open(lfw_test_file) as f:
        f.readline()  # drop the first line
        for line in f:
            t = line.rstrip().split()
            if len(t) == 3:
                name1 = t[0]
                name2 = t[0]
                id1 = t[1]
                id2 = t[2]
                label = True
            elif len(t) == 4:
                name1 = t[0]
                id1 = t[1]
                name2 = t[2]
                id2 = t[3]
                label = False
            else:
                continue
            image1 = name2path(lfw_image_dir, name1, id1)
            image2 = name2path(lfw_image_dir, name2, id2)
            if os.path.exists(image1) and os.path.exists(image2):
                images1.append(image1)
                images2.append(image2)
                labels.append(label)
    return images1, images2, labels


def compute_distance(embeddings1, embeddings2):
    diff = embeddings1 - embeddings2
    distances = torch.sqrt(torch.sum(diff**2, 1))
    return distances


def pick_best_threshold(train_distances, labels):
    max_correct = 0
    max_threshold = 0
    for threshold in range(1, 30, 1):
        threshold = threshold / 10
        pred = train_distances <= threshold
        correct = pred.eq(labels).sum()
        if correct > max_correct:
            max_threshold = threshold
            max_correct = correct

    return max_threshold, max_correct.float() / labels.size(0)


def batch_inference(images1, images2, net, device, batch_size=100):
    all_distances = []
    for i in range(0, len(images1) - batch_size + 1, batch_size):
        end = min(len(images1), i + batch_size)
        sub_images1 = images1[i: end]
        sub_images1 = [test_transform(Image.open(img)) for img in sub_images1]
        sub_images1 = torch.stack(sub_images1)
        sub_images1 = sub_images1.to(device)

        sub_images2 = images2[i: end]
        sub_images2 = [test_transform(Image.open(img)) for img in sub_images2]
        sub_images2 = torch.stack(sub_images2)
        sub_images2 = sub_images2.to(device)

        embeddings1 = net(sub_images1)
        embeddings2 = net(sub_images2)
        distances = compute_distance(embeddings1, embeddings2)
        all_distances.append(distances)
    all_distances = torch.cat(all_distances)
    return all_distances


def test(clnet, lfw_image_dir, lfw_test_file, fold=10, device=None):
    clnet.train(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clnet.to(device)
    images1, images2, labels = read_test_file(lfw_image_dir, lfw_test_file)
    logging.info(f"Found {len(labels)} pairs to test.")
    labels = torch.tensor(labels, dtype=torch.uint8, device=device)
    distances = batch_inference(images1, images2, clnet, device=device)
    print('raw distance size', distances.size())
    del images2
    del images1
    num = distances.size(0)
    test_size = int(num/fold)
    random_indices = torch.randperm(num)
    distances = distances[random_indices]
    print(distances)
    print('distances size', distances.size())
    labels = labels[random_indices]
    labels.to(device)
    accuracy = 0
    print(labels)
    for i in range(0, num - test_size + 1, test_size):
        train_labels = torch.cat([labels[0:i], labels[i + test_size:]])
        train_distances = torch.cat([distances[0:i], distances[i + test_size:]])
        test_distances = distances[i:i + test_size]
        test_labels = labels[i:i + test_size]
        threshold, train_accuracy = pick_best_threshold(train_distances, train_labels)
        pred = test_distances <= threshold
        correct = pred.eq(test_labels).sum().float()
        test_accuracy = correct / test_size
        logging.info(f"Picked threshold: {threshold:.2f}. Train accuracy: {train_accuracy:.4f}."
                     f" Test accuracy: {test_accuracy:.4f}.")
        accuracy += correct / test_size
    accuracy /= fold
    logging.info(f"Final LFW Test Accuracy: {accuracy:.4f}.")
    return accuracy

