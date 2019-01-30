import torch
from .transforms import test_transform

import logging
import os
from PIL import Image


def name2path(lfw_image_dir, name, id):
    path = os.path.join(lfw_image_dir, name, f"{name}_{int(id):04d}.jpg")
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


def pick_best_threshold(train_distances, labels, threshold_range):
    max_correct = 0
    max_threshold = 0
    for threshold in range(-30, 30, 1):
        threshold = threshold / 30 * threshold_range
        pred = train_distances >= threshold
        correct = pred.eq(labels).sum()
        if correct > max_correct:
            max_threshold = threshold
            max_correct = correct

    return max_threshold, max_correct.float() / labels.size(0)


def compute_l2_distance(embeddings1, embeddings2):
    diff = embeddings1 - embeddings2
    distances = torch.sqrt(torch.sum(diff**2, 1))
    return distances


def compute_embeddings(net, images, input_size, flip, device):
    transform = test_transform(input_size)
    inputs = [transform(Image.open(img)) for img in images]
    inputs = torch.stack(inputs).to(device)
    embeddings = net(inputs)
    if flip:
        flipped_inputs = [transform(Image.open(img).transpose(Image.FLIP_LEFT_RIGHT)) for img in images]
        flipped_inputs = torch.stack(flipped_inputs).to(device)
        flipped_embeddings = net(flipped_inputs)
        embeddings = torch.cat([embeddings, flipped_embeddings], dim=1)
    return embeddings


def batch_inference(images1, images2, net, input_size, metric_fun, flip, device, batch_size=1):
    all_distances = []
    for i in range(0, len(images1) - batch_size + 1, batch_size):
        end = min(len(images1), i + batch_size)
        embeddings1 = compute_embeddings(net, images1[i: end], input_size, flip, device)
        embeddings2 = compute_embeddings(net, images2[i: end], input_size, flip, device)
        distances = metric_fun(embeddings1, embeddings2).data
        all_distances.append(distances)
    all_distances = torch.cat(all_distances)
    return all_distances


def test(net, lfw_image_dir, lfw_test_file,  input_size, metric, flip, fold=10, device=None):
    net.train(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print('lfw_test_file', lfw_test_file)
    images1, images2, labels = read_test_file(lfw_image_dir, lfw_test_file)
    logging.info(f"Found {len(labels)} pairs to test.")
    labels = torch.tensor(labels, dtype=torch.uint8, device=device)
    if metric == 'l2':
        metric_fun = compute_l2_distance
        threshold_range = 3
    elif metric == 'cosine':
        metric_fun = torch.nn.functional.cosine_similarity
        threshold_range = 1
    else:
        raise ValueError(f"Metric {metric} is not supported. Only l2 and cosine are supported right now.")
    distances = batch_inference(images1, images2, net, input_size, metric_fun=metric_fun, flip=flip, device=device)
    del images2
    del images1
    num = distances.size(0)
    test_size = int(num/fold)
    random_indices = torch.randperm(num)
    distances = distances[random_indices]
    labels = labels[random_indices]
    labels.to(device)
    accuracy = 0
    for i in range(0, num - test_size + 1, test_size):
        train_labels = torch.cat([labels[0:i], labels[i + test_size:]])
        train_distances = torch.cat([distances[0:i], distances[i + test_size:]])
        test_distances = distances[i:i + test_size]
        test_labels = labels[i:i + test_size]
        threshold, train_accuracy = pick_best_threshold(train_distances, train_labels, threshold_range)
        pred = test_distances >= threshold
        correct = pred.eq(test_labels).sum().float()
        test_accuracy = correct / test_size
        logging.info(f"Picked threshold: {threshold:.2f}. Train accuracy: {train_accuracy:.4f}."
                     f" Test accuracy: {test_accuracy:.4f}.")
        accuracy += correct / test_size
    accuracy /= fold
    logging.info(f"Final LFW Test Accuracy: {accuracy:.4f}.")
    return accuracy

