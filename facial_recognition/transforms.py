from torchvision import transforms


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1, 1, 1])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(160),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
