from torchvision import transforms


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1, 1, 1])
train_transform = lambda input_size: transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(160),
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


test_transform = lambda input_size: transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    normalize
])
