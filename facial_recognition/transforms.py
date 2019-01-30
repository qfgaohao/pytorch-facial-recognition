from torchvision import transforms
from PIL import Image
import collections
from torchvision.transforms import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


class ResizedPad:
    def __init__(self, size, interpolation=Image.BILINEAR):
        """Resize and pad the image to the target size while keeping the w/h ratio.
        """
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        img_w, img_h = img.size
        h, w = self.size
        if img_w / img_h > w / h:
            resized_w = w
            resized_h = int(img_h / img_w * resized_w)
            padding = int((h - resized_h)/2)
            padding = (0, padding, 0, h - resized_h - padding)
        else:
            resized_h = h
            resized_w = int(img_w / img_h * resized_h)
            padding = int((w - resized_w)/2)
            padding = (padding, 0, w - resized_w - padding, 0)
        resized_image = F.resize(img, (resized_h, resized_w), self.interpolation)
        img = F.pad(resized_image, padding)
        return img

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1, 1, 1])
train_transform = lambda size: transforms.Compose([
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # transforms.RandomGrayscale(0.1),
    # transforms.RandomRotation(10),
    ResizedPad(size),
#    transforms.RandomResizedCrop(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


test_transform = lambda size: transforms.Compose([
    ResizedPad(size),
    #transforms.Resize(size),
    transforms.ToTensor(),
    normalize
])
