# SimCLR (Chen et al., 2020)을 참고하여 구현했습니다.

import torchvision.transforms as T


def _get_kernel_size(img_size):
    return round(img_size * 0.1) // 2 * 2 + 1


def get_train_transformer(img_size):
    kernel_size = _get_kernel_size(img_size)
    transformer = T.Compose(
        [
            T.RandomResizedCrop(size=img_size, scale=(0.6, 1), ratio=(3 / 4, 4 / 3), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.05)], p=0.8,
            ),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 2))], p=0.5,
            ),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transformer


def get_val_transformer(img_size):
    transformer = T.Compose([
        T.Resize(size=img_size),
        T.CenterCrop(size=img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transformer


if __name__ == "__main__":
    from PIL import Image

    image = Image.open("/Users/jongbeomkim/Documents/datasets/flickr8k_subset/Images/781118358_19087c9ec0.jpg")
    transformer = get_image_transformer(img_size=224)
    for idx in range(1, 11):
        out = transformer(image)
        out.save(f"data_augmentation/{idx}.jpg")
