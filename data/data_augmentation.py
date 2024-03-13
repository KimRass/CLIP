import torchvision.transforms as T


def get_kernel_size(img_size):
    return round(img_size * 0.06) // 2 * 2 + 1


def get_train_transformer(img_size, to_tensor=True):
    # "A random square crop from resized images is the only data augmentation
    # used during training."
    # 데이터가 부족하므로 SimCLR (Chen et al., 2020)를 참고하여 Data augmentation을
    # 도입했습니다.
    kernel_size = get_kernel_size(img_size)
    transforms = [
        T.RandomResizedCrop(size=img_size, scale=(0.8, 1), ratio=(3 / 4, 4 / 3), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.04)], p=0.7,
        ),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(0.1, 2))], p=0.2,
        ),
    ]
    if to_tensor:
        transforms.extend(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    transformer = T.Compose(transforms)
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
    transformer = get_train_transformer(img_size=224, to_tensor=False)
    for idx in range(1, 21):
        out = transformer(image)
        out.save(f"data_augmentation/{idx}.jpg")
