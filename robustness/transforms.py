from io import BytesIO

from PIL import Image, ImageFilter
from torchvision import transforms


class JpegCompression:
    def __init__(self, quality=70):
        self.quality = quality

    def __call__(self, image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class ResizeDownUp:
    def __init__(self, scale=0.5):
        self.scale = scale

    def __call__(self, image):
        w, h = image.size
        small = image.resize((max(1, int(w * self.scale)), max(1, int(h * self.scale))), Image.BICUBIC)
        return small.resize((w, h), Image.BICUBIC)


class GaussianBlur:
    def __init__(self, radius=1.0):
        self.radius = radius

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


def robustness_transform(name, image_size=224, jpeg_quality=70, resize_scale=0.5, blur_radius=1.0):
    ops = robustness_pil_ops(name, jpeg_quality, resize_scale, blur_radius)
    ops.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def robustness_pil_ops(name, jpeg_quality=70, resize_scale=0.5, blur_radius=1.0):
    ops = []
    if name == "jpeg":
        ops.append(JpegCompression(jpeg_quality))
    elif name == "resize":
        ops.append(ResizeDownUp(resize_scale))
    elif name == "blur":
        ops.append(GaussianBlur(blur_radius))
    elif name != "clean":
        raise ValueError(f"Unknown robustness transform: {name}")
    return ops
