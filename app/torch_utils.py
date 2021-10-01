import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path

channels_mean  = [0.49139968, 0.48215841, 0.44653091]
channels_stdev = [0.24703223, 0.24348513, 0.26158784]

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=channels_mean, std=channels_stdev),
                                    ])
    image = Image.open(io.BytesIO(image_bytes))
    t_img = transform(image).unsqueeze(0)
    return t_img, image

def get_prediction(image_tensor):
    model_path = Path("./cifar10_resnet18.pt")
    if not model_path.exists():
        gdd.download_file_from_google_drive(file_id='1--JIPQ_1XMQuPanEu4cWdHH0Qw0yvvBY',
                                            dest_path='./cifar10_resnet18.pt', unzip=False)
        print('Model downloaded')
    else:
        print('Model exists')
    model = torch.jit.load('./cifar10_resnet18.pt')
    print('Model loaded')
    model.eval()
    with torch.no_grad():
        labels_pred: Tensor = model(image_tensor)
        print('labels_pred.size():', labels_pred.size())
        labels_pred_max = labels_pred.argmax(dim=1, keepdim=True)
    labels_pred_max = labels_pred_max.cpu().numpy()
    print('labels_preds_max converted:', type(labels_pred_max))
    return labels_pred_max


