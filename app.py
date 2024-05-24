import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import tempfile
from PIL import Image
import os

# Define the Model class as provided
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        model = resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))

def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img):
    fmap, logits = model(img.to('cpu'))
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :])
    result = heatmap * 0.5 + img * 0.8 * 255
    result1 = heatmap * 0.5 / 255 + img * 0.8
    r, g, b = cv2.split(result1)
    result1 = cv2.merge((r, g, b))
    result1 = result1.clip(0, 1)  # Ensure the result is in [0, 1] range
    return int(prediction.item()), confidence, result1

class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=60, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(self.video_path)):
            if (i % a == first_frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    frame = frame[y:y + h, x:x + w, :]
                frames.append(self.transform(frame))
                if len(frames) == self.count:
                    break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def main():

    st.set_page_config(
    page_title="Deepfakes Detection",
    page_icon="🎭",
    layout="wide",
    )
    st.title("Deepfake Detection System")

    st.write("Deepfakes are AI-generated synthetic media that convincingly alter or fabricate images and videos. Our system identifies and classifies them using advanced machine learning techniques that analyze inconsistencies in visual data.")
    
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model(2).to(device)
        
        path_to_model = 'model.pt'
        
        state_dict = torch.load(path_to_model, map_location=device)
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()

        video_dataset = ValidationDataset(tfile.name, sequence_length=20, transform=train_transforms)
        
        prediction, confidence, result_img = predict(model, video_dataset[0])
        
        if prediction == 1:
            st.success("Prediction: REAL")
        else:
            st.warning("Prediction: FAKE")
        
        st.subheader(f"Confidence: {confidence:.2f}%")
        
        st.image(result_img, caption='Prediction Heatmap')  # Add clamp=True to ensure the image is within [0, 1]

if __name__ == '__main__':
    main()
