import gradio as gr
import torch
import numpy as np
from model import DigitClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitClassifier().to(device)
model.load_state_dict(torch.load("digit_classifier.pth", map_location=device))
model.eval()

def predict_digit(image):
    # تحويل الصورة إلى 28x28 رمادية
    image = np.array(image).astype(np.float32)
    if image.shape != (28, 28):
        image = np.resize(image, (28, 28))
    image = image / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        prediction = torch.argmax(outputs, dim=1).item()
    return f"الرقم المتوقع هو: {prediction}"

gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(image_mode='L', sources=['upload'], width=28, height=28),
    outputs="text",
    live=True,
    title="تصنيف الأرقام المكتوبة يدويًا",
    description="ارفع صورة رقم مكتوب يدويًا وسيقوم النموذج بتحديده"
).launch()

