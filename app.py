import gradio as gr
import numpy as np
import joblib

# تحميل النموذج والمحول
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_digit(image):
    image = image.reshape(64)
    image_scaled = scaler.transform([image])
    prediction = model.predict(image_scaled)[0]
    return f"الرقم المتوقع هو: {prediction}"

gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(shape=(8, 8), image_mode='L', invert_colors=True, source='canvas'),
    outputs="text",
    live=True,
    title="تصنيف الأرقام المكتوبة يدويًا",
    description="ارسم رقمًا وسيقوم النموذج بتحديده"
).launch()

