import cv2
import numpy as np
import PIL
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torchvision
import validators


def load_model(path):
    num_classes = 2
    input_size = 299
    model = torchvision.models.inception_v3()
    # Handle the auxilary net
    num_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, input_size


def preprocess_image(image):
    global input_size
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    return image


def model_predict(image):
    global model
    # 0: 'female',
    # 1: 'male'
    image = preprocess_image(image)
    with torch.no_grad():
        out = model(image)
    pct = torch.nn.functional.softmax(out, dim=1)[0] * 100
    female = pct[0].item()
    male = pct[1].item()
    gender_raw = abs(2 * female - 100)
    gender_bin = np.power(gender_raw, (1/2)) * 10
    if female < male:
        gender_raw = -gender_raw
        gender_bin = -gender_bin
    # female = 60; male = 100 - female
    # st.write(female, male)
    enby_raw = 100 - abs(female - male)
    enby_bin = np.power(enby_raw, (2)) / 100
    pred = {
        'female'    : round(female, 2),
        'male'      : round(male, 2),
        'max'       : round(np.power(max(female, male), (1/2)), 2),
        'enby_bin'  : round(enby_bin, 2),
        'enby_raw'  : round(enby_raw, 2),
        'gender_raw': round(gender_raw, 2),
        'gender_bin': round(gender_bin, 2),
    }
    return pred


def plot_pred(pred, bin=False):
    gender = pred['gender_bin'] if bin else pred['gender_raw']
    # st.write(gender)
    fig1 = go.Figure()
    fig1.add_trace(go.Indicator(
    mode = "gauge+number+delta",
    value = abs(gender/10),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Gender Score", 'font': {'size': 54}, },
    delta = {
        'reference': 100,
        'increasing': {'color': "rgba(0,0,0,0)"},
        'decreasing': {'color': "rgba(0,0,0,0)"}
        },
    gauge = {
        'axis': {
            'range': [-100, 100],
            'tickwidth': 2,
            'tickcolor': "rgba(0,0,0,0)",
            'visible': False
            },
        'bar': {'color': "rgba(0,0,0,0)"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [x, y], 'color': f'hsv({z},100,100)'}
            for x, y, z in zip(
                [x for x in range(-100, 100, 2)],
                [y for y in range(-95, 105, 2)],
                [z for z in range(220, 320)]
                )
            ],
        'threshold': {
            'value': gender,
            'thickness': 1,
            'line': {'color': "hsv(120,100,100)", 'width': 5},
        }}
    ))
    fig1.update_layout(
        font = {
            'color': "hsv(150,100,100)",
            'family': "Arial"}
    )
    fig2 = go.Figure()
    enby = pred['enby_bin'] if bin else pred['enby_raw']
    fig2.add_trace(go.Indicator(
        mode = "number+gauge",
        value = round(enby)/10,
        gauge = {
            'shape': "bullet",
            'axis': {
                'range': [0, 100],
                'tickcolor': "rgba(0,0,0,0)",
                'visible': False
            },
            'bordercolor': "gray",
            'bar': {
                'thickness': 1,
                'color': "hsva(270,100,100,0)"
                },
            'steps': [
                {'range': [x, y], 'color': f'rgba(0,0,0,{int(z+1<=enby)})'}
                for z, (x, y) in enumerate(zip(range(0, 24), range(2, 26)))
            ] + [
                {'range': [x, y], 'color': f'rgba(148,0,211,{int(z+26<=enby)})'}
                for z, (x, y) in enumerate(zip(range(25, 49), range(27, 51)))
            ] + [
                {'range': [x, y], 'color': f'rgba(255,255,255,{int(z+51<=enby)})'}
                for z, (x, y) in enumerate(zip(range(50, 74), range(52, 76)))
            ] + [
                {'range': [x, y], 'color': f'rgba(255,255,0,{int(z+76<=enby)})'}
                for z, (x, y) in enumerate(zip(range(75, 99), range(77, 100)))
                ],
            'threshold': {
            'value': enby,
            'thickness': 1,
            'line': {'color': "hsv(120,100,100)", 'width': 5},
            },
        },
        domain = {'x': [0, 1], 'y': [0.75, 1]},
        title = {'text': "",},
        ))
    fig2.update_layout(
        title={
            'text': 'Nonbinary Score',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 54},
            },
        font = {
            'color': "hsv(150,100,100)",
            'family': "Arial"}
        )
    return fig1, fig2


def application(img_arr, show_upload=False):
    if show_upload:
        img = PIL.Image.fromarray(img_arr)
        # with st.expander('Uploaded image:'):
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            st.write("")
        with col2:
            st.image(img, width=300)
        with col3:
            st.write("")
    if len(img_arr.shape) >= 3:
        if img_arr.shape[2] == 4:
            img_arr = img_arr[:,:,:3]
    else:
        img_arr = np.concatenate((
            img_arr[..., np.newaxis],
            img_arr[..., np.newaxis],
            img_arr[..., np.newaxis]
            ),
            axis=2
        )
        st.write(img_arr.shape)
    img = PIL.Image.fromarray(img_arr)
    pred = model_predict(img)
    with st.expander('Advanced options:'):
        functions = ("Realistic score", "Raw score")
        function = st.radio("Algorithm", functions)
        bin = function == functions[0]
    figs = plot_pred(pred, bin=bin)
    # st.write(pred)
    st.write('<br><br>', unsafe_allow_html=True)
    st.plotly_chart(figs[0], use_container_width=True)
    st.plotly_chart(figs[1], use_container_width=True)


def main():
    global model, input_size
    model, input_size = load_model('./cnn.pt')
    st.title('Gender Adversial')
    input_methods = (
        'Upload a picture (jpg, png)',
        'Take a picture (webcam)',
        'Link a picture (url)'
        )
    input_choice = st.radio('Choose an input method:', input_methods)
    if input_choice == input_methods[0]:
        img_upload = st.file_uploader(
            'Upload a picture',
            ['jpg', 'png']
            )
        if img_upload is not None:
            img_arr = np.array(PIL.Image.open(img_upload))
            application(img_arr, show_upload=True)
    elif input_choice == input_methods[1]:
        img_webcam = st.camera_input(
            "Take a picture"
            )
        if img_webcam is not None:
            bytes_raw = img_webcam.getvalue()
            bytes_arr = np.frombuffer(bytes_raw, np.uint8)
            img_arr_inv = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
            img_arr = cv2.cvtColor(img_arr_inv, cv2.COLOR_BGR2RGB)
            PIL.Image.fromarray(img_arr).save('webcam.png')
            with open('webcam.png', 'rb') as img_file:
                st.download_button(
                    label="Download image",
                    data=img_file,
                    file_name="picture.png",
                    mime="image/png"
                )
            application(img_arr)
    elif input_choice == input_methods[2]:
        img_url = st.text_input('Picture URL')
        if img_url != "":
            if validators.url(img_url):
                try:
                    response = requests.get(img_url, stream=True).raw
                    img_arr = np.array(PIL.Image.open(response))
                    application(img_arr, show_upload=True)
                except PIL.UnidentifiedImageError:
                    st.error('Invalid image URL. Please link the URL of the image, not the page!')
            else:
                st.error('Invalid image URL. Please check that you copied the correct URL!')


if __name__ == '__main__':
    main()



