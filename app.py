import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Model architecture classes
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class CNN_NeuralNet(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Load trained model
@st.cache_resource
def load_model():
    model = CNN_NeuralNet(in_channels=3, num_diseases=38)
    model.load_state_dict(torch.load('Plant_Leaves_disease_detection.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("🌱 PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on kaggle.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

# Class names (update with your actual classes)
class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Prediction Page
st.title('🌱Disease Detector')
st.write("Upload plant leaf images and determine whether the leaves are healthy or diseased")  

test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
if test_image is not None:
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
if st.button("Predict"):
    if test_image is None:
        st.warning("Please upload an image first!")
    else:
        # Preprocess and predict
        image = Image.open(test_image)
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(image_tensor)
            _, prediction = torch.max(output, 1)
            result = class_names[prediction.item()]

        # Process result for display
        disease_info = result.split("_")
        plant_name = disease_info[0].replace('_', ' ').split('(')[0].strip()
        condition = disease_info[-1].replace('_', ' ').strip()
        
        # Format final output
        if 'healthy' in condition.lower():
            display_text = f"{plant_name} healthy"
        else:
            display_text = f"{plant_name} {condition} disease"

        # Show results
        st.snow()
        st.subheader("Model Prediction")
        st.write(f'**Class:** "{plant_name}"')  # <-- Added class display line
        st.success(f"🌿 **{display_text.capitalize()}**")
        
        # Add confidence score
        confidence = torch.max(F.softmax(output, dim=1)).item() * 100
        st.write(f"🔍 Confidence: {confidence:.2f}%")




















# import streamlit as st
# import torch
# from torchvision import transforms
# from PIL import Image
# import torch.nn as nn
# import torch.nn.functional as F

# # Model architecture classes
# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, labels = batch
#         out = self(images)
#         loss = F.cross_entropy(out, labels)
#         return loss

#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)
#         loss = F.cross_entropy(out, labels)
#         acc = accuracy(out, labels)
#         return {'val_loss': loss.detach(), 'val_acc': acc}

# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# def ConvBlock(in_channels, out_channels, pool=False):
#     layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#               nn.BatchNorm2d(out_channels),
#               nn.ReLU(inplace=True)]
#     if pool: layers.append(nn.MaxPool2d(4))
#     return nn.Sequential(*layers)

# class CNN_NeuralNet(ImageClassificationBase):
#     def __init__(self, in_channels, num_diseases):
#         super().__init__()
#         self.conv1 = ConvBlock(in_channels, 64)
#         self.conv2 = ConvBlock(64, 128, pool=True)
#         self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
#         self.conv3 = ConvBlock(128, 256, pool=True)
#         self.conv4 = ConvBlock(256, 512, pool=True)
#         self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
#         self.classifier = nn.Sequential(
#             nn.MaxPool2d(4),
#             nn.Flatten(),
#             nn.Linear(512, num_diseases))

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.res1(out) + out
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.res2(out) + out
#         out = self.classifier(out)
#         return out

# # Load trained model
# @st.cache_resource
# def load_model():
#     model = CNN_NeuralNet(in_channels=3, num_diseases=38)
#     model.load_state_dict(torch.load('Plant_Leaves_disease_detection.pth', map_location='cpu'))
#     model.eval()
#     return model

# model = load_model()

# #Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# #Main Page
# if(app_mode=="Home"):
#     st.header("PLANT DISEASE RECOGNITION SYSTEM")
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍
    
#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)

# #About Project
# elif(app_mode=="About"):
#     st.header("About")
#     st.markdown("""
#                 #### About Dataset
#                 This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
#                 This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
#                 A new directory containing 33 test images is created later for prediction purpose.
#                 #### Content
#                 1. train (70295 images)
#                 2. test (33 images)
#                 3. validation (17572 images)

#                 """)

# # Class names (update with your actual classes)
# class_names = [
#                 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
#                 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
#                 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
#                 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
#                 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#                 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
#                 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
#                 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
#                 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
#                 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
#                 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                 'Tomato___healthy'
# ]

# # Image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])

# # Streamlit UI
# st.title('🌱Disease Detector')
# st.write("Upload plant leaf images and determine whether the leaves are healthy or diseased")  

# uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     if st.button('Predict'):
        
#         # Preprocess image
#         image_tensor = transform(image).unsqueeze(0)
        
        
#         # Prediction
#         with torch.no_grad():
#             output = model(image_tensor)
#             _, prediction = torch.max(output, 1)
#             result = class_names[prediction.item()]
        
#         # Display result
#         disease_info = result.split("___")
#         st.snow()
#         st.subheader(f'Model is Predicting its a: {disease_info[-1].replace("_", " ")}')
#         st.write(f'**Model is predicting its a**: {disease_info[0]}')