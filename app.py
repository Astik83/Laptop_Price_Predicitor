import streamlit as st
import pickle
import math
import pandas as pd  # Import pandas

# Load the model and dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Streamlit app
st.title("Laptop Price Predictor")

# Brand selection
company = st.selectbox('Select Brand', df['Company'].unique())

# Type of laptop
type_name = st.selectbox('Type of Laptop', df['TypeName'].unique())

# RAM size
ram = st.selectbox('RAM (in GB)', [4,6, 8, 12, 16, 32])

# Weight of the laptop
weight = st.number_input('Weight (in kg)', min_value=0.5, max_value=5.0, value=1.5)

# Touchscreen option
touchscreen = st.selectbox('Does it have a Touchscreen?', ['Yes', 'No'])
touchscreen = 1 if touchscreen == 'Yes' else 0

# IPS display option
ips = st.selectbox('Does it have an IPS display?', ['Yes', 'No'])
ips = 1 if ips == 'Yes' else 0

# Select screen resolution
screen_resolution = st.selectbox('Select Screen Resolution', [
    '1920x1080', '1366x768', '2560x1600', '3840x2160', '2560x1440', '1280x800',
    '3200x1800', '1920x1200', '2880x1800', '3840x2400'
])

# Convert screen resolution to width and height
resolution_map = {
    '1920x1080': (1920, 1080),
    '1366x768': (1366, 768),
    '2560x1600': (2560, 1600),
    '3840x2160': (3840, 2160),
    '2560x1440': (2560, 1440),
    '1280x800': (1280, 800),
    '3200x1800': (3200, 1800),
    '1920x1200': (1920, 1200),
    '2880x1800': (2880, 1800),
    '3840x2400': (3840, 2400)
}

# Get the screen resolution width and height
width, height = resolution_map[screen_resolution]

# Input for diagonal screen size (in inches), allowing point increments (e.g., 13.3 to 14)
diagonal_size = st.number_input('Enter Screen Diagonal Size (in inches)', min_value=10.0, max_value=20.0, value=15.0, step=0.1)

# Calculate PPI (Pixels Per Inch)
ppi = math.sqrt(width**2 + height**2) / diagonal_size

# Display calculated PPI
st.write(f"The calculated PPI for the selected screen resolution is: {ppi:.2f}")

# CPU details and other inputs (corrected unique method usage)
cpu_clock_speed = st.number_input('CPU Clock Speed (GHz)', min_value=1.0, max_value=5.0, value=2.5)
cpu_generation = st.selectbox('CPU Generation', df['Cpu_Generation_Category'].unique())
cpu_type = st.selectbox('CPU Type', df['Cpu_Type_Category'].unique())  # Corrected
ssd = st.selectbox('SSD Size (in GB)', [0, 128, 256, 512, 1024])
hdd = st.selectbox('HDD Size (in GB)', [0, 128, 256, 512, 1024, 2048])
gpu_brand = st.selectbox('GPU Brand', df['Gpu brand'].unique())  # Corrected the column name
os = st.selectbox('Operating System', df['os'].unique())  # Corrected the column name

# Prediction button
if st.button('Predict Price'):
    # Prepare the input data for prediction
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips],
        'ppi': [ppi],
        'Cpu_Clock_Speed': [cpu_clock_speed],
        'Cpu_Generation_Category': [cpu_generation],
        'Cpu_Type_Category': [cpu_type],
        'SSD': [ssd],
        'HDD': [hdd],
        'Gpu brand': [gpu_brand],
        'os': [os]
    })

    # Make the prediction using the pre-trained pipeline
    prediction_log = pipe.predict(input_data)  # Prediction on log scale

    # Reverse the log transformation to get the actual price
    prediction = math.exp(prediction_log[0])

    # Display the prediction
    st.write(f"The predicted price of the laptop is: ₹{prediction:,.2f}")

# Footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 20px;
            background-color: #2C3E50;
            color: white;
            font-size: 14px;
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
            margin: 0; /* Ensure no margin */
            box-sizing: border-box; /* Ensures the width is calculated correctly */
        }
        .footer a {
            color: #3498db;
            text-decoration: none;
        }
        .footer p {
            margin: 0;
        }
    </style>
    <div class="footer">
        <p>&copy; 2025 Astik Shah. All Rights Reserved. | <a href="https://www.linkedin.com/in/astik-shah-04aa46344/" target="_blank">LinkedIn</a></p>
    </div>
""", unsafe_allow_html=True)

