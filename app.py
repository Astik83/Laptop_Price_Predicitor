import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ----------------------- Premium CSS Styling -----------------------
st.markdown("""
    <style>
        /* Import a modern Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif;
        }

        /* Overall app background with a background image and soft overlay */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1593642702821-c8da6771f0c6?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-position: center;
        }

        /* Transparent container background */
        .transparent-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 30px;
            margin: 20px auto;
            max-width: 800px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }

        /* Header styling */
        h1 {
            color: #2C3E50;
            font-weight: 600;
            text-align: center;
            margin-bottom: 20px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }

        /* Centered subtitle */
        .subheader {
            text-align: center;
            color: #7F8C8D;
            margin-bottom: 30px;
            color:#2C3E50;

        }

        /* Button styling */
        .stButton > button {
            width: 100%;
            background-color: #3498db;
            color: white;
            font-weight: 600;
            font-size: 1.1em;
            padding: 10px;
            border: none;
            border-radius: 8px;
            transition: transform 0.2s;
        }
        .stButton > button:hover {
            background-color: #2980b9;
            transform: scale(1.05);
        }

        /* Prediction box styling */
        .prediction-box {
            background-color: #F8F8F8;
            border-left: 5px solid #3498db;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .prediction-box h3 {
            margin: 0;
            color: #2C3E50;
            font-weight: 600;
        }
        .prediction-box p {
            margin: 10px 0;
            color: #34495e;
        }

        /* Recommended laptop card styling */
        .card {
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin: 10px;
            text-align: center;
        }
        .card img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 10px;
        }
        .card h4 {
            margin: 10px 0 5px;
            color: #2C3E50;
        }
        .card p {
            margin: 5px 0;
            color: #7F8C8D;
        }
        .card a {
            text-decoration: none;
        }
        .card button {
            width: 100%;
            background-color: #3498db;
            color: white;
            font-weight: 600;
            padding: 8px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .card button:hover {
            background-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------- Data and Model Loading -----------------------
# Common screen resolutions
resolution_options = {
    "1366x768": (1366, 768),
    "1920x1080 (Full HD)": (1920, 1080),
    "1920x1200": (1920, 1200),
    "2560x1444 (2K)": (2560, 1440),
    "3840x2160 (4K UHD)": (3840, 2160),
    "1280x800": (1280, 800),
    "1440x900": (1440, 900),
    "1600x900": (1600, 900),
    "2560x1600": (2560, 1600),
    "3200x1800": (3200, 1800),
    "2880x1864": (2880, 1864),
}

# Load the pre-trained pipeline and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# ----------------------- App Header -----------------------
st.markdown("<h1>Laptop Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Predict the market value of your ideal laptop configuration.</p>",
            unsafe_allow_html=True)

# ----------------------- Main Form -----------------------
with st.container():
    with st.form("prediction_form"):
        st.markdown("<div class='transparent-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            company = st.selectbox('Brand', df['Company'].unique())
            processor = st.selectbox('Processor', df['Processor'].unique())
            ram = st.selectbox('RAM (GB)', df['RAM'].unique())
        with col2:
            os = st.selectbox('Operating System', df['Operating System'].unique())
            storage = st.selectbox('Storage (GB)', df['Storage'].unique())
            weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5)

        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
        selected_resolution = st.selectbox("Screen Resolution", list(resolution_options.keys()))
        width, height = resolution_options[selected_resolution]
        diagonal_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)

        # Calculate Pixels Per Inch (PPI)
        ppi = np.sqrt(width ** 2 + height ** 2) / diagonal_size

        submitted = st.form_submit_button("Calculate Price")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            input_data = pd.DataFrame({
                'Company': [company],
                'Processor': [processor],
                'RAM': [ram],
                'Operating System': [os],
                'Storage': [storage],
                'Weight': [weight],
                'Touchscreen': [1 if touchscreen == 'Yes' else 0],
                'PPI': [ppi]
            })

            # Make prediction
            prediction = pipe.predict(input_data)[0]

            # Display prediction result in a premium styled box
            st.markdown(f"""
                <div class="prediction-box">
                    <h3>Estimated Value</h3>
                    <p style="font-size: 2em; color: #3498db;">₹{prediction:,.2f} </p>
                    <p>Based on Standard market trends and specifications</p>
                </div>
            """, unsafe_allow_html=True)

# ----------------------- Recommended Laptops Section -----------------------
st.markdown("<h2 style='text-align: center; color:#2C3E50;'>Recommended Laptops</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color:#2C3E50; margin-bottom: 30px;'>Explore some of the best laptop options available online</p>",
    unsafe_allow_html=True)


# Sample Laptop Data with image URLs and detail URLs
def get_online_laptop_data():
    return [
        {
            "name": "HP Intel",
            "image": "https://rukminim2.flixcart.com/image/832/832/xif0q/computer/k/i/f/-original-imah4qscfq3ddvc7.jpeg?q=70&crop=false",
            "price": "₹51,490",
            "specs": ["Intel i5", "16GB RAM", "512GB SSD", "15.6\" FHD Display", "1.69kg"],
            "url": "https://www.flipkart.com/hp-intel-core-i5-12th-gen-1235u-16-gb-512-gb-ssd-windows-11-home-15s-fq5330tu-15-fd0111tu-thin-light-laptop/p/itm5b5f94f7b044c"
        },
        {
            "name": "HP Pavilion ",
            "image": "https://rukminim2.flixcart.com/image/416/416/xif0q/computer/x/p/x/-original-imah4qsjagxxggnw.jpeg?q=70&crop=false",
            "price": "₹67,990",
            "specs": ["Intel i5", "16GB RAM", "512GB SSD", "15.6\" FHD Display", "1.75kg"],
            "url": "https://www.flipkart.com/hp-pavilion-intel-core-i5-13th-gen-1340p-16-gb-512-gb-ssd-windows-11-home-15-eg3079tu-thin-light-laptop/p/itm4ae6958287d36"
        },
        {
            "name": "DELL Vostro",
            "image": "https://rukminim2.flixcart.com/image/832/832/xif0q/computer/d/g/8/-original-imah9gv6n4wsxd2e.jpeg?q=70&crop=false",
            "price": "₹51,490",
            "specs": ["Intel i5", "8GB RAM", "512GB SSD", "14\" FHD Display", "1.66kg"],
            "url": "https://www.flipkart.com/dell-vostro-3530-intel-core-i5-13th-gen-1334u-8-gb-512-gb-ssd-windows-11-home-thin-light-laptop/p/itm68eb02f2bb17d"
        },
        {
            "name": "MacBook Air",
            "image": "https://rukminim2.flixcart.com/image/832/832/xif0q/computer/m/7/y/-original-imagypv6datec8tp.jpeg?q=70&crop=false",
            "price": "₹1,34,999",
            "specs": ["M3", "8GB RAM", "512GB SSD", "15\" 2880x1864", "1.51kg"],
            "url": "https://www.flipkart.com/apple-macbook-air-m3-8-gb-512-gb-ssd-macos-sonoma-mryv3hn-a/p/itmb1aa0cc739560"
        },
    ]


laptop_data = get_online_laptop_data()

# 3-column layout for the laptop cards
cols = st.columns(3)

# Loop through each laptop and display in respective column
for idx, laptop in enumerate(laptop_data):
    with cols[idx % 3]:
        st.markdown(f"""
            <div class="card">
                <img src="{laptop['image']}" alt="{laptop['name']}">
                <h4>{laptop['name']}</h4>
                <p style="font-size: 1.2em; color: #3498db;"><b>{laptop['price']}</b></p>
                <p>{' • '.join(laptop['specs'])}</p>
                <div style="margin-top: 10px;">
                    <a href="{laptop['url']}" target="_blank">
                        <button>View Details</button>
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ----------------------- Additional Flipkart Link -----------------------
st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <a href="https://www.flipkart.com" target="_blank">
            <button style="background-color: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-size: 1.1em;">
                Visit Flipkart for More Laptops
            </button>
        </a>
    </div>
""", unsafe_allow_html=True)
