# import csv
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# from bs4 import BeautifulSoup as bs
# import time

# # Define the base URL
# base_url = 'https://www.flipkart.com/search?q=laptop&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&as-pos=1&as-type=HISTORY&as-backfill=on&page='

# # Set up the Selenium driver with options
# options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # Run in headless mode (no GUI)
# options.add_argument('--no-sandbox')
# options.add_argument('--disable-dev-shm-usage')

# # Initialize the driver
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# # Prepare data for CSV
# laptop_data = []

# try:
#     # Loop through pages
#     for page in range(1, 42):  # Adjust the range for the number of pages to scrape
#         print(f"Scraping page {page}...")
        
#         # Construct the URL for the current page
#         url = f"{base_url}{page}"
        
#         # Retry logic for pages
#         max_retries = 3
#         retries = 0
#         while retries < max_retries:
#             try:
#                 driver.get(url)
#                 WebDriverWait(driver, 30).until(
#                     EC.presence_of_element_located((By.CLASS_NAME, '_75nlfW'))
#                 )
#                 break  # Break if successful
#             except Exception as e:
#                 retries += 1
#                 print(f"Retrying page {page} ({retries}/{max_retries})...")
#                 time.sleep(2)  # Delay between retries
#         if retries == max_retries:
#             print(f"Failed to scrape page {page} after {max_retries} retries. Skipping...")
#             continue
        
#         # Add delay between pages
#         time.sleep(2)

#         # Extract the page source
#         html_content = driver.page_source
        
#         # Parse the HTML with BeautifulSoup
#         soup = bs(html_content, 'html.parser')
        
#         # Extract product links
#         product_links = soup.find_all('a', class_='CGtC98')  # Update class name to match Flipkart's structure
        
#         for product in product_links:
#             try:
#                 # Construct full product URL
#                 product_url = 'https://www.flipkart.com' + product.get('href')
#                 driver.get(product_url)

#                 # Wait for product page to load
#                 WebDriverWait(driver, 30).until(
#                     EC.presence_of_element_located((By.CLASS_NAME, '_39kFie'))
#                 )
                
#                 # Extract product details page source
#                 product_html = driver.page_source
#                 product_soup = bs(product_html, 'html.parser')

#                 # Extract laptop name
#                 laptop_name = product_soup.find('span', class_='VU-ZEz')
#                 laptop_name = laptop_name.text.strip() if laptop_name else 'N/A'

#                 # Extract price
#                 price = product_soup.find('div', class_='Nx9bqj')
#                 price = price.text.strip() if price else 'N/A'

#                 # Extract specifications
#                 specs_sections = product_soup.find_all('div', class_='GNDEQ-')
#                 spec_data = {}

#                 for section in specs_sections:
#                     rows = section.find_all('tr')
#                     for row in rows:
#                         key_cell = row.find('td', class_='+fFi1w')
#                         value_cell = row.find('td', class_='Izz52n')

#                         if key_cell and value_cell:
#                             key = key_cell.text.strip()
#                             value = value_cell.text.strip()
#                             spec_data[key] = value

#                 # Extract additional details
#                 processor = spec_data.get('Processor Name', 'N/A')
#                 ram = spec_data.get('RAM', 'N/A')
#                 storage = spec_data.get('SSD Capacity', 'N/A')
#                 os = spec_data.get('Operating System', 'N/A')
#                 display = spec_data.get('Screen Size', 'N/A')
#                 warranty = spec_data.get('Warranty Summary', 'N/A')
#                 screen_resolution = spec_data.get('Screen Resolution', 'N/A')
#                 weight = spec_data.get('Weight', 'N/A')
#                 clock_speed = spec_data.get('Processor Clock Speed', 'N/A')
#                 touchscreen = spec_data.get('Touchscreen', 'N/A')

#                 # Append data to the list
#                 laptop_data.append([
#                     laptop_name, price, processor, ram, os, storage,
#                     display, warranty, screen_resolution, weight,
#                     clock_speed, touchscreen
#                 ])

#             except Exception as e:
#                 print(f"Error scraping product details: {e}")
    
#     # Save data to CSV
#     with open('laptops_detailed.csv', 'w', newline='', encoding='utf-8') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(['Laptop Name', 'Price', 'Processor', 'RAM', 'Operating System', 'Storage', 'Display', 'Warranty', 'Screen Resolution', 'Weight', 'Clock Speed', 'Touchscreen'])  # Write header
#         csvwriter.writerows(laptop_data)  # Write data rows

#     print("Detailed laptop data saved to 'laptops_detailed.csv' successfully!")

# except Exception as e:
#     print(f"An error occurred: {e}")

# finally:
#     # Quit the driver
#     driver.quit()


# import matplotlib.pyplot as plt

# def plot_scraping_workflow():
#     plt.figure(figsize=(10, 6))
#     processes = ['Start', 'Access Flipkart', 'Parse HTML (BeautifulSoup)', 'Handle Dynamic Elements (Selenium)', 'Extract Data', 'Store in CSV', 'End']
#     positions = range(len(processes))

#     plt.bar(positions, [1] * len(processes), tick_label=processes, color='skyblue')
#     plt.title('Web Scraping Workflow for Flipkart')
#     plt.ylabel('Process Stage')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig('web_scraping_workflow.png')
#     plt.show()

# plot_scraping_workflow()


# import numpy as np 
# import pandas as pd 
# df=pd.read_csv('laptops_detailed.csv')
# df.head(10)
# df.info()
# df.isnull().sum()
# df.duplicated().sum()
# df=df.drop_duplicates()
# df.duplicated().sum()
# df.drop('Clock Speed',axis=1,inplace=True)
# df.head()
# df.isnull().sum()
# df.shape

# import seaborn as sns


# sns.displot(x='Storage',data=df,kde=True)
# sns.displot(x='Weight',data=df,kde=True)
# df['Weight'] = df['Weight'].astype(str).str.replace('g', '', case=False)
# df['Weight'] = df['Weight'].astype(str).str.replace('k', '', case=False)
# df['Weight'] = df['Weight'].astype(str).str.replace('s', '', case=False)
# df['Weight'] = (
#     df['Weight']
#     .astype(str)  # Ensure it's a string
#     .str.replace(',', '.', regex=False)  # Replace comma with a dot for decimals
#     .str.replace('kg', '', case=False, regex=False)  # Remove 'kg' (case-insensitive)
#     .str.strip()  # Remove leading and trailing spaces
#     .astype(float)  # Convert to float
# )

# # Clean RAM and Weight columns by removing 'GB' and 'Kg' (case-insensitive)
# df['RAM'] = df['RAM'].astype(str).str.replace('GB', '').astype(int)
# df['Weight'] = df['Weight'].astype(str).str.replace('kg', '', case=False).astype(float)

# # Clean Storage column (convert TB to GB)
# def convert_storage(value):
#     value_str = str(value)  # Convert the value to a string for checking
#     if 'TB' in value_str or 'tb' in value_str:
#         # Convert TB to GB (1 TB = 1024 GB)
#         return float(value_str.replace(' TB', '').replace('tb', '')) * 1024
#     else:
#         # Keep GB as it is
#         return float(value_str.replace(' GB', '').replace('gb', ''))

# # Apply the conversion to the Storage column
# df['Storage'] = df['Storage'].apply(convert_storage)

# # Display the cleaned DataFrame
# df.head(1)

# df.shape
# df.info()
# # Fill missing 'Storage' with rounded mean and 'Weight' with median
# df['Storage'].fillna(round(df['Storage'].median(), 1), inplace=True)
# df['Weight'].fillna(round(df['Weight'].median(), 1), inplace=True)

# # Check if missing values are filled
# df.isnull().sum()

# df['Price'].head(10)
# # Remove the '₹' symbol and commas, then convert the price column to numeric (integer)
# df['Price'] = df['Price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(int)

# # Display the cleaned DataFrame


# df['Price'].head(3)

# sns.displot(x='Price', data=df, kde=True)
# df.head(4)
# # Get unique values in 'Column1'
# unique_values = df['Screen Resolution'].unique()

# print("Unique values:", unique_values)
# df['Screen Resolution'] = df['Screen Resolution'].astype(str).str.replace('Pixel', '', case=False)
# df['Screen Resolution'] = df['Screen Resolution'].astype(str).str.replace('Pixels', '', case=False)
# import pandas as pd
# import re



# # Function to clean and extract resolution
# def extract_resolution(resolution):
#     # Convert to lowercase and strip spaces
#     resolution = resolution.lower().strip()

#     # Remove words that indicate warranty (e.g., 'warranty', 'year', 'manufacturer', etc.)
#     if any(keyword in resolution for keyword in ['warranty', 'year', 'manufacturer', 'on-site']):
#         return None, None  # Ignore these as they are not resolutions

#     # Replace multiple symbols (like 'X', '*', '×', etc.) with 'x'
#     resolution = re.sub(r'[^0-9x]', ' ', resolution)  # Remove non-numeric characters except 'x'
#     resolution = resolution.replace('×', 'x').replace('X', 'x').replace('*', 'x')

#     # Remove extra spaces and ensure correct format
#     resolution = re.sub(r'\s+', ' ', resolution)  # Replace multiple spaces with a single space
#     resolution = re.sub(r'(\d+)\s*x\s*(\d+)', r'\1x\2', resolution)  # Ensure 'width x height' format

#     # Extract width and height
#     match = re.search(r'(\d{3,5})x(\d{3,5})', resolution)  # Match only valid resolutions
#     if match:
#         return int(match.group(1)), int(match.group(2))
    
#     return None, None  # Return None if no valid resolution found

# # Apply the function to extract width and height
# df[['Width', 'Height']] = df['Screen Resolution'].apply(lambda x: pd.Series(extract_resolution(str(x))))

# # Find and print rows with missing values for debugging
# missing_rows = df[df['Width'].isna() | df['Height'].isna()]
# print("Rows with missing values:\n", missing_rows)

# # Handling missing values (choose one option)
# df.dropna(subset=['Width', 'Height'], inplace=True)  # Option 1: Remove invalid rows
# # df.fillna({'Width': 1920, 'Height': 1080}, inplace=True)  # Option 2: Fill missing values with a default



# df.isnull().sum()
# df
# df.drop('Screen Resolution',axis=1,inplace=True)
# df.head(1)
# # Get unique values in 'Column1'
# unique_values = df['Display'].unique()

# print("Unique values:", unique_values)

# import pandas as pd
# import re




# # Function to extract the display size in inches
# def extract_display_size(display_str):
#     match = re.search(r'\(([\d.]+)\s*inch', display_str, re.IGNORECASE)
#     if match:
#         return float(match.group(1))  # Extract and convert to float
#     return None  # Return None if no match is found

# # Apply the function to extract inches
# df['Display'] = df['Display'].apply(extract_display_size)



# df.head(5)
# df.isnull().sum()
# df.shape
# # Get unique values in 'Column1'
# unique_values = df['Processor'].unique()

# print("Unique values:", unique_values)

# df['Processor'].value_counts()
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df contains 'Processor' and 'Price' columns
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='Processor', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("Processor", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("Processor vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()

# import pandas as pd

# # Define a threshold for categorizing lesser-used processors into 'Other'
# LESS_USED_THRESHOLD = 5  # Adjust based on dataset distribution

# # Count occurrences of each processor type
# processor_counts = {
#     "i9": 9, "i7": 111, "i5": 226, "i3": 143,
#     "Ultra 9": 6, "Ultra 7": 22, "Ultra 5": 18,
#     "Core 9": 6, "Core 7": 3, "Core 5": 7, "Core N": 1,
#     "Ryzen 9": 3, "Ryzen 7": 43, "Ryzen 5": 53, "Ryzen 3": 30,
#     "Ryzen Z1": 1, "M1": 4, "M2": 11, "M3": 21, "M4": 5,
#     "Snapdragon Elite": 6, "Snapdragon Plus": 5,
#     "Pentium": 2, "Celeron": 27, "Athlon": 4, "MediaTek": 1,
#     "Dual Core": 3
# }

# def categorize_processor(processor_name):
#     processor_name = processor_name.lower()

#     # Common processor categories
#     categories = {
#         "core i9": "i9", "core i7": "i7", "core i5": "i5", "core i3": "i3",
#         "core ultra 9": "Ultra 9", "core ultra 7": "Ultra 7", "core ultra 5": "Ultra 5",
#         "core 9": "Core 9", "core 7": "Core 7", "core 5": "Core 5", "core n": "Core N",
#         "ryzen 9": "Ryzen 9", "ryzen 7": "Ryzen 7", "ryzen 5": "Ryzen 5", "ryzen 3": "Ryzen 3",
#         "ryzen z1": "Ryzen Z1", "m1": "M1", "m2": "M2", "m3": "M3", "m4": "M4",
#         "snapdragon x elite": "Snapdragon Elite", "snapdragon x plus": "Snapdragon Plus",
#         "pentium": "Pentium", "celeron": "Celeron", "athlon": "Athlon", "mediatek": "MediaTek",
#         "dual core": "Dual Core"
#     }

#     for key, value in categories.items():
#         if key in processor_name:
#             # Ensure that categories with sufficient count are kept, else classify as 'Other'
#             if processor_counts.get(value, 0) < LESS_USED_THRESHOLD:
#                 return "Other"
#             return value

#     return "Other"



# # Apply categorization
# df["Processor"] = df["Processor"].apply(categorize_processor)

# print(df)

# df['Processor'].value_counts()

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df contains 'Processor' and 'Price' columns
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='Processor', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("Processor", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("Processor vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt


# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='Operating System', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("Operating System", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("Operating System vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()

# # Get unique values in 'Column1'
# unique_values = df['Operating System'].unique()

# print("Unique values:", unique_values)
# def cat_os(inp):
#     # Normalize input to lowercase
#     inp = str(inp).lower()

#     # Define OS categories using sets
#     windows_variants = {'windows 11 home', 'windows 11 pro', 'windows 10 home', 'windows 10 pro', 'windows 10'}
#     mac_variants = {'macos sequoia', 'mac os monterey', 'macos sonoma', 'mac os big sur'}

#     if inp in windows_variants:
#         return 'Windows'
#     elif inp in mac_variants:
#         return 'Mac'
#     elif 'chrome' in inp:
#         return 'Chrome OS'
#     elif 'ubuntu' in inp or 'prime os' in inp or 'dos' in inp:
#         return 'Others/No OS/Linux'
#     else:
#         return 'Others/No OS/Linux'

# df['Operating System']=df['Operating System'].apply(cat_os)
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df contains 'Processor' and 'Price' columns
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='Operating System', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("Operating System", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("Operating System vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()

# df.head()
# correlation_matrix = df.select_dtypes(include=['number']).corr()
# correlation_matrix['Price']



# # Function to calculate PPI
# def calculate_ppi(row):
#     width, height, screen_size = row['Width'], row['Height'], row['Display']
#     return np.sqrt(width**2 + height**2) / screen_size

# # Apply function to create 'PPI' column
# df['PPI'] = df.apply(calculate_ppi, axis=1)


# df.head(1)
# correlation_matrix = df.select_dtypes(include=['number']).corr()
# correlation_matrix['Price']

# df = df.drop(columns=['Width', 'Height', 'Display'])

# df.head()
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df contains 'Processor' and 'Price' columns
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='RAM', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("RAM", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("RAM vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df contains 'Processor' and 'Price' columns
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='Storage', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("Storage", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("Storage vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()

# df.head()
# df['Laptop Name'].head(10)
# df["Company"] = df["Laptop Name"].apply(lambda x: x.split()[0])

# df.head()
# df = df.drop(columns=['Laptop Name'])

# df.head(1)
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df contains 'Processor' and 'Price' columns
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='Company', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("Company", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("Company vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()


# import pandas as pd
# import re



# # Function to extract warranty duration in years
# def extract_warranty_years(warranty_text):
#     warranty_text = warranty_text.lower()
    
#     # Extract numbers followed by "year", "yr", or "years"
#     match = re.search(r'(\d+)\s*(?:year|yr|years)', warranty_text)
    
#     if match:
#         return int(match.group(1))  # Extract the numeric value
    
#     # Extract months and convert to years
#     match = re.search(r'(\d+)\s*months?', warranty_text)
#     if match:
#         return round(int(match.group(1)) / 12, 2)  # Convert months to years
    
#     return 0  # Default if no valid warranty info found

# # Apply function
# df["Warranty_Years"] = df["Warranty"].apply(extract_warranty_years)
# print(df)

# # Binary column: Does the laptop have a warranty?
# df["Has_Warranty"] = df["Warranty_Years"].apply(lambda x: 1 if x > 0 else 0)

# # Binary column: Long warranty (More than 2 years)
# df["Long_Warranty"] = df["Warranty_Years"].apply(lambda x: 1 if x > 2 else 0)

# # Binary column: Short warranty (1 year or less)
# df["Short_Warranty"] = df["Warranty_Years"].apply(lambda x: 1 if x <= 1 else 0)

# print(df[["Warranty_Years", "Has_Warranty", "Long_Warranty", "Short_Warranty"]])

# def categorize_warranty_type(warranty_text):
#     warranty_text = warranty_text.lower()
#     if "onsite" in warranty_text:
#         return "Onsite"
#     elif "carry" in warranty_text:
#         return "Carry-in"
#     elif "international" in warranty_text:
#         return "International"
#     elif "domestic" in warranty_text:
#         return "Domestic"
#     else:
#         return "Other"

# df["Warranty_Type"] = df["Warranty"].apply(categorize_warranty_type)
# print(df[["Warranty", "Warranty_Type"]])

# df = df.drop(columns=['Warranty'])

# df.head(1)
# correlation_matrix = df.select_dtypes(include=['number']).corr()
# correlation_matrix['Price']

# df = df.drop(columns=['Warranty_Years','Has_Warranty','Long_Warranty','Short_Warranty'])


# df.head()
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df contains 'Processor' and 'Price' columns
# plt.figure(figsize=(12, 6))
# sns.barplot(data=df, x='Warranty_Type', y='Price', palette='viridis')

# # Improve readability
# plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# plt.xlabel("Warranty_Type", fontsize=12)
# plt.ylabel("Price (₹)", fontsize=12)
# plt.title("Warranty_Type vs Price", fontsize=14)
# plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.show()

# df.info()
# df = pd.get_dummies(df, columns=["Warranty_Type"], drop_first=True)
# df[df.filter(like="Warranty_Type_").columns] = df.filter(like="Warranty_Type_").astype(int)

# correlation_matrix = df.select_dtypes(include=['number']).corr()
# correlation_matrix['Price']

# # Compute correlation matrix
# correlation_matrix = df.select_dtypes(include=['number']).corr()

# # Set figure size
# plt.figure(figsize=(6, 4))

# # Generate heatmap
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# # Title
# plt.title("Feature Correlation Heatmap")

# # Show plot
# plt.show()
# df.head()
# df = df.drop(columns=['Warranty_Type_Domestic','Warranty_Type_International','Warranty_Type_Onsite','Warranty_Type_Other'])

# df.head()
# from sklearn.preprocessing import LabelEncoder

# # Initialize the LabelEncoder
# le = LabelEncoder()

# # Fit and transform the 'Touch Screen' column
# df['Touchscreen'] = le.fit_transform(df['Touchscreen']) #yes=1 and No=0

# df.head()
# X=df.drop(columns=['Price'])
# y=df['Price']
# X.head()
# y.head()
# from sklearn.model_selection import train_test_split

# # Splitting data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42  # Adjust test_size and random_state as needed
# )



# X_train.head(1)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import r2_score, mean_absolute_error

# # List of regression models to compare
# models = {
#     "Linear Regression": LinearRegression(),
#     "Decision Tree": DecisionTreeRegressor(max_depth=20),
#     "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
#     "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
# }

# # Preprocessing step (One-Hot Encoding for categorical columns)
# step1 = ColumnTransformer(
#     transformers=[
#         ('onehot', OneHotEncoder(drop='first', sparse_output=False), [0, 2, 7])  # Encoding categorical columns
#     ],
#     remainder='passthrough'  # Leave numerical columns unchanged
# )

# # Dictionary to store results
# results = {}

# plt.figure(figsize=(12, 8))

# # Train & Evaluate each model
# for i, (name, model) in enumerate(models.items()):
#     # Create a pipeline
#     pipe = Pipeline([
#         ('step1', step1),
#         ('model', model)
#     ])

#     # Train the model
#     pipe.fit(X_train, y_train)

#     # Predictions
#     y_pred = pipe.predict(X_test)

#     # Metrics
#     r2 = r2_score(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     results[name] = {"R2 Score": r2, "MAE": mae}

#     # Scatter plot for each model
#     plt.subplot(2, 2, i + 1)
#     plt.scatter(y_test, y_pred, alpha=0.6)
#     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')  # Ideal line
#     plt.xlabel("Actual Price")
#     plt.ylabel("Predicted Price")
#     plt.title(f"{name}\nR²: {r2:.3f}, MAE: {mae:.2f}")

# # Show scatter plots
# plt.tight_layout()
# plt.show()

# # Print Model Comparison
# print("Model Performance Comparison:")
# for model, metrics in results.items():
#     print(f"{model}: R² = {metrics['R2 Score']:.4f}, MAE = {metrics['MAE']:.2f}")

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import r2_score, mean_absolute_error

# # Step 1: Preprocessing (One-Hot Encoding for categorical columns)
# step1 = ColumnTransformer(
#     transformers=[
#         ('onehot', OneHotEncoder(drop='first', sparse_output=False), [0, 2, 7])  # Encoding categorical columns
#     ],
#     remainder='passthrough'  # Leave numerical columns unchanged
# )

# # Step 2: Model - Use RandomForestRegressor instead of GradientBoostingRegressor
# step2 = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)

# # Combine into a pipeline
# pipe = Pipeline([
#     ('step1', step1),
#     ('step2', step2)
# ])

# # Train the pipeline
# pipe.fit(X_train, y_train)

# # Predictions
# y_pred = pipe.predict(X_test)

# # Evaluation
# print('R2 score:', r2_score(y_test, y_pred))
# print('MAE:', mean_absolute_error(y_test, y_pred))

# import pickle
# pickle.dump(df,open('df.pkl','wb'))
# pickle.dump(pipe,open('pipe.pkl','wb'))





# # Save processed data
# import pickle
# with open("df.pkl", "wb") as f:
#     pickle.dump(df, f)

# # Train a model (assuming variable 'pipe' is trained)
# with open("pipe.pkl", "wb") as f:
#     pickle.dump(pipe, f)

# # Push to GitHub
# import subprocess
# subprocess.run("git config --global user.email 'you@example.com'", shell=True)
# subprocess.run("git config --global user.name 'GitHub Actions'", shell=True)
# subprocess.run("git add df.pkl pipe.pkl", shell=True)
# subprocess.run("git commit -m 'Weekly model update' || echo 'Nothing to commit'", shell=True)
# subprocess.run("git push origin main", shell=True)
