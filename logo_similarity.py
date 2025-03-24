from fastparquet import ParquetFile
from IPython.display import display
import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import requests
from urllib.parse import urljoin

# Reading Parquet file
df = ParquetFile('logos.snappy.parquet').to_pandas()
display(df)

# Data existence check
if df.empty or "domain" not in df.columns:
    raise ValueError("The file does not contain the 'domain' column")

# Initialize the YOLOv8 model
model = YOLO("yolov8m.pt")

# Create main directory for all logos
main_logo_dir = "logos"
os.makedirs(main_logo_dir, exist_ok=True)

# Function to create folder for each site
def create_website_folder(website):
    website_name = website.split("//")[-1].replace(".", "_")
    website_folder = os.path.join(main_logo_dir, website_name)
    os.makedirs(website_folder, exist_ok=True)
    return website_folder

# Function for extracting images from a website
def get_images_from_website(url):
    chrome_path = "name***"
    options = Options()
    options.add_argument("--headless")
    options.binary_location = chrome_path

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    images = []
    for img_tag in soup.find_all("img"):
        img_url = img_tag.get("src")
        if img_url:
            images.append(urljoin(url, img_url))
    return images

# Function for downloading the image
def download_image(image_url, index, website_folder):
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        img_path = os.path.join(website_folder, f"image_{index}.jpg")
        with open(img_path, "wb") as file:
            file.write(response.content)
        return img_path
    return None

# Function for detecting logos with the model: YOLOv8
def detect_logo(image_path, website_folder):
    results = model(image_path)
    image = cv2.imread(image_path)

    # check if the model returned results
    if not results or len(results) == 0:
        print(f" No results found for: {image_path}")
        return

    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()  # Extrage bounding boxes

        #Boxes empty :
        if boxes.size == 0:
            print(f"No detection in: {image_path}")
            continue

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_logo = image[y1:y2, x1:x2]
            logo_filename = os.path.join(website_folder, f"logo_{i}_{j}.png")
            cv2.imwrite(logo_filename, cropped_logo)
            print(f" Saved logo : {logo_filename}")

# Function for counting empty folders
def count_empty_folders(directory):
    empty_count = 0
    total_folders = 0
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            if not os.listdir(folder_path):  # if folder is empty
                empty_count += 1
            total_folders += 1
    print(f"Total number of folders: {total_folders}")
    print(f"Number of empty folders: {empty_count}")
    return empty_count, total_folders

# Function for extracting logos directly from the website
def extract_logo_direct_from_site(website, website_folder):
    website_url = "https://" + website
    image_urls = get_images_from_website(website_url)
    if not image_urls:
        print(f" No image found for : {website}")
        return

    # search for images with the class 'logo' or in the image URL
    filtered_images = []
    for img_url in image_urls:
        if 'logo' in img_url:  # filter
            filtered_images.append(img_url)

    # If we didn't find any images containing 'logo', we look for images with the class 'logo'
    if not filtered_images:
        chrome_path = "name ***"
        options = Options()
        options.add_argument("--headless")
        options.binary_location = chrome_path
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(website_url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        for img_tag in soup.find_all("img"):
            if "logo" in img_tag.get("class", []):
                img_url = img_tag.get("src")
                if img_url:
                    filtered_images.append(urljoin(website_url, img_url))

    # If we found images, we download and process them:
    if filtered_images:
        for img_index, img_url in enumerate(filtered_images):
            img_path = download_image(img_url, img_index, website_folder)
            if img_path:
                detect_logo(img_path, website_folder)
    else:
        print(f"No relevant images were found on the : {website}")

# search function with Clearbit :
def get_logo_clearbit(website, website_folder):
    website_url = "https://logo.clearbit.com/" + website
    image_urls = get_images_from_website(website_url)

    # If the logo is accessible, I process it:
    if image_urls:
        for img_index, img_url in enumerate(image_urls):
            img_path = download_image(img_url, img_index, website_folder)
            if img_path:
                detect_logo(img_path, website_folder)
    else:
        print(f"Clearbit failed for :  {website}")

# Processing for every website
for index in range(0, 20, 1):
    website = df["domain"].iloc[index]
    print(f"Processing for  : {website}")

    # create folder for this website
    website_folder = create_website_folder(website)

    # search :  Clearbit
    get_logo_clearbit(website, website_folder)

    # if empty_folder search website
    empty_folders, total_folders = count_empty_folders(main_logo_dir)
    if empty_folders > 0:
        print(f"The folder is empty for  {website},try directly from the website.")
        extract_logo_direct_from_site(website, website_folder)

    #count empty folders
    empty_folders, total_folders = count_empty_folders(main_logo_dir)
    print(f"Empty folders : {empty_folders}/{total_folders}")

    rata_succes = (total_folders - empty_folders) * 100 / total_folders
    print(f"Success rate: {rata_succes:.2f}%")


    if rata_succes < 97:
        print(f"Success rate under 97%!")

#try https:// si https://www.