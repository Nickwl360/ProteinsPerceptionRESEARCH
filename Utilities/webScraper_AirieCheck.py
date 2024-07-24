import requests
from lxml import html
import schedule
import time
import smtplib,ssl
from email.mime.text import MIMEText
import logging
import os
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from html import unescape

# Create a directory for logs if it doesn't exist
log_directory = 'OutputLogsAirie'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging to write to a file in the specified directory
log_file_path = os.path.join(log_directory, 'apartment_monitor.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Function to check the website
def check_website():
    url = 'https://www.airiedenver.com/floor-plans'

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure the browser runs in headless mode
    chrome_service = Service('C:/Users/Nickl/OneDrive/Desktop/chromedriver/chromedriver-win64/chromedriver.exe')  # Replace with the path to your ChromeDriver
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    driver.get(url)
    time.sleep(5)  # Wait for JavaScript to load the content

    page_source = driver.page_source
    #print(page_source)
    driver.quit()

    tree = html.fromstring(page_source)
    apartments = tree.xpath('//div[@id="all_available_tab"]//div[contains(@class, "pro-sec rracFloorplan available")]')
    print(apartments)
    available_apartments = []
    # Define the date range
    start_date = datetime.strptime('2024-07-25', '%Y-%m-%d')
    end_date = datetime.strptime('2024-12-14', '%Y-%m-%d')
    for apartment in apartments:
        room_type= apartment.xpath('.//@data-room-type')[0]
        dates_str = apartment.xpath('.//@data-available-date')[0]
        #fixes non-html syntax
        dates_str = unescape(dates_str)
        dates_str = dates_str.strip('[]"')
        dates = [datetime.strptime(date.strip('"&quot;"'), '%Y-%m-%d') for date in dates_str.split(',')]

        for date in dates:
            if start_date<= date < end_date and (room_type == '4962' or room_type == '4961'):
                available_apartments.append((date.strftime('%Y-%m-%d'),room_type))


    # Check if new apartments are available
    if available_apartments:
        send_email(available_apartments)
        logging.info(f'New apartments found: {available_apartments}')
    else:
        logging.info('No new apartments found within the requested date')


# Function to send email
def send_email(apartments):

    smtpserver = "smtp.gmail.com"
    port = 587
    sender = 'Nickwl360@gmail.com'
    receiver = 'Nickwl360@gmail.com'
    password = 'KoolKar5115'
    message = f"""\
        Subject: NEW APARTMENTS????
        To: {receiver}
        From: {sender}

        New apartments within your requested dates."""

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtpserver, port, context=context) as server:
        server.starttls(context=context)

        server.login("Nickwl360@gmail.com", password)
        server.sendmail(sender, receiver, message)






# Schedule the script to run every hour
#schedule.every().hour.do(check_website)
check_website()
# Main loop
# while True:
#     schedule.run_pending()
#     time.sleep(1)
