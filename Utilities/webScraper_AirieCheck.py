import requests
from lxml import html
import schedule
import time
import smtplib
from email.mime.text import MIMEText
import logging
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from html import unescape
from twilio.rest import Client

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
    print('checking')
    url = 'https://www.airiedenver.com/floor-plans'

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Ensure the browser runs in headless mode

    #laptop
    #chrome_service = Service('C:/Users/Nickl/OneDrive/Desktop/chromedriver/chromedriver-win64/chromedriver.exe')  # Replace with the path to your ChromeDriver
    #HomePC
    chrome_service = Service('C:/Users/Nick/Desktop/chromedriver-win64/chromedriver-win64/chromedriver.exe')  # Replace with the path to your ChromeDriver

    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

    driver.get(url)
    time.sleep(5)  # Wait for JavaScript to load the content

    page_source = driver.page_source
    #print(page_source)
    driver.quit()

    tree = html.fromstring(page_source)
    apartments = tree.xpath('//div[@id="all_available_tab"]//div[contains(@class, "pro-sec rracFloorplan available")]')
    available_apartments = []
    # Define the date range
    start_date = datetime.strptime('2024-08-14', '%Y-%m-%d')
    end_date = datetime.strptime('2024-09-14', '%Y-%m-%d')
    for apartment in apartments:
        room_type= apartment.xpath('.//@data-room-type')[0]
        dates_str = apartment.xpath('.//@data-available-date')[0]
        type = apartment.xpath('.//@data-fpname')[0]
        price = apartment.xpath('.//@data-min-price')[0]

        #fixes non-html syntax
        dates_str = unescape(dates_str)
        dates_str = dates_str.strip('[]"')
        dates = [datetime.strptime(date.strip('"&quot;"'), '%Y-%m-%d') for date in dates_str.split(',')]

        for date in dates:
            if start_date<= date < end_date and (room_type == '4962' or room_type == '4961'):
                available_apartments.append((date.strftime('%Y-%m-%d'),type,price))


    # Check if new apartments are available
    if available_apartments:
        send_email(available_apartments)
        send_sms(available_apartments)
        logging.info(f'New apartments found: {available_apartments}')
    else:
        logging.info('No new apartments found within the requested date')
def send_email(apartments):
    password = 'wiiqviujxztyisxq'
    subject = 'APARTMENT ALERT: GO APPLY NOW'
    body = 'Possible move in date, type, price:\n'+''.join([str(apartment) for apartment in apartments])
    sender = 'Nickwl360@gmail.com'
    reciever = 'ejonwl22@gmail.com'
    smtpserver = "smtp.gmail.com"
    msg = MIMEText(body)
    msg['Subject']=subject
    msg['From']=sender
    msg['To']=reciever

    with smtplib.SMTP_SSL(smtpserver, 465) as server:
        # server.starttls()
        server.login("Nickwl360@gmail.com", password)
        server.send_message(msg)
        logging.info('success')
        server.quit()
def send_sms(apartments):
    account_sid = 'AC5005f86fb6682875c396ea9d651f3404'
    auth_token = '18902be4efed073979d569c636971e24'
    twilio_number = '+18557255216'
    myphone_number = '+13178641468'

    client = Client(account_sid, auth_token)
    body = 'NEW APT: Possible move in date, type, price:\n'+''.join([str(apartment) for apartment in apartments])
    try:
        message = client.messages.create(
            body=body,
            from_=twilio_number,
            to=myphone_number
        )
        logging.info('SMS sent successfully')
    except Exception as e:
        logging.error(f'Error sending SMS: {e}')
    return
check_website()
schedule.every().hour.do(check_website)
while True:
    schedule.run_pending()
    time.sleep(1)
