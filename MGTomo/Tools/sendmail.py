import smtplib, ssl, json
from os.path import expanduser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

std_config = expanduser("~") + '/.ssh/emailpw.json'

def sendmail(config: str = std_config, subject = '', msg = ''):
  with open(config, 'r') as f:
    data = json.loads(f.read())

  port = 465  # For SSL
  smtp_server = data['smtp']
  sender_email = data['sender']  # Enter your address
  receiver_email = data['receiver']  # Enter receiver address

  # Create a multipart message and set headers
  message = MIMEMultipart()
  message["From"] = sender_email
  message["To"] = receiver_email
  message["Subject"] = subject

  message.attach(MIMEText(msg, "plain"))
  message = message.as_string()

  context = ssl.create_default_context()
  with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(data['user'], data['password'])
    server.sendmail(sender_email, receiver_email, message)
