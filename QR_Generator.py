import pandas as pd
import os
import qrcode

file = 'QR_Asset_Management_Dataset.xlsx'

assets = pd.read_excel(file , sheet_name = "Assets")
os.makedirs("qr_code" , exist_ok =  True)

for asset in assets['Asset_ID']:
    
    qr = qrcode.QRCode(version = 1 , box_size=10, border=4)
    qr.add_data(asset)
    qr.make(fit=True)
    image = qr.make_image(fill_color = "Black" , back_colour = "White")
    
    Asset_ID = assets['Asset_ID']
    image.save(f"qr_code/{asset}.png")
print('All Images Made')