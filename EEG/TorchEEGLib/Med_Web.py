import requests
import json
from PIL import Image
from io import BytesIO
import base64
import numpy as np

# 要传入的数据
files = {'image': open('/data/tmpexec/tb_log/07_test.tif', 'rb')}

# 要监听的地址
# url = 'http://127.0.0.1:6006/upload'  # 假设 Flask 应用监听在本地的 5000 端口上
url = 'http://region-42.seetacloud.com:12571/upload'

response = requests.post(url, files=files)

if response.status_code == 200:
    processed_data = json.loads(response.content)
    bytes_steam = processed_data['image_str'].encode()
    bytes_img = base64.b64decode(bytes_steam)
    img = BytesIO(bytes_img)
    processed_img = Image.open(img)
    processed_img.save("/data/tmpexec/tb_log/07_test_ai.jpg")
    #img_array = np.asarray(processed_img)
    #processed_img.show()
    #print(img_array.shape)
else:
    print('请求失败：', response.status_code)

