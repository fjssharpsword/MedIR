import requests
import json

#https://www.apispace.com/23329/api/aigc/guidence/#scroll=100
#CSDN账号登录，Token: tom8fj2j8klrh2cao2n5urfp8ilzp51b

def Txt2Img():
    url = "https://23329.o.apispace.com/aigc/txt2img"

    payload = {"task":"txt2img.sd","params":{"model":"anime",\
                                            "text":"国风, 美漫, 可爱, 华丽, 中国女孩",\
                                            "w":512,\
                                            "h":512,\
                                            "guidance_scale":14,\
                                            "negative_prompt":"",\
                                            "sampler":"k_euler",\
                                            "seed":1072366942,\
                                            "num_steps":25,\
                                            "notify_url":""}}

    headers = {
        "X-APISpace-Token":"tom8fj2j8klrh2cao2n5urfp8ilzp51b",
        "Authorization-Type":"apikey",
        "Content-Type":"application/json"
    }

    response=requests.request("POST", url, data=json.dumps(payload), headers=headers)
    print(response.text)
    res_dic = json.loads(response.text)
    uid = res_dic['data']['uid']
 
    return uid

def QueryImg(uid):

    url = "https://23329.o.apispace.com/aigc/query-image"

    payload = {"uid":uid}

    headers = {
        "X-APISpace-Token":"tom8fj2j8klrh2cao2n5urfp8ilzp51b",
        "Authorization-Type":"apikey",
        "Content-Type":"application/json"
    }

    response=requests.request("POST", url, data=json.dumps(payload), headers=headers)
    print(response.text)
    res_dic = json.loads(response.text)
    cdn = res_dic['data']['cdn']

    return cdn

if __name__ == "__main__":
    uid = Txt2Img()
    cdn = QueryImg(uid=uid)
    print(cdn)