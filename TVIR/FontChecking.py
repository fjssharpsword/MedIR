from PIL import ImageFont, Image, ImageDraw
from deep_translator import GoogleTranslator
import pandas as pd

def translater():
    #https://pillow.readthedocs.io/en/stable/reference/ImageFont.html
    #https://github.com/nidhaloff/deep-translator

    ttf_path = '/data/pycode/MedIR/TVIR/docs/simfang.ttf'
    font = ImageFont.truetype(font=ttf_path, size=50)

    cn_txt = '功能'
    translater = GoogleTranslator(source='auto', target='de')
    de_text = translater.translate(cn_txt) #german

    translater = GoogleTranslator(source='auto', target='en')
    en_text = translater.translate(cn_txt) #english

    translater = GoogleTranslator(source='auto', target='ko')
    ko_text = translater.translate(en_text) #korean

    translater = GoogleTranslator(source='auto', target='ja')
    ja_text = translater.translate(en_text) #japanese
    
    txt_str = cn_txt + '--' + en_text + '--' + de_text + '--' + ko_text + '--' + ja_text
    width, height = font.getsize(txt_str)
    canvas = Image.new('RGB', [width, height], (255,255,255))
    draw = ImageDraw.Draw(canvas)
    draw.text((0,0), txt_str, font=font, fill='#000000') #fill with white
    canvas.save("/data/pycode/MedIR/TVIR/imgs/cn_font.png")

def read_excel():
    xlsx_path = '/data/pycode/MedIR/TVIR/docs/trans_test.xlsx'
    df_xls = pd.read_excel(xlsx_path) 
    
    #txt_str = df_xls['Farsi(Persian)'][35]
    txt_str = df_xls['Bengali'][11]
    print(txt_str)

    ttf_path = '/data/pycode/MedIR/TVIR/docs/arial-unicode-ms.ttf'
    font = ImageFont.truetype(font=ttf_path, size=50)

    width, height = font.getsize(txt_str)
    canvas = Image.new('RGB', [width, height], (255,255,255))
    draw = ImageDraw.Draw(canvas)
    draw.text((0,0), txt_str, font=font, fill='#000000') #fill with white
    canvas.save("/data/pycode/MedIR/TVIR/imgs/bengali_font.png")

def grid_to_image():
    xlsx_path = '/data/pycode/MedIR/TVIR/docs/trans_test.xlsx'
    


def main():
    #translater()
    #read_excel()
    grid_to_image()

if __name__ == "__main__":
    main()