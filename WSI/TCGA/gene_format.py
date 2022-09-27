
#通过Python实现txt以空格分隔时候转换为csv文件
import csv
csvFile = open("/data/pycode/MedIR/WSI/data/gene_sample.csv",'w',newline='',encoding='utf-8')
writer = csv.writer(csvFile)
csvRow = []
 
f = open("/data/pycode/MedIR/WSI/data/HiSeqV2.txt",'r',encoding='GB2312')
for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)
 
f.close()
csvFile.close()