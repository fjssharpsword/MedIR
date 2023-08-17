import paddle
import os
import xml.etree.ElementTree as ET

def main():

    train_file = open('/data/tmpexec/LogoDet/PaddleDetection/dataset/qmul_openlogo/train.txt', 'w')
    test_file = open('/data/tmpexec/LogoDet/PaddleDetection/dataset/qmul_openlogo/test.txt', 'w')
    label_file = open('/data/tmpexec/LogoDet/PaddleDetection/dataset/qmul_openlogo/label_list.txt', 'w')
    img_dir = '/data/fjsdata/TVLogo/QMUL-OpenLogo/openlogo/JPEGImages/'
    ann_dir = '/data/fjsdata/TVLogo/QMUL-OpenLogo/openlogo/Annotations/'
   
    with open('/data/fjsdata/TVLogo/QMUL-OpenLogo/openlogo/ImageSets/Main/train_test/train_all.txt', "r") as f:
        for line in f.readlines():
            name = line.strip('\n')
            img_path = os.path.join(img_dir+name+'.jpg')
            ann_path = os.path.join(ann_dir+name+'.xml')
            if os.path.exists(img_path) and os.path.exists(ann_path):
                train_file.write(img_path +' ' + ann_path + '\n')
    train_file.close()

    with open('/data/fjsdata/TVLogo/QMUL-OpenLogo/openlogo/ImageSets/Main/train_test/test_all.txt', "r") as f:
        for line in f.readlines():
            name = line.strip('\n')
            img_path = os.path.join(img_dir+name+'.jpg')
            ann_path = os.path.join(ann_dir+name+'.xml')
            if os.path.exists(img_path) and os.path.exists(ann_path):
                test_file.write(img_path +' ' + ann_path + '\n')
    test_file.close()
   
    lbl_name_list = []
    for xml_name in os.listdir(ann_dir):
        xml_path = os.path.join(ann_dir, xml_name)
        xml_tree = ET.parse(xml_path)
        root = xml_tree.getroot()
        object_list = root.findall("object")
        for object in object_list:
            lbl_name = object.find("name").text
            if lbl_name not in lbl_name_list:
                lbl_name_list.append(lbl_name)
                label_file.write(lbl_name + '\n')
    label_file.close()

if __name__ == "__main__":
    #paddle.utils.run_check()
    #print(paddle.__version__)
    main()

    #nohup python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6 tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_qmullogo.yml --eval >> output/yolov3_mobilenet_v1_qmullogo/LogoDet_YoLov3.log 2>&1 &
    #python3 tools/eval.py -c configs/yolov3/yolov3_mobilenet_v1_qmullogo.yml -o weights=output/yolov3_mobilenet_v1_qmullogo/best_model.pdparams