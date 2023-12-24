from opts import parse_opts_offline
import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split
import yaml
import numpy as np
from mmengine.fileio import dump, load
# extract data from xml file

def extract_data_from_xml(opts):
    annot_folder = os.path.join(opts.root_path, opts.annot_files)
    img_paths = [] 
    img_sizes = [] 
    img_labels = [] 
    classes = []
    class_to_idx = {}
    
    annot_files = os.listdir(annot_folder)
    for file in annot_files:
        xml_path = os.path.join(annot_folder,file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root[1].text
        width,height = int(root[2][0].text),int(root[2][1].text)
        labels = []
        for obj in root.findall("object"):
            class_name = obj[0].text
            if class_name not in classes:
                classes.append(class_name)
                class_to_idx[class_name] = len(classes) - 1
            class_idx = class_to_idx[class_name]
            bb = obj[-1]
            x_min,y_min,x_max,y_max = int(bb[0].text),int(bb[1].text),int(bb[2].text),int(bb[3].text)
            
            labels.append([class_idx,x_min,y_min,x_max,y_max])
        img_paths.append(img_name)
        img_sizes.append((width,height))
        img_labels.append(labels)
    return img_paths,img_sizes,img_labels,classes,class_to_idx

                
        
    
   
def convert_to_coco_format(data,classes,class_to_idx,url):
    annotations = []
    images = []
    obj_count = 0
    for idx,[image_path,labels,image_size] in enumerate(data):
        image_width , image_height = image_size
        images.append(
            dict(id=idx, file_name=image_path, height=image_height, width=image_width))
        for bbox in labels:
            class_idx,x_min,y_min,x_max,y_max = bbox
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=class_idx,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),    
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[
            {"id":class_to_idx[v],'name':v} for v in classes
        ])
    dump(coco_format_json, url)

def split_train_test_val(data,seed = 0,val_size = 0.1,test_size = 0.1,is_shuffle = True):
    data = [[img,l,s]for img,l,s in zip(*data)]
    train_data , val_data = train_test_split(
        data , test_size=val_size , random_state=seed, shuffle=is_shuffle
        )
    train_data , test_data = train_test_split(
        train_data , test_size=test_size , random_state=seed, shuffle=is_shuffle
        )
    print("train: ",len(train_data))
    print("val: ",len(val_data))
    print("test: ",len(test_data))
    return train_data,test_data,val_data


def save_data(data,opts,data_type,classes,class_to_idx):
    save_dir = opts.coco_data_dir
    # create folder if it not exits
    os.makedirs(save_dir,exist_ok=True)
    
    # make images and labels folder
    os.makedirs(os.path.join(save_dir,f"images/{data_type}"),exist_ok=True)
    os.makedirs(os.path.join(save_dir,"annotations"),exist_ok=True)
    
    for image_path,_,_ in data:
        print(os.path.join(opts.root_path,"images",image_path))
        # copy image
        shutil.copy(
            os.path.join(opts.root_path,"images",image_path),
            os.path.join(save_dir,f'images/{data_type}/{image_path}'),
        )
        
    convert_to_coco_format(data,classes,class_to_idx, os.path.join(save_dir,f'annotations/{data_type}.json'))
    

if __name__ == '__main__': 
    opts = parse_opts_offline()
    img_paths,img_sizes,img_labels,classes,class_to_idx = extract_data_from_xml(opts)
    train_data,test_data,val_data = split_train_test_val([img_paths,img_labels,img_sizes])
    # save data according to yolov8 format
    save_data(train_data,opts,"train",classes,class_to_idx)
    save_data(test_data,opts,"test",classes,class_to_idx)
    save_data(val_data,opts,"val",classes,class_to_idx)
    
    # create_yaml_file(opts,class_labels=classes,nc = len(classes))
   