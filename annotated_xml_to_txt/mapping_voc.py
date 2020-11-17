import lxml
from xml.etree import ElementTree
import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', help= 'path annotated xml files', default="Annotation/")
args = parser.parse_args()
path = args.p

gt_paths = glob.glob(path+'/*.xml')
for gt in gt_paths:
    filename = os.path.splitext(os.path.split(gt)[1])[0]
    tree = ElementTree.parse(gt)
    xmin_obj = tree.findall('.//xmin')
    xmin_list = [t.text for t in xmin_obj]
    xmax_obj = tree.findall('.//xmax')
    xmax_list = [t.text for t in xmax_obj]

    ymin_obj = tree.findall('.//ymin')
    ymin_list = [t.text for t in ymin_obj]
    ymax_obj = tree.findall('.//ymax')
    ymax_list = [t.text for t in ymax_obj]

    class_obj = tree.findall('.//name')
    class_list = [t.text for t in class_obj]

    s=''
    traingt = 'train_gt'
    if not os.path.exists(traingt):
        os.makedirs(traingt)
    for a,b,c,d,e in zip(xmin_list,ymin_list,xmax_list,ymax_list,class_list):
        s+='{}|{}|{}|{}|"{}"\n'.format(a,b,c,d,e)
        f = open('{}/{}.txt'.format(traingt,filename),'w')
        f.write(s)
