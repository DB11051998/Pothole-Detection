from xml.etree import ElementTree

def extract_boxes(filename):
    """
        description:- this fuction extracts bounding box from the xml annoted file
        input:- filename-> path of the file
        returns:- boxes-> list of dictionary containing the co-ordinates of the boxes
        
    """
    
    # load and parse the file
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    boxes = list()
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        #coors = [xmin, ymin, xmax, ymax]
        #gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
        boxes.append({"x1":xmin,"x2":xmax,"y1":ymin,"y2":ymax})
    # extract image dimensions
    #width = int(root.find('.//size/width').text)
    #height = int(root.find('.//size/height').text)
    return boxes

def get_iou(bb1, bb2):
    """
    Description:- this function computes the intersection of union
    inputs:- bb1-> coordinates containing box1
             bb2-> coordinates containing box2
    returns:- Intersection of union score
    """
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def generate_labels_images(annot,images):
    """
    Description:- generates bounding box images and their labels
    Inputs:- annot-> path of annotation files
             images-> path of images file
    returns:- numpy array of train images and their labels
    
    """
    
    for e,i in enumerate(os.listdir(annot)):
    try:
        if i.startswith("img"):
            filename = i.split(".")[0]+".jpg"
            annotname= i.split(".")[0]+".xml"
            print(e,filename)
            image = cv2.imread(os.path.join(images,filename))
            box = extract_boxes(os.path.join(annot,annotname))
            #gtvalues=[]
            #for row in df.iterrows():
            #    x1 = int(row[1][0].split(" ")[0])
            #    y1 = int(row[1][0].split(" ")[1])
            #    x2 = int(row[1][0].split(" ")[2])
            #    y2 = int(row[1][0].split(" ")[3])
            #    gtvalues.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for e,result in enumerate(ssresults):
                if e < 2000 and flag == 0:
                    for gtval in box:
                        x,y,w,h = result
                        iou = get_iou(gtval,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                        if counter < 30:
                            if iou > 0.70:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(1)
                                counter += 1
                        else :
                            fflag =1
                        if falsecounter <30:
                            if iou < 0.3:
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                        else :
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1
    except Exception as e:
        print(e)
        print("error in "+filename)
        continue
    return train_images,train_labels
