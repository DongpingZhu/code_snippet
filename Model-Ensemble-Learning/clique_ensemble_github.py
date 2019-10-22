import cv2
import numpy as np
import time
import os
import multiprocessing as mp

maxBoxes=1600
vis=False
csvformat = 1
subformat = 1
full_IDs = 1
iou_thresh = 0.5

#each csv should be formatted like this all images in csv should be sorted by ID in alphabetical order(used in way parser works)

# ImageID,LabelName,Score,YMin,XMin,YMax,XMax
# 8e7753cc8d100ce5,/m/083wq,0.02683,0.368,0.502,0.746,0.653
# 8e7753cc8d100ce5,/m/04yx4,0.02682,0.293,0.281,0.814,0.658
# 8e7753cc8d100ce5,/m/083wq,0.02682,0.632,0.134,0.670,0.207
# 8e7753cc8d100ce5,/m/0bjyj5,0.02681,0.000,0.000,0.621,0.547


subcompress_path = 'PATH TO SUBMISSION OUTPUT FILE FORMATTED ImageId,PredictionString'
subcsv_path = 'PATH TO SUBMISSION OUTPUT FILE FORMATTED ImageID,LabelName,Score,YMin,XMin,YMax,XMax'

models = ['PATH TO ONE OF MODELS',
          'PATH TO ONE OF MODELS',
          'PATH TO ONE OF MODELS',
          'PATH TO ONE OF MODELS',
          'PATH TO ONE OF MODELS',
          'PATH TO ONE OF MODELS',
          'PATH TO ONE OF MODELS']

def IOUvec(arr):
    x11, y11, x12, y12 = np.split(arr[:,[3,2,5,4]], 4, axis=1)
    x21, y21, x22, y22 = np.split(arr[:,[3,2,5,4]], 4, axis=1)

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))

    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea + 1e-6)
    iou = np.maximum(iou,0)
    iou = np.minimum(iou,1)

    return iou

def ensemble_boxes(boxes,label,type):
    box_arr = np.array(boxes)

    if type == 'max':
        score = np.max(box_arr[:,1])
    if type == '1-x':
        score = 1-np.prod(1-box_arr[:,1])

    box_arr[:, 1]=box_arr[:,1]/np.sum(box_arr[:,1])
    box_arr[:,2:6]=np.multiply(box_arr[:,2:6].T, box_arr[:, 1]).T
    final_bbox=np.sum(box_arr[:,2:6],axis=0)

    return [label, score, final_bbox.item(0), final_bbox.item(1), final_bbox.item(2), final_bbox.item(3)]

def get_cliques(box_list):

    iou_arr=IOUvec(np.array(box_list))
    edges=[]

    # box [idx, float(Score), float(YMin), float(XMin), float(YMax), float(XMax)]

    for i in range(len(box_list)):
        for j in range(i):
            if iou_arr[i,j]>iou_thresh:
                edges.append([i,j,iou_arr[i,j]])

    edges = sorted(edges, key=lambda edge: -edge[2])
    cliques=[v for v in range(len(box_list))]

    for edge in edges:
        clique0 = [i for i in range(len(box_list)) if cliques[i]==cliques[edge[0]]]
        clique1 = [i for i in range(len(box_list)) if cliques[i]==cliques[edge[1]]]
        colors0 = set(box_list[i][0] for i in clique0)
        colors1 = set(box_list[i][0] for i in clique1)

        if len(colors0.intersection(colors1)) == 0:
            for i in range(len(box_list)):
                if cliques[i]==cliques[edge[0]]:
                    cliques[i]=cliques[edge[1]]

    cliques_dict={}

    for clique in list(set(cliques)):
        cliques_dict[clique]=[]

    for i in range(len(box_list)):
        cliques_dict[cliques[i]].append(box_list[i])

    return list(cliques_dict.values())

def visualise(img_path,boxes,output):
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    boxes = sorted(boxes, key=lambda box: box[1])
    boxes = boxes[-50:]

    for box in boxes:
        cv2.rectangle(img,(int(width*box[3]), int(height*box[2])), (int(width*box[5]), int(height*box[4])),(np.random.randint(0, 255),np.random.randint(0, 255),np.random.randint(0, 255)),int(5*box[1]))

    cv2.imwrite(output,img)

def process_bboxes(ImID,boxes_dict):
    result_boxes = []

    for key in boxes_dict.keys():
        cliques_result = get_cliques(boxes_dict[key])

        for clq in cliques_result:
            if len(clq) == 1:
                clq[0][0] = key
                result_boxes.append(clq[0])
            else:
                result_boxes.append(ensemble_boxes(clq, key, type='1-x'))

    result_boxes = sorted(result_boxes, key=lambda box: -box[1])
    result_boxes = result_boxes[:maxBoxes]
    return [ImID,result_boxes]

ImIdList=[]

if full_IDs == 1:
    with open('PATH TO SAMPLE SUBMISSION FROM KAGGLE USED TO GET IDs OF IMAGES', 'r') as samsub:
        for i, line in enumerate(samsub):
            if i!=0:
                ImIdList.append(line.split(',')[0])
else:
    for idx, img_name in enumerate(sorted(os.listdir('PATH TO FOLDER WITH IMAGES USED TO GET IMAGES IDs LIST'))):
        ImIdList.append(img_name.split('.')[0])

lines_read_dict={}

for id in ImIdList:
    lines_read_dict[id]=[-1 for i in range(len(models))]

with open(models[0], 'r') as f0, \
    open(models[1], 'r') as f1, \
    open(models[2], 'r') as f2, \
    open(models[3], 'r') as f3, \
    open(models[4], 'r') as f4, \
    open(models[5], 'r') as f5, \
    open(models[6], 'r') as f6, \
    open(subcompress_path, 'w') as subcompress,\
    open(subcsv_path, 'w') as subcsv:

    files = [f0,f1,f2,f3,f4,f5,f6]

    for f in files:
        line=f.readline()
    if subformat:
        subcompress.write('ImageId,PredictionString\n')
    if csvformat:
        subcsv.write('ImageID,LabelName,Score,YMin,XMin,YMax,XMax\n')

    boxes_dict_for_id={}

    for idx, ImID in enumerate(sorted(ImIdList)):
        boxes_dict_for_id[ImID] = {}

    packslist=[[]]
    for idx, ImID in enumerate(sorted(ImIdList)):
        if len(packslist[-1])==50:
            packslist.append([])

        packslist[-1].append(ImID)

    glob_idx=0
    beg = time.time()

    for pack in packslist:
        for ImID in pack:
            lines_read_list = [0 for i in range(len(models))]

            for idxf, f in enumerate(files):
                while (True):
                    line = f.readline()
                    if line == '':
                        break
                    ImageID, LabelName, Score, YMin, XMin, YMax, XMax = line.split(',')
                    lines_read_list[idxf]+=1

                    XMin = max(min(1.0, float(XMin)), 0.0)
                    YMin = max(min(1.0, float(YMin)), 0.0)
                    XMax = max(min(1.0, float(XMax)), 0.0)
                    YMax = max(min(1.0, float(YMax)), 0.0)
                    Score = max(min(1.0, float(Score)), 0.0)

                    if XMin < XMax and YMin < YMax:

                        if not LabelName in boxes_dict_for_id[ImageID].keys():
                            boxes_dict_for_id[ImageID][LabelName]=[]

                        boxes_dict_for_id[ImageID][LabelName].append([idxf, Score, YMin, XMin, YMax, XMax])

                    if ImageID != ImID:
                        break

            lines_read_dict[ImID]=lines_read_list
            print(lines_read_list)

        results_boxes_float = {}
        output_queue = mp.Queue()

        batchesnum = 3
        batches_list = [[] for b in range(batchesnum)]

        for k,ImID in enumerate(pack):
            batches_list[k%batchesnum].append([ImID,boxes_dict_for_id[ImID]])

        def process_batch(batch,output):
            for batch_item in batch:
                output.put(process_bboxes(batch_item[0],batch_item[1]))

        processes = [mp.Process(target=process_batch, args=[batch,output_queue]) for batch in batches_list]

        for p in processes:
            p.daemon = True
            p.start()

        while True:
            running = any(p.is_alive() for p in processes)
            while not output_queue.empty():
                tmp=output_queue.get()
                results_boxes_float[tmp[0]] = tmp[1]
            if not running:
                break

        for p in processes:
            p.join()

        for ImID in pack:
            if vis:
                visualise('PATH TO IMAGES WICH ARE USED TO VISUALIZE'+ImID+'.jpg', results_boxes_float[ImID], 'PATH TO OUTPUT OF VISUALISED IMAGES/{}.jpg'.format(glob_idx))

            results_boxes_str=[]

            if subformat:
                for box in results_boxes_float[ImID]:
                    if box[1]>0.02:
                        results_boxes_str.append('{} {} {} {} {} {}'.format(box[0],
                                                                  '{:.3f}'.format(box[1]*0.999+0.001).lstrip('0'),
                                                                  '{:.3f}'.format(box[3]).lstrip('0'),
                                                                  '{:.3f}'.format(box[2]).lstrip('0'),
                                                                  '{:.3f}'.format(box[5]).lstrip('0'),
                                                                  '{:.3f}'.format(box[4]).lstrip('0')))
                    else:
                        results_boxes_str.append('{} {} {} {} {} {}'.format(box[0],
                                                                            '{:.3f}'.format(box[1] * 0.999 + 0.001).lstrip('0'),
                                                                            '{:.3f}'.format(box[3]).lstrip('0'),
                                                                            '{:.3f}'.format(box[2]).lstrip('0'),
                                                                            '{:.3f}'.format(box[5]).lstrip('0'),
                                                                            '{:.3f}'.format(box[4]).lstrip('0')))

                subcompress.write(ImID + ',' + str(' '.join(results_boxes_str)) + '\n')
            if csvformat:
                for box in results_boxes_float[ImID]:
                    if box[1]>0.02:
                        subcsv.write('{},{},{},{},{},{},{}\n'.format(ImID,box[0],
                                                                  '{:.3f}'.format(box[1]*0.999+0.001),
                                                                  '{:.3f}'.format(box[2]),
                                                                  '{:.3f}'.format(box[3]),
                                                                  '{:.3f}'.format(box[4]),
                                                                  '{:.3f}'.format(box[5])))
                    else:
                        subcsv.write('{},{},{},{},{},{},{}\n'.format(ImID, box[0],
                                                                  '{:.3f}'.format(box[1] * 0.999 + 0.001),
                                                                  '{:.3f}'.format(box[2]),
                                                                  '{:.3f}'.format(box[3]),
                                                                  '{:.3f}'.format(box[4]),
                                                                  '{:.3f}'.format(box[5])))

            del boxes_dict_for_id[ImID]
            del results_boxes_float[ImID]
            del results_boxes_str
            glob_idx+=1

        print('Time left: {} , idx: {}'.format((time.time() - beg) / 60 / 60 / (glob_idx + 1) * (len(ImIdList) - glob_idx), glob_idx))

with open('PATH TO OUTPUT OF SUMMARY OF ENSEMBLING IF NOT USED DELETE THIS PART','w') as summary:
    summary.write('ID,0,1,2,3,4,5,6\n')

    for ImID in ImIdList:
        summary.write(ImID+','+','.join(str(x) for x in lines_read_dict[ImID])+'\n')