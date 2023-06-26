import json
from pycocotools import coco
import numpy as np
import pickle as pkl

def bboxiou(pred_box, ann_box, ann_offset, pred_offset=None):
    ann_box[:, 2:] += ann_box[:, :2]
    matched_id = []
    offset = []
    for i, bbox in enumerate(pred_box):
        x1, y1, x2, y2, score = bbox
        x1min = np.maximum(ann_box[:,0], x1)
        y1min = np.maximum(ann_box[:,1], y1)
        x2max = np.minimum(ann_box[:,2], x2)
        y2max = np.minimum(ann_box[:,3], y2)
        
        w = np.maximum(0, x2max - x1min)
        h = np.maximum(0, y2max - y1min)
        inter = w * h
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (ann_box[:, 2] - ann_box[:, 0]) * (ann_box[:, 3] - ann_box[:, 1])
        iou = inter / (area1 + area2 - inter)
        if max(iou)>0:
            idx = np.argmax(iou)
            matched_id.append(idx)
            offset.append(ann_offset[idx])
        else:
            matched_id.append(-1)
            if pred_offset is not None:
                offset.append(pred_offset[i])
            else:
                offset.append([0, 0])
    return matched_id, offset 
        
    
def compare_json_pkl(ann_list:list,
                    pred_list:list,
                    using_bbox_key='building_bbox',):
    def annwise2imagewise(anns):
        imagewise=[]
        bbox =[]
        offset = []
        now_image = None
        for ann in anns:
            if ann['image_id']!=now_image:
                if now_image is not None:
                    imagewise.append({'image_id':now_image,
                                      using_bbox_key:bbox,
                                      'offset':offset})
                now_image = ann['image_id']
                bbox = []
                offset = []
            bbox.append(ann[using_bbox_key])
            offset.append(ann['offset'])
        return imagewise
    imagewise_ann = annwise2imagewise(ann_list)
    distances = []
    for ann, pred in zip(imagewise_ann, pred_list):
        ann_bbox = np.array(ann[using_bbox_key])
        pred_bbox = np.array(pred[0][0])
        ann_offset = np.array(ann["offset"])
        pred_offset = np.array(pred[2])
        
        matched_id, matched_offset = bboxiou(pred_bbox, ann_bbox, ann_offset, )
        distance = np.sqrt(np.sum((matched_offset - pred_offset)**2, 1))
        distances.extend(distance.tolist())
    print('average offset distance: {}'.format(np.mean(distances)))
    return distances 

def compare_ordered_json(ann_list:list,
                         pred_list:list,
                         check=True):
    mean_dis = []
    mean_len = []
    mean_ang = []
    for ann, pred in zip(ann_list, pred_list):
        ann_offset = np.array(ann["offset"])
        pred_offset = np.array(pred['offset'][:2])
        distance = np.sqrt(np.sum((ann_offset - pred_offset)**2))
        ann_len = np.sqrt(np.sum(ann_offset**2))
        pred_len = np.sqrt(np.sum(pred_offset**2))
        a = sum(ann_offset*pred_offset)/(np.linalg.norm(ann_offset) * np.linalg.norm(pred_offset))
        # if pred_len<120:
        #     continue
        if check:
            if sum(ann['building_bbox'])!=sum(pred['building_bbox']):
                # print(pred["file_name"])
                pass
            else:
                # print('yes')
                if not np.isnan(a):
                    mean_ang.append(np.arccos(a))
                    # mean_dis.append(distance)
                    # mean_len.append(abs(ann_len-pred_len))
                else:
                    mean_ang.append(0)
                    # mean_dis.append(0)
                    # mean_len.append(0)
                mean_dis.append(distance)
                mean_len.append(abs(ann_len-pred_len))
                # mean_len.append(ann_len-pred_len)
        
    return mean_dis, mean_len, mean_ang

# if __name__=='__main__':
#     pred = pkl.load(open('baseline_test_0.5.pkl','rb'))
#     ann = json.load(open('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json'))
#     compare_json_pkl(ann['annotations'], pred)

if __name__=='__main__':
    pred = json.load(open('off_bbox_mask_guassian.json'))
    ann = json.load(open('/config_data/BONAI_data/BONAI-20230403T091731Z-002/BONAI/coco/bonai_shanghai_xian_test.json'))
    
    # offset = []
    # for a in ann['annotations']:
    #     offset.append(a['offset'])
    # offset = np.array(offset)
    # lenth = np.sqrt(np.sum(offset**2, 1))
    # print(np.mean(lenth, 0))
    
    mean_dis, mean_len, mean_ang = compare_ordered_json(ann['annotations'],pred)
    print('mean: {}, {}, {}'.format(np.mean(mean_dis), np.mean(mean_len), np.mean(mean_ang)))
    pass
    