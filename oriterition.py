from __future__ import print_function,division
from PIL import Image,ImageDraw
import random
import math

import numpy as np

def get_segment_list(segment,min_gap=5):
    seg_list = []
    _len = len(segment)
    i=0
    start = segment[i]
    end = segment[i]+1
    count = 1
    while i<_len-1:
        if segment[i+1] - segment[i] == 1:
            end = segment[i+1]
        else:
            end = segment[i]
            if end - start > min_gap:
                seg_list.append((count,end-start,start,end))
            start = segment[i+1]
            count += 1
        i+=1
    if end - start > min_gap:
        seg_list.append((count,end - start,start, end))
    return seg_list
def draw_ellipse(draw,point,r,fill=(255,0,0)):
    draw.ellipse((point[0]-r,point[1]-r,point[0]+r,point[1]+r),fill=fill)

def draw_bbox(draw,point, new_point,color = (255,0,255)):
    s_x,s_y = point
    e_x,e_y = new_point
    draw.line((s_x,s_y,s_x,e_y), fill=color, width=3)
    draw.line((s_x, s_y, e_x, s_y), fill=color, width=3)
    draw.line((s_x, e_y, e_x, e_y), fill=color, width=3)
    draw.line((e_x, s_y, e_x, e_y), fill=color, width=3)

def get_angle_list(divide_nums):
    assert 360 % divide_nums == 0, 'divide_num must be divided by 360'
    stage = 360 // divide_nums
    angle_list = []
    for i in range(divide_nums):
        r = random.randint(stage*i,stage*(i+1))
        angle_list.append(r)
    return angle_list

def get_bound_boxes(img_draw,point,angle_list,line_length,img_size):

    _w,_h = img_size
    w,h = point
    bbox_list = []
    for angle in angle_list:
        new_w = w + line_length * math.sin(angle)
        new_h = h + line_length * math.cos(angle)

        if new_w < 0:
            new_w = 0
        elif new_w > _w:
            new_w = _w

        if new_h < 0:
            new_h = 0
        elif new_h > _h:
            new_h = _h


        # if DRAW:
            # img_draw.line([point[0],point[1],new_w,new_h],fill=(0,0,0),width=4)
            # draw_bbox(img_draw,point,(new_w,new_h))

        if new_w < w:
            max_w = w
            min_w = new_w
        else:
            max_w = new_w
            min_w = w

        if new_h < h:
            max_h = h
            min_h = new_h
        else:
            max_h = new_h
            min_h = h

        bbox_list.append((min_w,min_h,max_w,max_h))
    return bbox_list

def calculate_overlap(img_draw,point,mask_np,divide_nums):
    w,h = point
    angle_list = get_angle_list(divide_nums)
    _h, _w = mask_np.shape
    line_length = _h * 0.1
    bbox_list = get_bound_boxes(img_draw,point,angle_list,line_length,(_w,_h))
    mask2 = Image.new(mode='L',size=(_w,_h))
    mask2_np = np.array(mask2)
    for bbox in bbox_list:
        min_w,min_h,max_w,max_h = bbox
        min_w, min_h, max_w, max_h = int(min_w),int(min_h),int(max_w),int(max_h)
        mask2_np[min_h:max_h,min_w:max_w] = 255

    pixel_sum = np.logical_and(mask_np,mask2_np).sum()

    return pixel_sum

def choose_point(img_name,line_num,DRAW = False):
    img = Image.open(img_name)
    img_draw = ImageDraw.Draw(img)
    w, h = img.size
    np_img = np.array(img)
    ny, nx = np.where(np_img[:, :, 3] != 0)
    mask = Image.new(mode='L',size=img.size)
    mask_np = np.array(mask)
    mask_np[np.where(np_img[:, :, 3] != 0)] = 255

    line_gap = h // line_num
    min_gap = 10
    point_list = []
    for i in range(line_num):
        s,e = i * line_gap,(i+1)*line_gap

        if s < h * 0.1:
            s = int(h * 0.1)
        if e > h * 0.9:
            e = int(h * 0.9)
        # stage_sample = int((e-s) * 0.01)
        stage_sample = 20
        line_start = random.sample(range(s,e),stage_sample)
        line_start = sorted(line_start, reverse=True)
        line_list = [((0, _h), (w, _h)) for _h in line_start]

        store_list = []
        for item in line_list:
            line_id = item[0][1]
            if DRAW:
                img_draw.line([item[0][0], line_id, item[1][0], line_id], (0, 0, 0), 3)
            line_pixel = np_img[line_id, :, :]
            segment = np.where(line_pixel[:, 3] != 0)
            seg_list = get_segment_list(segment[0], min_gap)

            for seg in seg_list:
                cnt, _len, start, end = seg
                start_lap = calculate_overlap(img_draw,(start,line_id),mask_np,divide_nums=18)
                end_lap = calculate_overlap(img_draw, (end, line_id), mask_np, divide_nums=18)

                store_list.append(((start,line_id),start_lap,-1))
                store_list.append(((end, line_id), end_lap,1))


                if DRAW:
                    draw_ellipse(img_draw,(start,line_id),5,fill=(255,0,0))
                    draw_ellipse(img_draw, (end, line_id), 5, fill=(0, 0, 255))
                # break
        store_list = sorted(store_list,key=lambda x:x[1])
        if store_list == []:
            continue
        _w,_h = store_list[0][0]
        if store_list[0][2] == -1:
            _w = _w + min_gap // 2
        elif store_list[0][2] == 1:
            _w = _w - min_gap // 2
        draw_ellipse(img_draw,(_w,_h),r = 10,fill=(20,20,20))
        point_list.append((_w,_h))
    #img.save('orention.png')
    return point_list
if __name__ == '__main__':
    img_name = '4.png'
    line_num = 3
    choose_point(img_name,line_num)
