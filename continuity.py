from PIL import Image,ImageDraw
import random
import numpy as np
import math
import os
from orientation import choose_point

def resize_prop(im,max_l=800):
  """
  :param im: image to be reiszed
  :param max_l: max length of the image
  :return: resized image and ratio
  """
  w,h=im.size
  _max=max(w,h)
  ratio=max_l/_max
  #ratio=1 if ratio>1 else ratio
  out=im.resize((int(w*ratio),int(h*ratio)))
  return out,ratio


def cal_rotated_point(to_be_rotated,stand,angle):
  """
  :param to_be_rotated: point which need to be rotated
  :param stand: ratate according to this point
  :param angle: rotation angle
  :return: rotated point
  """
  angle = angle % 360.0
  a = -math.radians(angle)
  x,y=to_be_rotated
  xm,ym=stand
  #y = 2 * ym - y
  xr = (x - xm) * math.cos(a) - (y - ym) * math.sin(a) + xm
  yr = (x - xm) * math.sin(a) + (y - ym) * math.cos(a) + ym
  #yr = 2 * ym - yr
  return (xr,yr)

def intersection_angle(start,end1,end2):
  """
  all points have been already translated to same coordinate system.
  two vector has same start point
  :param start:start
  :param end1: initial end
  :param end2: random point end
  :return:  angle
  """
  l1 = (end1[0] - start[0], end1[1] - start[1])
  l2 = (end2[0] - start[0], end2[1] - start[1])
  dot = l1[0] * l2[0] + l1[1] * l2[1]
  tt1 = math.hypot(l1[0], l1[1])
  tt2 = math.hypot(l2[0], l2[1])
  radius = math.acos(dot / tt1 / tt2)
  angle=math.degrees(radius)
  return angle


def cal_box_ratio(box1,box2):
  """
  calculate overlap ratio of two bounding boxes
  :param box1:left,upper,right,bottom
  :param box2:left,upper,right,bottom
  :return:overlap ratio
  """
  x1,y1,x2,y2=box1
  area1=(x2-x1)*(y2-y1)
  _x1,_y1,_x2,_y2=box2
  area2=(_x2-_x1+1)*(_y2-_y1+1)

  xx1=max(x1,_x1)
  yy1=max(y1,_y1)
  xx2=min(x2,_x2)
  yy2=min(y2,_y2)

  w=max(0.0,xx2-xx1+1)
  h=max(0.0,yy2-yy1+1)
  inter=w*h

  #ovr=inter/(area1+area2-inter)
  ovr=inter/area2
  return ovr


def paste_item(angle,dist,top,bottom,temp_start,im):
  """
  rotate item to paste it in the image
  :param angle: rotation angle
  :param dist: dist between top and bottom,and is rotation axis length
  :param top: item rotation axis top
  :param bottom: item rotation axis bottom
  :param temp_start: start position where pasting item in result image
  :param im: item
  :return:rotated image ,narrow bounding box(result image UCS),rotation degree,and end(start),center in result image
  """
  im_w, im_h = im.size
  radian = math.radians(angle)
  y_gap = dist * math.cos(radian)
  x_gap = dist * math.sin(radian)
  end = (temp_start[0] + x_gap, temp_start[1] - y_gap)

  #########################
  start, end = end, temp_start

  translated_end = (start[0] - top[0] + bottom[0], start[1] - top[1] + bottom[1])
  ita = intersection_angle(start, translated_end, end)
  cross = (translated_end[0] - start[0]) * (end[1] - start[1]) - (end[0] - start[0]) * (translated_end[1] - start[1])
  if cross > 0:
    ita = -ita
  rotated_start = cal_rotated_point(top, (im_w // 2, im_h // 2), ita)
  initial_center = (im_w // 2, im_h // 2)
  translated_center = (start[0] - rotated_start[0] + initial_center[0], start[1] - rotated_start[1] + initial_center[1])
  rotated_im = im.rotate(ita, expand=True)
  rm_w, rm_h = rotated_im.size
  pasted_start = (int(translated_center[0] - rm_w // 2), int(translated_center[1] - rm_h // 2))


  narrow_box = rotated_im.getbbox()
  rotated_im = rotated_im.crop(narrow_box)
  box = [pasted_start[0] + narrow_box[0], pasted_start[1] + narrow_box[1], pasted_start[0] + narrow_box[2],
         pasted_start[1] + narrow_box[3]]
  return rotated_im,box,ita,start,translated_center


def check_continuity(res,rotated_im,box,boxes):
  """
    ensure continuity
  """
  x1,y1,x2,y2=box
  x_gap=x2-x1
  y_gap=y2-y1
  w,h=res.size

  #boxes.append([x1+w,y1,w-1,y2])
  if x1<0:
    res.paste(rotated_im, (x1+w,y1), rotated_im)
    if y1<0:
      res.paste(rotated_im, (x1 , y1+h), rotated_im)
      res.paste(rotated_im, (x1 + w, y1+h), rotated_im)
      return res, boxes
    if y2>h:
      res.paste(rotated_im, (x1, y1 - h), rotated_im)
      res.paste(rotated_im, (x1 + w, y1 - h), rotated_im)
      return res, boxes
  if y1<0:
    res.paste(rotated_im, (x1, y1 + h), rotated_im)
    if x2>w:
      res.paste(rotated_im, (x1-w, y1), rotated_im)
      res.paste(rotated_im, (x1-w, y1 + h), rotated_im)
      return res,boxes
  if x2>w:
    res.paste(rotated_im, (x1 - w, y1), rotated_im)
    if y2>h:
      res.paste(rotated_im, (x1 , y1-h), rotated_im)
      res.paste(rotated_im, (x1 - w, y1-h), rotated_im)
      return res,boxes
  if y2>h:
    res.paste(rotated_im, (x1, y1-h), rotated_im)
  return res,boxes



def add_layer_info(layers,final_ratio,angle,center,size,serial,oss_link,flip):
  layer_dict = {}
  layer_dict['rotation'] = 360-angle
  layer_dict['lightness'] = 0
  layer_dict['center'] = [final_ratio*center[0],final_ratio*center[1]]
  layer_dict['saturation'] = 0
  layer_dict['brightness'] = 0
  layer_dict['size'] = [final_ratio*size[0],final_ratio*size[1]]
  layer_dict['serial'] =  serial
  layer_dict['recolor'] = False
  layer_dict['contrast'] = 0
  layer_dict['alpha'] = 1
  layer_dict['hue'] = 0
  layer_dict['res_thumb'] = oss_link
  layer_dict['flip'] = flip
  layers.append(layer_dict)
  return layers


def preprocess_items( items,do_pic_w=4724,do_pic_h = 7087 ):
  """
   process each item for three scale and mirrored items
   firstly,pasting items in the (do_pic_w,do_pic_h) image to get points coordinates,and finally translate the
   points coordinates to the final image's coordinates according to proportion
  :param items: input items
  :param do_pic_w:w
  :param do_pic_h:h
  :return: items
  """

  # w,h,max_l is manually chosen values in proper proption(final images,items),do not change these values
  w = 4770
  h = 7550
  max_l = 800
  max_l = max_l * do_pic_h / h

  # small(1-percent) big(1+percent)
  percent = 0.2
  all_items_info = {}  # "path":{"single_item_info":single_item_info,"pasted_pos_num":n}
  for item in items:
    single_item_info = {"small": {}, "normal": {}, "big": {}}
    item_path = item["path"]
    oss_link = item["oss_link"]
    top = item["top"]
    bottom = item["bottom"]
    item_size = item["size"]
    serial = item["serial"]
    im = Image.open(item_path)

    pasted_pos_num = len(top)
    for key in single_item_info.keys():
      _max_l = max_l
      if key == "small":
        _max_l *= (1 - percent)
        _max_l = int(_max_l)
      elif key == "big":
        _max_l *= (1 + percent)
        _max_l = int(_max_l)
      im1, ratio = resize_prop(im, max_l=_max_l)
      _im_w, _im_h = im1.size
      _top = [(point[0] * ratio, point[1] * ratio) for point in top]
      _bottom = (bottom[0] * ratio, bottom[1] * ratio)
      mirror_top = [(_im_w - 1 - point[0], point[1]) for point in _top]
      mirror_bottom = (_im_w - 1 - _bottom[0], _bottom[1])
      dists = [math.hypot(point[0] - _bottom[0], point[1] - _bottom[1]) for point in _top]
      single_item_info[key] = {"image": im1, "top": _top, "bottom": _bottom,
                               "mirror_image": im1.transpose(Image.FLIP_LEFT_RIGHT),
                               "mirror_top": mirror_top, "mirror_bottom": mirror_bottom, "dists": dists,
                               "serial": serial, "oss_link": oss_link}

    all_items_info[item_path] = {"single_item_info": single_item_info, "pasted_pos_num": pasted_pos_num}

  return all_items_info,(do_pic_w,do_pic_h)




def choose_one_item(all_items_info):
  """
  choose one item for next pasting
  :param all_items_info:items info
  :return:...
  """
  choose_item = random.choice(list(all_items_info.keys()))
  item_info = all_items_info[choose_item]["single_item_info"]
  pasted_pos_num = all_items_info[choose_item]["pasted_pos_num"]

  choice = random.choice(range(pasted_pos_num))
  size_key = np.random.choice(list(item_info.keys()), p=[0.15, 0.7, 0.15])
  choose_item = item_info[size_key]
  im = choose_item["image"]
  top = choose_item["top"]
  bottom = choose_item["bottom"]
  dists = choose_item["dists"]
  serial = choose_item["serial"]
  oss_link = choose_item["oss_link"]

  mirror_im = choose_item["mirror_image"]
  mirror_top = choose_item["mirror_top"]
  mirror_bottom = choose_item["mirror_bottom"]

  flip = False
  # mirror flip
  if random.random() > 0.5:
    flip = True
    im, top, bottom = mirror_im, mirror_top, mirror_bottom
  return im,top,bottom,choice,dists,serial,oss_link,flip



def concat(items=None,pic_final_size=None):
  layers=[]
  all_items_info,(w, h) =preprocess_items(items)
  final_w,final_h=pic_final_size
  final_ratio = final_h / h
  res = Image.new('RGBA', (w, h), (0, 0, 0))
  draw = ImageDraw.Draw(res)

  paste_point=[]  # start point
  paste_point.append((w//2, h//2))
  # choose best direction in split_num spans(each span angle is div_angle)
  split_num=6
  div_angle=360//split_num
  split_prob=1
  max_area_ratio=1
  max_box_ratio=0.5   # manual 0.5

  dif_split_prob=0.5  # whether split in different point,if True ,max split point num is two
  # critical global info,record boxes info of items which have been drawn in final image
  boxes=[]
  use_gap=True
  continual_gap=int(0.025*w) if int(0.025*w)>=1 else 1
  # whether visualize boxes and points
  verbose=False
  # whether save the mid size image
  save_pic=True

  while len(paste_point)!=0:
    start = paste_point.pop(0)
    temp_start=start
    im,top,bottom,choice,dists,serial,oss_link,flip=choose_one_item(all_items_info)
    im_w, im_h = im.size

    area_ratio=[]
    box_ratio=[]
    paste_info=[]
    for i in range(split_num):
      lo=div_angle*i
      hi = div_angle * (i + 1)
      if i==split_num-1:
        hi=360
      angle = random.randint(lo, hi)

      # rotate item and paste it in the image
      rotated_im,box,ita,start,center=paste_item(angle, dists[choice], top[choice], bottom, temp_start,im)

      # record paste information and rotated ends
      rotated_start = cal_rotated_point(top[choice], (im_w // 2, im_h // 2), ita)
      rotated_ends = [cal_rotated_point(point, (im_w // 2, im_h // 2), ita) for point in top]
      ends = [(start[0] - rotated_start[0] + point[0], start[1] - rotated_start[1] + point[1]) for point in rotated_ends]
      paste_info.append([rotated_im,ends,box,ita,center])

      # calculate necessary info for overlap function
      temp = np.array(res.crop(box))

      ## pixel area overlap
      area_temp=np.sum(temp[:,:,3]!=0)
      overlap=np.logical_and(np.array(rotated_im)[:,:,3]!=0,temp[:,:,3]!=0)
      _area_ratio=np.sum(overlap)/area_temp if area_temp!=0 else 0
      area_ratio.append(_area_ratio)

      # box overlap
      single_box_ratio=[]
      for pre_box in boxes:
        single_box_ratio.append(cal_box_ratio(pre_box,box))
      if len(single_box_ratio)!=0:
        # this is not accurate,different boxex may have overlap,so I maybe sum up the same area several times
        box_ratio.append(np.sum(single_box_ratio))

    min_box_ratio=0
    ix=-1
    if len(box_ratio)!=0:
      ix=np.argmin(box_ratio)
      min_box_ratio=box_ratio[ix]
      min_area_ratio = area_ratio[ix]

    if min_box_ratio<=max_box_ratio:
      if ix==-1:
        ix=np.argmin(area_ratio)
        min_area_ratio=area_ratio[ix]
      if min_area_ratio<max_area_ratio:
        rotated_im, ends, box,ita,center = paste_info[ix]

        if use_gap:
          if box[2]+continual_gap>w or box[3]+continual_gap>h:
            continue
        # update boxes
        boxes.append(box)
        if save_pic:
          res.paste(rotated_im,(box[0],box[1]), rotated_im)
        res,boxes=check_continuity(res,rotated_im,box,boxes)
        """this is added for sensen's invoking"""
        # record layer info for final pic
        layers=add_layer_info(layers,final_ratio,ita%360,center,(im_w,im_h),serial,oss_link,flip)

        if verbose:
          x1,y1,x2,y2=box
          _h=box[3]-box[1]
          draw.line([x1,y1,x1,y1+_h],(255,97,0))
          draw.line([x1, y1, x2, y2-_h], (255,97,0))
          draw.line([x2, y2-_h, x2, y2 ], (255,97,0))
          draw.line([x1, y1+_h, x2, y2], (255,97,0))
          draw.ellipse([temp_start[0] - 10, temp_start[1] - 10, temp_start[0] + 10, temp_start[1] + 10], fill=(0, 0, 255))
          draw.ellipse([ends[choice][0] - 10, ends[choice][1] - 10, ends[choice][0] + 10, ends[choice][1] + 10], fill=(255, 255, 0))

        for end in ends:
          if end[0]>=0 and end[0]<w and end[1]>=0 and end[1]<h:
            paste_point.append(end)
  if save_pic:
    save_path=os.path.join("temp.png")
    res.save(save_path)
  return layers

def get_items_info(num,pic_final_size,thumbs):
  """
  :param num:num of thumb_pics
  :param pic_final_size: size of final big image
  :param thumbs: thumb image dict ('serial':[oss_link,local_link,size])
  :return: layer info
  """
  items=[]
  for key in thumbs.keys():
    thumb=thumbs[key]
    oss_link,local_link,item_size=thumb
    top = choose_point(local_link, 3)
    img = Image.open(local_link)
    img=np.array(img)
    ny, nx = np.where(img[:, :, 3] != 0)
    bottom = (nx[-1], ny[-1])
    items.append({"serial":key,"size":item_size,"path":local_link,"oss_link":oss_link,"top":top,"bottom":bottom})
  layers=concat(items=items,pic_final_size=pic_final_size)
  return layers


"""manually choose point test"""
def manual_get_items_info(num,pic_final_size,thumbs):
  """
  :param num:num of thumb_pics
  :param pic_final_size: size of final big image
  :param thumbs: thumb image dict ('serial':[oss_link,local_link,size])
  :return: layer info
  """
  manual_items = [{"path": "1.png", "top": [(882, 186), (417, 720), (438, 1410)], "bottom": (588, 1944)}
    , {"path": "2.png", "top": [(434, 874), (1688, 3019), (1501, 1881)], "bottom": (1941, 3976)}
    , {"path": "4.png", "top": [(1276, 1611), (874, 1303)], "bottom": (814, 3366)}
    , {"path": "5.png", "top": [(576, 1092), (906, 441)], "bottom": (984, 1878)}
    , {"path": "6.png", "top": [(314, 382), (343, 208)], "bottom": (60, 743)}]
  items=[]
  for key in thumbs.keys():
    thumb=thumbs[key]
    oss_link,local_link,item_size=thumb
    top = choose_point(local_link, 3)
    img = Image.open(local_link)
    img=np.array(img)
    ny, nx = np.where(img[:, :, 3] != 0)
    bottom = (nx[-1], ny[-1])
    items.append({"serial":key,"size":item_size,"path":local_link,"oss_link":oss_link,"top":top,"bottom":bottom})
  items[0]['path']='1.png'
  items[0]['top']=manual_items[0]["top"]
  items[0]['bottom'] = manual_items[0]["bottom"]

  items[1]['path'] = '2.png'
  items[1]['top'] = manual_items[1]["top"]
  items[1]['bottom'] = manual_items[1]["bottom"]

  layers=concat(items=items,pic_final_size=pic_final_size)
  return layers


def main():
  layers=get_items_info(2, (4724, 7087), {
    '1': ['http://lire.oss-cn-hangzhou.aliyuncs.com/tmp/046a5dd8-4e97-40b3-970d-fa05ef3bb01b.png','1.png', (779, 1153)],
    '2': ['http://lire.oss-cn-hangzhou.aliyuncs.com/tmp/b465671d-8449-4fd5-a830-896fc01eec59.png', '2.png', (502, 816)],
    })

if __name__=='__main__':
 main()
