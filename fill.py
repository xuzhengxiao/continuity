from PIL import Image,ImageDraw
import random
import numpy as np
import math
import os
import uuid

def resize_prop(im,max_l=600):
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
  calculate overlap ratio of two bounding boxed
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
  :param temp_start: start position when item put in result image
  :param im: item
  :return:rotated image ,narrow bounding box(result image UCS),rotation radian,and end(start) in result image
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
  return rotated_im,box,ita,start


def check_continuity(res,rotated_im,box,boxes):
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



# def concat(item_path="6.png",max_l=520):
#   #random.seed(47)
#   w=3000
#   h=2000
#
#   res = Image.new('RGBA', (w, h),(0,0,0))
#   draw = ImageDraw.Draw(res)
#
#   im = Image.open(item_path)
#   im, ratio = resize_prop(im, max_l=max_l)
#   im_w, im_h = im.size
#
#   top = [(314,382),(343,208)]
#   bottom = (60, 743)
#   # top = [(434, 874), (1688, 3019), (1501, 1881)]
#   # bottom = (1941, 3976)
#   choice_num=len(top)
#
#   top = [(point[0] * ratio, point[1] * ratio) for point in top]
#   bottom = (bottom[0] * ratio, bottom[1] * ratio)
#   ##########################################
#   dists = [math.hypot(point[0] - bottom[0], point[1] - bottom[1]) for point in top]
#
#   paste_point=[]  #start
#   paste_point.append((w//2, h//2))
#   split_num=6
#   div_angle=360//split_num
#   split_prob=1
#   max_area_ratio=1
#   max_box_ratio=0.5
#
#   dif_split_prob=0.5  # whether split in different point,if True ,max split point num is two
#   # critical global info,record boxes info of items which have been drawn in final image
#   boxes=[]  #########################
#   use_gap=True
#   continual_gap=50
#   # whether visualize boxes and points
#   verbose=True
#
#   while len(paste_point)!=0:
#     start = paste_point.pop(0)
#     temp_start=start
#     choices = [random.choice(range(choice_num))]
#     for choice in choices:
#       area_ratio=[]
#       box_ratio=[]
#       paste_info=[]
#       for i in range(split_num):
#         lo=div_angle*i
#         hi = div_angle * (i + 1)
#         if i==split_num-1:
#           hi=360
#         angle = random.randint(lo, hi)
#
#         # rotate item and paste it in the image
#         rotated_im,box,ita,start=paste_item(angle, dists[choice], top[choice], bottom, temp_start,im)
#
#         # record paste information and rotated ends
#         rotated_start = cal_rotated_point(top[choice], (im_w // 2, im_h // 2), ita)
#         rotated_ends = [cal_rotated_point(point, (im_w // 2, im_h // 2), ita) for point in top]
#         ends = [(start[0] - rotated_start[0] + point[0], start[1] - rotated_start[1] + point[1]) for point in rotated_ends]
#         paste_info.append([rotated_im,ends,box])
#
#         # calculate necessary info for overlap function
#         temp = np.array(res.crop(box))
#
#         ## pixel area overlap
#         area_temp=np.sum(temp[:,:,3]!=0)
#         overlap=np.logical_and(np.array(rotated_im)[:,:,3]!=0,temp[:,:,3]!=0)
#         _area_ratio=np.sum(overlap)/area_temp if area_temp!=0 else 0
#         area_ratio.append(_area_ratio)
#
#         # box overlap
#         single_box_ratio=[]
#         for pre_box in boxes:
#           single_box_ratio.append(cal_box_ratio(pre_box,box))
#         if len(single_box_ratio)!=0:
#           # this is not accurate,different boxex may have overlap,so I maybe sum up the same area several times
#           box_ratio.append(np.sum(single_box_ratio))
#
#       min_box_ratio=0
#       ix=-1
#       if len(box_ratio)!=0:
#         ix=np.argmin(box_ratio)
#         min_box_ratio=box_ratio[ix]
#         min_area_ratio = area_ratio[ix]
#
#       if min_box_ratio<=max_box_ratio:
#         if ix==-1:
#           ix=np.argmin(area_ratio)
#           min_area_ratio=area_ratio[ix]
#         if min_area_ratio<max_area_ratio:
#           rotated_im, ends, box = paste_info[ix]
#
#           if use_gap:
#             if box[2]+continual_gap>w or box[3]+continual_gap>h:
#               continue
#           # update boxes
#           boxes.append(box)
#           res.paste(rotated_im,(box[0],box[1]), rotated_im)
#           res,boxes=check_continuity(res,rotated_im,box,boxes)
#
#           if verbose:
#             x1,y1,x2,y2=box
#             _h=box[3]-box[1]
#             draw.line([x1,y1,x1,y1+_h],(255,97,0))
#             draw.line([x1, y1, x2, y2-_h], (255,97,0))
#             draw.line([x2, y2-_h, x2, y2 ], (255,97,0))
#             draw.line([x1, y1+_h, x2, y2], (255,97,0))
#             draw.ellipse([temp_start[0] - 5, temp_start[1] - 5, temp_start[0] + 5, temp_start[1] + 5], fill=(0, 0, 255))
#             draw.ellipse([ends[choice][0] - 5, ends[choice][1] - 5, ends[choice][0] + 5, ends[choice][1] + 5], fill=(255, 255, 0))
#
#           for end in ends:
#             if end[0]>=0 and end[0]<w and end[1]>=0 and end[1]<h:
#               paste_point.append(end)
#
#   res.save("222.png")

#100
def concat(items=None,max_l=800,save_dir="00_show"):
  # random.seed(47)
  # np.random.seed(47)
  # w=500
  # h=750
  w=4770
  h=7550
  percent=0.2
  res = Image.new('RGBA', (w, h), (0, 0, 0))
  draw = ImageDraw.Draw(res)

  all_items_info={}  #"path":{"single_item_info":single_item_info,"pasted_pos_num":n}
  for item in items:
    single_item_info = {"small": {}, "normal": {}, "big": {}}
    item_path=item["path"]
    top=item["top"]
    bottom=item["bottom"]
    im = Image.open(item_path)

    pasted_pos_num = len(top)
    for key in single_item_info.keys():
      _max_l=max_l
      if key=="small":
        _max_l*=(1-percent)
        _max_l=int(_max_l)
      elif key=="big":
        _max_l *= (1 + percent)
        _max_l = int(_max_l)
      im1, ratio = resize_prop(im, max_l=_max_l)
      _im_w, _im_h = im1.size
      _top = [(point[0] * ratio, point[1] * ratio) for point in top]
      _bottom = (bottom[0] * ratio, bottom[1] * ratio)
      mirror_top=[(_im_w-1-point[0],point[1]) for point in _top]
      mirror_bottom=(_im_w-1-_bottom[0],_bottom[1])
      dists = [math.hypot(point[0] - _bottom[0], point[1] - _bottom[1]) for point in _top]
      single_item_info[key]={"image":im1,"top":_top,"bottom":_bottom,"mirror_image":im1.transpose(Image.FLIP_LEFT_RIGHT),
                      "mirror_top":mirror_top,"mirror_bottom":mirror_bottom,"dists":dists}

    all_items_info[item_path]={"single_item_info":single_item_info,"pasted_pos_num":pasted_pos_num}

  paste_point=[]  #start
  paste_point.append((w//2, h//2))
  split_num=6
  div_angle=360//split_num
  split_prob=1
  max_area_ratio=1
  max_box_ratio=0.5

  dif_split_prob=0.5  # whether split in different point,if True ,max split point num is two
  # critical global info,record boxes info of items which have been drawn in final image
  boxes=[]  #########################
  use_gap=True
  continual_gap=int(0.025*w)
  # whether visualize boxes and points
  verbose=True


  while len(paste_point)!=0:
    start = paste_point.pop(0)
    temp_start=start

    choose_item=random.choice(list(all_items_info.keys()))
    item_info=all_items_info[choose_item]["single_item_info"]
    pasted_pos_num=all_items_info[choose_item]["pasted_pos_num"]

    choice = random.choice(range(pasted_pos_num))
    size_key=np.random.choice(list(item_info.keys()),p=[0.15,0.7,0.15])
    choose_item=item_info[size_key]
    im=choose_item["image"]
    top=choose_item["top"]
    bottom=choose_item["bottom"]
    dists=choose_item["dists"]
    im_w, im_h = im.size

    mirror_im=choose_item["mirror_image"]
    mirror_top=choose_item["mirror_top"]
    mirror_bottom=choose_item["mirror_bottom"]
    # mirror flip
    if random.random()>0.5:
      im,top,bottom=mirror_im,mirror_top,mirror_bottom
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
      rotated_im,box,ita,start=paste_item(angle, dists[choice], top[choice], bottom, temp_start,im)

      # record paste information and rotated ends
      rotated_start = cal_rotated_point(top[choice], (im_w // 2, im_h // 2), ita)
      rotated_ends = [cal_rotated_point(point, (im_w // 2, im_h // 2), ita) for point in top]
      ends = [(start[0] - rotated_start[0] + point[0], start[1] - rotated_start[1] + point[1]) for point in rotated_ends]
      paste_info.append([rotated_im,ends,box])

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
        rotated_im, ends, box = paste_info[ix]

        if use_gap:
          if box[2]+continual_gap>w or box[3]+continual_gap>h:
            continue
        # update boxes
        boxes.append(box)
        res.paste(rotated_im,(box[0],box[1]), rotated_im)
        res,boxes=check_continuity(res,rotated_im,box,boxes)

        if not verbose:
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

  save_path=os.path.join(save_dir,str(uuid.uuid4())+".png")
  res.save(save_path)


def curve(num=1,item_path="2.png",max_l = 320, w=1500,h=1000):
  """
  :param num: how many curve need to be generated
  :param item_path: the path of item which need to be pasted in the image
  :param max_l: max length of the item
  :param w: w of output image
  :param h: h of output image
  :return: None
  """
  # put generated random point,each item like (x,y,angle)
  curve_point=[]
  res = Image.new('RGBA', (w, h))
  im = Image.open(item_path)
  im, ratio = resize_prop(im, max_l=max_l)
  im_w, im_h = im.size
  # find top and bottom of item
  temp=np.array(im)
  ny,nx=np.where(temp[:,:,3]!=0)
  top=(nx[0],ny[0])
  bottom=(nx[-1],ny[-1])

  # offset=intersection_angle(top,bottom,(top[0],im_h-1))
  # if top[0]>bottom[0]:
  #   offset=-offset

  dist = math.hypot(top[0] - bottom[0], top[1] - bottom[1])

  y_num=round(h/dist)
  w_gap=w//num
  for i in range(num):
    curve_point.append([])
    start=None
    end=None
    for j in range(y_num):
      if j == 0:
        start = (w_gap * i + w_gap // 2, 3*h//4)
      else:
        start=end
      angle=random.randint(-90,90)

      r=math.radians(angle)
      y_gap=dist*math.cos(r)
      x_gap=dist*math.sin(r)
      end=(start[0]+x_gap,start[1]-y_gap)
      curve_point[i].append((end,start,angle))  ########## (start,end,angle)

  draw=ImageDraw.Draw(res)
  for item in curve_point:
    for data in item:
      start,end,_angle=data
      # rotate img to map the line defined by top and bottom
      translated_end=(start[0]-top[0]+bottom[0],start[1]-top[1]+bottom[1])
      ita=intersection_angle(start,translated_end,end) ##### cal angle
      ################################ offset is vital ####################################################
      # if _angle>offset:
      #   ita=-ita
      cross=(translated_end[0]-start[0])*(end[1]-start[1])-(end[0]-start[0])*(translated_end[1]-start[1])
      if cross>0:
        ita=-ita

      rotated_start=cal_rotated_point(top,(im_w//2,im_h//2),ita)
      initial_center=(im_w//2,im_h//2)
      translated_center=(start[0]-rotated_start[0]+initial_center[0],start[1]-rotated_start[1]+initial_center[1])

      rotated_im = im.rotate(ita, expand=True)
      rm_w, rm_h = rotated_im.size
      pasted_start=(int(translated_center[0]-rm_w//2),int(translated_center[1]-rm_h//2))

      res.paste(rotated_im,pasted_start,rotated_im)

      #draw.line([start[0],start[1],end[0],end[1]],fill=(255,0,255),width=6)
  res.save("222.png")



def diamond(item_path="2.png",max_l = 520, w=3000,h=2000):
  # random.seed(33)
  """
  :param num: how many curve need to be generated
  :param item_path: the path of item which need to be pasted in the image
  :param max_l: max length of the item
  :param w: w of output image
  :param h: h of output image
  :return: None
  """
  res = Image.new('RGBA', (w, h))
  res1 = Image.new('RGBA', (w, h))
  im = Image.open(item_path)
  im, ratio = resize_prop(im, max_l=max_l)
  im_w, im_h = im.size
  # find top and bottom of item
  temp=np.array(im)
  ny,nx=np.where(temp[:,:,3]!=0)
  top=(nx[0],ny[0])
  bottom=(nx[-1],ny[-1])
  # offset = intersection_angle(top, bottom, (top[0], im_h - 1))
  # if top[0]>bottom[0]:
  #   offset=-offset

  dist=math.hypot(top[0]-bottom[0],top[1]-bottom[1])

  dia = Diamond(w, h, dist, 4)
  try:
    dia.generate()
  except:
    print ("error when generating diamond")

  draw = ImageDraw.Draw(res1)
  for key in dia.adj_info.keys():
    for item in dia.adj_info[key]:
      draw.line([key[0], key[1], item[0], item[1]], fill=(255, 0, 255), width=5)
  res1.save("diamond.png")

  drew=list()
  for key in dia.adj_info.keys():
    for item in dia.adj_info[key]:
      # if random.random()<0.3:
      #   continue
      # items=dia.adj_info[key]
      # items.sort(key=lambda x:x[1],reverse=True)
      start = key
      end = item
      if (start,end) in drew or (end,start) in drew :
        continue
      ##################################################
      ## two judgement is necessary and vital
      if start[1]>end[1]:
        start,end=end,start
      # angle = intersection_angle(start, end, (start[0], start[1] + 2))
      # if start[0]>end[0]:
      #   angle=-angle
      ###################################################


      drew.append((start,end))
      drew.append((end, start))
      # rotate img to map the line defined by top and bottom
      translated_end=(start[0]-top[0]+bottom[0],start[1]-top[1]+bottom[1])
      ita=intersection_angle(start,translated_end,end) ##### cal angle

      ############################# offset is vital
      # if angle < offset:
      #   ita = -ita

      cross = (translated_end[0] - start[0]) * (end[1] - start[1]) - (end[0] - start[0]) * (translated_end[1] - start[1])
      if cross > 0:
        ita = -ita

      rotated_start=cal_rotated_point(top,(im_w//2,im_h//2),ita)

      initial_center=(im_w//2,im_h//2)
      translated_center=(start[0]-rotated_start[0]+initial_center[0],start[1]-rotated_start[1]+initial_center[1])

      rotated_im = im.rotate(ita, expand=True)
      rm_w, rm_h = rotated_im.size
      pasted_start=(int(translated_center[0]-rm_w//2),int(translated_center[1]-rm_h//2))

      # draw.text((start[0]-random.randint(2,10),start[1]-random.randint(2,10)),"angel={}".format(angle), fill=(0, 0, 0), width=10)
      res.paste(rotated_im,pasted_start,rotated_im)

  res.save("111.png")


class Diamond:
  def __init__(self, w, h, l, num):
    self.w = w
    self.h = h
    self.l = l
    self.num = num
    # each item likes [x,y,remaining angle,adjacent point]  , adjacent point likes (x,y)
    self.q = []
    self.div_angle = int(360 / self.num)

    self.adj_info={}  ### each item is  center:[point1...n] info [x,y,remained angle or single angle???]
    self.points_info={} #### each item is point:remained angle

  def generate(self):
    # initial_x = random.randint(0, self.w)
    # initial_y = random.randint(0, self.h)
    initial_x = self.w//2
    initial_y = self.h//2
    self.q.append([initial_x, initial_y])
    xx=0 #################
    tt=0  #15
    end=False
    while len(self.q) != 0 :
      xx += 1
      # if xx>=9:
      #   break
      node = self.q.pop(0)
      x1, y1 = node
      # ensure point sits inside images,or dont need to draw diamonds around it
      if x1 >= 0 and x1 < self.w and y1 >= 0 and y1 < self.h:
        # TODO: update,donot use []
        solid = []
        hollow = []
        if (x1,y1) in self.adj_info:
          solid.extend(self.adj_info[x1,y1])

        for ix in range(len(solid)-1):
          hollow.append((solid[ix][0]+solid[ix+1][0]-x1,solid[ix][1]+solid[ix+1][1]-y1))
        # if len(solid)==self.num:
        #   hollow.append((solid[0][0]+solid[-1][0]-x1,solid[0][1]+solid[-1][1]-y1))


        while not (x1,y1) in self.adj_info or len(self.adj_info[(x1,y1)]) < self.num :

          # if xx>tt:
          #   break   ####################################################
          if not (x1,y1) in self.adj_info:
            angle = 90
            r = math.radians(angle)
            y_gap = self.l * math.cos(r)
            x_gap = self.l * math.sin(r)
            x2 = int(x1 - x_gap)
            y2 = int(y1 + y_gap)
            self.adj_info[(x1, y1)] = [(x2,y2)]
            solid=[(x2, y2)]  #######################
            self.q.append((x2, y2))  #############################
          else:
            coor=self.adj_info[(x1, y1)][-1]
            x2,y2=coor

          if not (x1,y1) in self.points_info:
            self.points_info[(x1,y1)]=360
          remained_angle=self.points_info[(x1,y1)]
          offset=30
          #rotate_angle = random.randint(self.div_angle-offset,self.div_angle+offset)
          #rotate_angle = random.randint(70, 110)
          l=self.num-len(self.adj_info[(x1,y1)])+1
          div=remained_angle//l
          rotate_angle=random.randint(div-offset,div+offset)
          # TODO: control the angle ##############################################################################
          if rotate_angle>=remained_angle-10:
            print ("angle error {}".format(remained_angle-rotate_angle))
            #break
          x4, y4 = cal_rotated_point((x2, y2), (x1, y1), rotate_angle)
          x4=int(x4)
          y4=int(y4)

          x3 = int(x2 + x4 - x1)
          y3 = int(y2 + y4 - y1)
          solid.append((x4, y4))
          hollow.append((x3, y3))


          # check when enqueue
          if not (x3, y3) in self.points_info or self.points_info[(x3, y3)] > 0:
            self.q.append((x3, y3))
          if not (x4, y4) in self.points_info or self.points_info[(x4, y4)] > 0:
            self.q.append((x4, y4))
          self.adj_info[(x1, y1)].append((x4,y4)) #####################################
          self.points_info[(x1,y1)]=remained_angle-rotate_angle  ###########

          if not (x2,y2) in self.points_info:
            self.points_info[(x2,y2)]=360
          self.points_info[(x2, y2)] = self.points_info[(x2,y2)] - (180-rotate_angle)

          if not (x4, y4) in self.points_info:
            self.points_info[(x4, y4)] = 360
          self.points_info[(x4, y4)] = self.points_info[(x4, y4)] - (180 - rotate_angle)

          if not (x3, y3) in self.points_info:
            self.points_info[(x3, y3)] = 360
          self.points_info[(x3, y3)] = self.points_info[(x3, y3)] - rotate_angle

        # TODO: last updation maybe error ????????????????????????????????
        last=(solid[0][0]+solid[-1][0]-x1,solid[0][1]+solid[-1][1]-y1)
        hollow.append(last)
        if not last in self.points_info or self.points_info[last] > 0:
          self.q.append(last)

        # hollow = hollow[:self.num]   ################################################################################
        # print ("after hollow ", len(hollow))
        if len(hollow)!=self.num or len(solid)!=self.num:
          print ("hollow ",hollow)
          print ("solid ",solid)
          end=True
          break

        if not last in self.points_info:
          self.points_info[last]=360
        self.points_info[last]-=self.points_info[(x1,y1)]
        self.points_info[solid[0]] -= (180-self.points_info[(x1, y1)])
        ################################################ 之前忘记更新了
        self.points_info[solid[-1]]-= (180-self.points_info[(x1, y1)])
        self.points_info[(x1,y1)]=0


        for ix,item in enumerate(solid):

          if not item in self.adj_info:
            self.adj_info[item]=[]

          # TODO : must sort the points (逆时针，如果有前两个点)

          t = ix - 1 + self.num if ix - 1 < 0 else ix - 1

          temp=[]
          new=[hollow[ix],(x1,y1),hollow[t]]
          old=self.adj_info[item]
          ix1=[]
          ix2=[]
          s1=set()
          s2=set()
          s1.update(new)
          s2.update(old)
          common=s1&s2
          for data in common:
            ix1.append(new.index(data))
            ix2.append(old.index(data))
          if len(ix1)!=0:
            ix1.sort()
            ix2.sort()
            if (ix1[0]!=0):
              temp.extend(new)
              if ix2[-1]+1<len(old):
                temp.extend(old[ix2[-1]+1:])
            else:
              temp.extend(old)
              if ix1[-1]+1<len(new):
                temp.extend(new[ix1[-1]+1:])
          else:
            for p in old:
              if not p in new:
                new.append(p)
            temp=new
          self.adj_info[item] = temp
          # temp=[hollow[ix],(x1,y1),hollow[t]]
          # for p in self.adj_info[item]:
          #   if not p in temp:
          #     temp.append(p)
          #
          #
          # self.adj_info[item]=temp


          # if not hollow[ix] in self.adj_info[item]:
          #   self.adj_info[item].append(hollow[ix])
          # if not (x1,y1) in self.adj_info[item]:
          #   self.adj_info[item].append((x1,y1))
          # if not hollow[t] in self.adj_info[item]:
          #   self.adj_info[item].append(hollow[t])

        for ix, item in enumerate(hollow):

          if not item in self.adj_info:
            self.adj_info[item] = []
          t = (ix + 1) % self.num

          temp = []
          new = [solid[t],solid[ix]]
          old = self.adj_info[item]
          ix1 = []
          ix2 = []
          s1 = set()
          s2 = set()
          s1.update(new)
          s2.update(old)
          common = s1 & s2
          for data in common:
            ix1.append(new.index(data))
            ix2.append(old.index(data))
          if len(ix1) != 0:
            ix1.sort()
            ix2.sort()
            if (ix1[0] != 0):
              temp.extend(new)
              if ix2[-1] + 1 < len(old):
                temp.extend(old[ix2[-1] + 1:])
            else:
              temp.extend(old)
              if ix1[-1] + 1 < len(new):
                temp.extend(new[ix1[-1] + 1:])
          else:
            for p in old:
              if not p in new:
                new.append(p)
            temp = new

          self.adj_info[item] = temp

          # temp=[solid[t],solid[ix]]
          # for p in self.adj_info[item]:
          #   if not p in temp:
          #     temp.append(p)
          # self.adj_info[item] = temp


          # if not solid[t] in self.adj_info[item]:
          #   self.adj_info[item].append(solid[t])
          # if not solid[ix] in self.adj_info[item]:
          #   self.adj_info[item].append(solid[ix])


      if end:break

# 1.png
# top=[(882,186),(417,720),(438,1410)]
# bottom = (588,1944)

# 2.png
# top = [(434, 874), (1688, 3019), (1501, 1881)]
# bottom = (1941, 3976)

# 4.png
# top = [(1276,1611), (874,1303)]
# bottom = (814,3366)

# 5.png
# top = [(576,1092), (906,441)]
# bottom = (984,1878)

# 6.png
# top = [(314,382),(343,208)]
# bottom = (60, 743)
def main():
  items=[{"path":"1.png","top":[(882,186),(417,720),(438,1410)],"bottom":(588,1944)}
         ,{"path":"2.png","top":[(434, 874), (1688, 3019), (1501, 1881)],"bottom":(1941, 3976)}
         ,{"path":"4.png","top":[(1276,1611), (874,1303)],"bottom":(814,3366)}
         ,{"path":"5.png","top":[(576,1092), (906,441)],"bottom":(984,1878)}
         ,{"path":"6.png","top":[(314,382),(343,208)],"bottom":(60, 743)}]

  for i in range(1,5):
    print (i)
    choosed_items = random.sample(items,i)
    #choosed_items=[items[1]]
    concat(items=choosed_items)
    #exit(0)
  # for item in items:
  #   print (item["path"])
  #   concat(item_path=item["path"],top=item["top"],bottom=item["bottom"])




if __name__=='__main__':

 main()
