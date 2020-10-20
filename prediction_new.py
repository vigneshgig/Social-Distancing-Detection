# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
from imutils import perspective
from scipy.spatial import distance as dist
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode ,Visualizer
from collections import OrderedDict
from numpy import array, diff, where, split
# division = [40 for i in range(5)]
# division.extend([65,70,80,85,85])
# division = [40 for i in range(5)]
# division.extend([130,130,130,130,130])
# division_x =[0 for i in range(10)]
# division_y = [0 for i in range(10)]
# division = [7,10,25,70,70,70,100,100,130,130,200,200,210,210,230]
# division =[exponent**2 for exponent in range(3, 13)]
division = [170*0.77**i for i in range(10)]
division.reverse()
division_y = [100,60,50,40,30,20,-20,-40,-60,-80]
division_y.reverse()
division_x = [40,30,20,10,5,-10,-20,-30,40,-50]
division_x.reverse()
y_division = 540/10



class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.frame_count = 0
        self.maximum_wait = OrderedDict()
        self.all_track_id = []
        self.count = 0
        self.time_count = 0

    def create_track(self,id):
        self.objects[id] = 1
    def disappear(self,id):
        if id in self.maximum_wait:
            self.maximum_wait[id] +=1
        else:
            self.maximum_wait[id]  = 1
    def detrack(self, id,index):
        del self.maximum_wait[id]
        del self.objects[id]
        del self.all_track_id[index]
        
    def update(self, id):
        self.objects[id] += 1
        
    
    
        

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # print('=====================>',predictions['instances'].pred_classes)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
    
            if success:
                frame = cv2.resize(frame,(960,540),interpolation = cv2.INTER_CUBIC)

                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                k = 0
                try:
                    vis_frame,colors = video_visualizer.draw_instance_predictions(frame, predictions)
                    k = 1
                except:
                    vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
                if k == 1:    
                    boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
                    classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
                    person_list = []
                    person_track = []
                    for box,class_label,color in zip(boxes,classes,colors):
                        if int(class_label) == 0:
                            pixel_width  = box[2]-box[0]
                            # print(box,'=========================>')
                            # print(pixel_width,'============================>')
                            box = np.asarray([[box[0],box[1]],[box[2],box[3]]])
                            # pixel_per_metric = 15.45
                            # original_width = pixel_width * pixel_per_metric
                            # distance_z = (original_width*3)/pixel_width  #D’ = (W x F) / P  
                            distance_z = pixel_width
                            cX = np.average(box[:, 0])
                            cY = np.average(box[:, 1])
                            # cY = cY + distance_z
                            person_list.append([cX,cY,distance_z])
                            person_track.append(color)
                    # print('<=============================>',person_list,'<=============================>')
            #find the center of the box by top-left x and bottom-right x / 2 and same for y 
        
                  
                
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
        
            # vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            # D = dist.cdist(person_list,person_list,'euclidean')
                # print(person_list,D)
            # def midpoint(ptA, ptB):
	        #     return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
            self.time_count += 1
            
            vis_frame = frame
            if k == 1:
                person = sorted(zip(person_list,person_track))
                
                hh,ww,c = (540,960,3)
                # hh,ww,c = vis_frame.shape
                # aspect_ratio = 960/540
                
                
                # width_scale = (530/960)
                # height_scale = (600/540)
                # result_width = int(vis_frame.shape[1]*width_scale)
                # result_height= int(vis_frame.shape[0]*height_scale)
                # result = np.zeros((result_width,result_height, 3))
                result = np.zeros((530,600,3))
                # x_scale = (result_width/vis_frame.shape[1])
                # y_scale = (result_height/vis_frame.shape[0])
                x_scale = (530/vis_frame.shape[1])
                y_scale = (600/vis_frame.shape[0])
                ht,wd,cc = result.shape
                # print(ww,wd)
                xx = (ww - wd) // 2
                yy = (hh - ht) // 2
                # print(xx, yy,'.................')
                color = (245,245,245)
                layer1 = np.full((hh,ww,cc), color, dtype=np.uint8)
                
                green_list = []
                yellow_list = []
                red_list = []
                for box_i,track_i in person:
                    for box_j,track_j in person:
                        objectid = str(track_i)+str(track_j)
                        objectid = objectid.replace('[','').replace(']','').replace('.','').replace(' ','')
                        if self.time_count % 10:
                            self.time_count = 0
                            for indexs,l in enumerate(self.all_track_id):
                                if l != objectid:
                                    self.disappear(l)
                                    if self.maximum_wait[l] >= 10000:
                                        self.detrack(l,indexs)
                            
                        if box_i != box_j:
                            xA,yA,zA = box_i
                            xB,yB,zB = box_j
                            z_check = abs(zA-zB)
                            D = dist.euclidean((xA,yA),(xB,yB))
                            division_index_A= yA/y_division
                            division_index_B= yB/y_division
                            A_div = division[int(division_index_A)]
                            B_div = division[int(division_index_B)]
                            yA = abs(yA + A_div)
                            yB = abs(yB + B_div)
                            xA = abs(xA + A_div)
                            xB = abs(xB + B_div)
                            
                            if abs(division_index_A - division_index_B) < 1.0:
                                Main_threshold = min(A_div,B_div)
                            else:
                                Main_threshold = 0.4
                            # cv2.line(vis_frame, (int(xA), int(yA)), (int(xB), int(yB)),
                            #             (255,0,0), 2)
                            # def midpoint(ptA, ptB):
	                        #     return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
                            # (mX, mY) = midpoint((xA, yA), (xB, yB))
                            # cv2.putText(vis_frame, "{:.1f}in".format(D), (int(mX), int(mY - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)
                            # # print('.........  ...')
                            if D < Main_threshold:  
                                if objectid in self.objects:
                                    self.update(id=objectid)
                                else:
                                    self.all_track_id.append(objectid)
                                    self.create_track(id=objectid)
                                if self.objects[objectid] <= 90:
                                    xA,yA,zA = box_i
                                    xB,yB,ZB = box_j
                                    # cv2.circle(vis_frame, (int(xA), int(yA)), 5, (255,0,0), -1)
                                    # cv2.circle(vis_frame, (int(xB), int(yB)), 5, (255,0,0), -1)
                                    # overlay = vis_frame.copy()
                                    cv2.circle(vis_frame, (int(xA), int(yA)), 3, (0,255,255), -1)
                                    cv2.circle(vis_frame, (int(xB), int(yB)), 3, (0,255,255), -1)
                                    cv2.line(vis_frame, (int(xA), int(yA)), (int(xB), int(yB)),
                                        (255,255,0), 2)
                                    if box_i not in red_list and box_i not in yellow_list:
                                        yellow_list.append(box_i)
                                        new_box_i_x = int(round((box_i[0]) * x_scale))
                                        new_box_i_y = int(round((box_i[1]) * y_scale))
                                        new_box_j_x = int(round((box_j[0]) * x_scale))
                                        new_box_j_y = int(round((box_j[1]) * y_scale))
                                        cv2.line(result, (int(new_box_i_x), int(new_box_i_y)), (int(new_box_j_x), int(new_box_j_y)),
                                        (255,255,0), 2)
                                    
                                
                                    # cv2.addWeighted(overlay, 0.1, vis_frame, 1 - 0.,0, vis_frame)
                                    
                                    
                                else:
                                    xA,yA,zA = box_i
                                    xB,yB,zB = box_j
                                    # overlay = vis_frame.copy()
                                    cv2.circle(vis_frame, (int(xA), int(yA)), 3, (0,0,255), -1)
                                    cv2.circle(vis_frame, (int(xB), int(yB)), 3, (0,0,255), -1)
                                    cv2.line(vis_frame, (int(xA), int(yA)), (int(xB), int(yB)),
                                        (255,0,0), 2)
                                    if box_i not in red_list:
                                        red_list.append(box_i)
                                        new_box_i_x = int(round((box_i[0]) * x_scale))
                                        new_box_i_y = int(round((box_i[1]) * y_scale))
                                        new_box_j_x = int(round((box_j[0]) * x_scale))
                                        new_box_j_y = int(round((box_j[1]) * y_scale))
                                        cv2.line(result, (int(new_box_i_x), int(new_box_i_y)), (int(new_box_j_x), int(new_box_j_y)),
                                        (0,0,255), 2)
                                    
                            else:
                                if box_i not in red_list and box_i not in yellow_list and box_i not in green_list:
                                    green_list.append(box_i)
                                if box_j not in red_list and box_j not in yellow_list and box_j not in green_list:
                                    green_list.append(box_j)
                for box_check,track_check in person:
                    if box_check in red_list:   
                        new_box_i_x = int(round((box_check[0]) * x_scale))
                        new_box_i_y = int(round((box_check[1]) * y_scale))
                        # track_i = track_i * 255.0
                        cv2.circle(result, (new_box_i_x,new_box_i_y), 5,(0,0,255), 5)
                    elif box_check in yellow_list:
                        new_box_i_x = int(round((box_check[0]) * x_scale))
                        new_box_i_y = int(round((box_check[1]) * y_scale))
                        # track_i = track_i * 255.0
                        cv2.circle(result, (new_box_i_x,new_box_i_y), 5,(0,255,255), 5)
                    elif box_check in green_list:
                        new_box_i_x = int(round((box_check[0]) * x_scale))
                        new_box_i_y = int(round((box_check[1]) * y_scale))
                        # track_i = track_i * 255.0
                        cv2.circle(result, (new_box_i_x,new_box_i_y), 5,(0,128,0), 5)
                cv2.putText(result, "{:.1f}".format(len(red_list)), (int(20), int(40)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),5)
                cv2.putText(result, "{:.1f}".format(len(yellow_list)), (int(20), int(70)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 5)
                cv2.putText(result, "{:.1f}".format(len(green_list)), (int(20), int(100)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 5)
                # for i in range(1,16):
                #     xA = 1
                #     yA = y_division * i
                #     xB = 700
                #     yB = yA
                    
                #     cv2.line(vis_frame, (int(xA), int(yA)), (int(xB), int(yB)),(255,0,0), 2)
                    
                    # print(vis_frame.shape,layer1.shape)
                    # cv2.imwrite('imagetest.jpg',layer1)
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                layer1[yy:yy+ht,xx:xx+wd] = result
                # vis_frame = cv2.resize(vis_frame,(960,540),interpolation = cv2.INTER_CUBIC)
                vis_frame = np.concatenate((vis_frame, layer1), axis=1)

            else:
                vis_frame = cv2.resize(vis_frame,(960,540),interpolation = cv2.INTER_CUBIC)
                hh,ww,c = vis_frame.shape
                result = np.zeros((530,600,3))
                # x_scale = (result_width/vis_frame.shape[1])
                # y_scale = (result_height/vis_frame.shape[0])
                x_scale = (530/vis_frame.shape[1])
                y_scale = (600/vis_frame.shape[0])      
                ht,wd,cc = result.shape
                # print(ww,wd)
                xx = (ww - wd) // 2
                yy = (hh - ht) // 2
                # print(xx, yy,'.................')
                color = (245,245,245)
                layer1 = np.full((hh,ww,cc), color, dtype=np.uint8)
                layer1[yy:yy+ht,xx:xx+wd] = result
                vis_frame = cv2.resize(vis_frame,(960,540),interpolation = cv2.INTER_CUBIC)
                # print(layer1.shape,vis_frame.shape)
                vis_frame = np.concatenate((vis_frame, layer1), axis=1)
                
                                # cv2.addWeighted(overlay, 0.1, vis_frame, 1 - 0.1,0, vis_frame)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
