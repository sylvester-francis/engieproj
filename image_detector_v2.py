import eel
eel.init('web')
import sys,glob
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2  
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile
@eel.expose
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    if image.getdata().mode == "RGBA":
        image = image.convert('RGB')
    np_array = np.array(image.getdata())
    reshaped = np_array.reshape((im_height, im_width, 3))
    return reshaped.astype(np.uint8)
@eel.expose     
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                        real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                        real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict 


@eel.expose   
def get_num_classes(pbtxt_fname):
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())    
    

@eel.expose    
def image_input():
    print('I2356990909-0-0-0==--0808')
    image_path = './data/images/input_images'
    try:
        test_record_fname = './data/annotations/test.record'
        train_record_fname = './data/annotations/train.record'
        label_map_pbtxt_fname = './data/annotations/label_map.pbtxt'
        pb_fname = './assets/inference_graphs/engie.pb'
        IMAGE_SIZE = (12, 8)
        PATH_TO_CKPT = pb_fname
        PATH_TO_LABELS = label_map_pbtxt_fname
        num_classes = get_num_classes(label_map_pbtxt_fname)
        assert os.path.isfile(pb_fname)
        assert os.path.isfile(PATH_TO_LABELS)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories) 
        for direct,subdirect,images in os.walk(image_path):
            for i,image_name in enumerate(images):
                image = Image.open(str(direct)+'/'+str(image_name))
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                output_dict = run_inference_for_single_image(image_np, detection_graph)
                image_res,class_name = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=10) 
                image_np= cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                if class_name == 'Critical':
                    print('Critical',image_name)
                    cv2.imwrite('./web/critical/image_'+str(i)+'.jpg',image_np)
                elif class_name == 'High':
                    print('High',image_name)
                    cv2.imwrite('./web/high/image_'+str(i)+'.jpg',image_np)
                elif class_name == 'Less':
                    print('less',image_name)
                    cv2.imwrite('./web/less/image_'+str(i)+'.jpg',image_np)
                else:
                    pass

    except Exception as e:
        print("Please select another image because the current image gives an irregular array shape",e)    
@eel.expose()                                                              
def call_from_html():
    image_input()
    print("function called from html")
    return "Hello,from python"    

if __name__ == "__main__":
    eel.start('Dashboardhtml.html', size=(720,640))  


