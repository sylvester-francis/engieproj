from flask import Flask, url_for, render_template,request
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
from flask_dropzone import Dropzone

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile


app = Flask(__name__)

app.config.update(
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_MAX_FILES=20,
    DROPZONE_UPLOAD_ACTION='uploadimage',  # URL or endpoint
    DROPZONE_UPLOAD_BTN_ID='upload',
)

dropzone = Dropzone(app)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    if image.getdata().mode == "RGBA":
        image = image.convert('RGB')
    np_array = np.array(image.getdata())
    reshaped = np_array.reshape((im_height, im_width, 3))
    return reshaped.astype(np.uint8)

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

def get_num_classes(pbtxt_fname):
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


@app.route('/process_images/', methods=['GET','POST'])
def image_input():
    image_path = './static/input_images'
    try:
        test_record_fname = './static/annotations/test.record'
        train_record_fname = './static/annotations/train.record'
        label_map_pbtxt_fname = './static/annotations/label_map.pbtxt'
        pb_fname = './static/inference_graphs/engie.pb'
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
                    cv2.imwrite('./static/critical/image_'+str(i)+'.jpg',image_np)
                elif class_name == 'High':
                    print('High',image_name)
                    cv2.imwrite('./static/high/image_'+str(i)+'.jpg',image_np)
                elif class_name == 'Less':
                    print('less',image_name)
                    cv2.imwrite('./static/less/image_'+str(i)+'.jpg',image_np)
                else:
                    pass
        return '', 204


    except Exception as e:
        print("Please select another image because the current image gives an irregular array shape",e)
        return '', 301

# Preprocessing Image
def process_images(img_path,save_directory,img_no):
    img = cv2.imread(img_path, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    filename = save_directory + 'Heat_map' + '_' + str(img_no) + '.jpg'
    cv2.imwrite(filename, heatmap_img)



@app.route('/', methods=['GET','POST'])
def hello():
    return render_template('Dashboardhtml.html'),200

@app.route('/upload-image/', methods=['GET','POST'])
def uploadimage():
    files = glob.glob('./static/input_images/*')
    i = int(sorted(files)[-1].split("_")[-1].split(".")[0]) if len(files) else 0
    for key, f in request.files.items():
        if key.startswith('file'):
            f.save(os.path.join("./static/uploads/", f.filename))
            i += 1
            process_images(os.path.join("./static/uploads/", f.filename),"./static/input_images/",i)
            print(i)
            os.remove(os.path.join("./static/uploads/", f.filename))
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)