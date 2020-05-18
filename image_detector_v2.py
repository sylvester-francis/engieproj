from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
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

class Ui_MainWindow(QObject):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(340, 241)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(340, 241))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./assets/img/icons8-camera-50.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 321, 181))
        self.groupBox.setObjectName("groupBox")
        self.widget = QtWidgets.QWidget(self.groupBox)
        self.widget.setGeometry(QtCore.QRect(10, 20, 301, 151))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.select_object = QtWidgets.QComboBox(self.widget)
        self.select_object.setObjectName("select_object")
        self.select_object.addItem("")
        self.select_object.addItem("")
        self.horizontalLayout.addWidget(self.select_object)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.file_path = QtWidgets.QLineEdit(self.widget)
        self.file_path.setObjectName("file_path")
        self.horizontalLayout_2.addWidget(self.file_path)
        self.browse_button = QtWidgets.QPushButton(self.widget)
        self.browse_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.browse_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("./assets/img/icons8-folder-16.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.browse_button.setIcon(icon1)
        self.browse_button.setObjectName("browse_button")
        self.horizontalLayout_2.addWidget(self.browse_button)
        self.browse_button.clicked.connect(self.select_file)
        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.process_button = QtWidgets.QPushButton(self.widget)
        self.process_button.setFocusPolicy(QtCore.Qt.NoFocus)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./assets/img/icons8-checkmark-32.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.process_button.setIcon(icon2)
        self.process_button.setObjectName("process_button")
        self.process_button.clicked.connect(self.process)
        self.horizontalLayout_3.addWidget(self.process_button)
        self.cancel_button = QtWidgets.QPushButton(self.widget)
        self.cancel_button.setFocusPolicy(QtCore.Qt.NoFocus)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("./assets/img/icons8-cancel-32.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.cancel_button.setIcon(icon3)
        self.cancel_button.setObjectName("cancel_button")
        self.cancel_button.clicked.connect(self.close)
        self.horizontalLayout_3.addWidget(self.cancel_button)
        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 340, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionClose.triggered.connect(self.close)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionHelp = QtWidgets.QAction(MainWindow)
        self.actionHelp.setObjectName("actionHelp")
        self.actionShortcuts = QtWidgets.QAction(MainWindow)
        self.actionShortcuts.setObjectName("actionShortcuts")
        self.menuFile.addAction(self.actionClose)
        self.menuHelp.addAction(self.actionAbout)
        self.menuHelp.addAction(self.actionHelp)
        self.menuHelp.addAction(self.actionShortcuts)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Detector "))
        self.groupBox.setTitle(_translate("MainWindow", "Image Detector"))
        self.label.setText(_translate("MainWindow", "Select Object to be detected:"))
        self.select_object.setItemText(0, _translate("MainWindow", "LiveCam"))
        self.select_object.setItemText(0, _translate("MainWindow", "Image"))
        self.label_2.setText(_translate("MainWindow", "Choose Image:"))
        self.browse_button.setShortcut(_translate("MainWindow", "Ctrl+B"))
        self.process_button.setText(_translate("MainWindow", "Process"))
        self.process_button.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.cancel_button.setText(_translate("MainWindow", "Cancel"))
        self.cancel_button.setShortcut(_translate("MainWindow", "Ctrl+Shift+X"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionClose.setStatusTip(_translate("MainWindow", "Close Application "))
        self.actionClose.setShortcut(_translate("MainWindow", "Ctrl+X"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setStatusTip(_translate("MainWindow", "Open About "))
        self.actionAbout.setShortcut(_translate("MainWindow", "Ctrl+Shift+A"))
        self.actionHelp.setText(_translate("MainWindow", "Help"))
        self.actionHelp.setStatusTip(_translate("MainWindow", "Open Help Document"))
        self.actionHelp.setShortcut(_translate("MainWindow", "Ctrl+H"))
        self.actionShortcuts.setText(_translate("MainWindow", "Shortcuts"))
        self.actionShortcuts.setStatusTip(_translate("MainWindow", "Open Shortcuts"))
        self.actionShortcuts.setShortcut(_translate("MainWindow", "Ctrl+Shift+S"))

    @pyqtSlot()    
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        if image.getdata().mode == "RGBA":
            image = image.convert('RGB')
        np_array = np.array(image.getdata())
        reshaped = np_array.reshape((im_height, im_width, 3))
        return reshaped.astype(np.uint8)
    @pyqtSlot()    
    def run_inference_for_single_image(self,image, graph):
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
    @pyqtSlot()
    def get_num_classes(self,pbtxt_fname):
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())    
    
    @pyqtSlot()
    def select_file(self):
        self.dir_name,_ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "","Images (*.jpg *.png);")
        if self.dir_name:
            self.file_path.setText(self.dir_name)   
    @pyqtSlot()
    def close(self):
        buttonReply = QMessageBox.question(None, 'Confirm', "Are you sure you want to close?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            sys.exit(1)             
    

    @pyqtSlot()
    def live_cam(self):
        label_map_pbtxt_fname = './data/annotations/label_map.pbtxt'
        PATH_TO_LABELS = label_map_pbtxt_fname
        pb_fname = './assets/inference_graphs/engie.pb'
        PATH_TO_CKPT = pb_fname
        PATH_TO_LABELS = label_map_pbtxt_fname
        NUM_CLASSES = self.get_num_classes(label_map_pbtxt_fname)
        assert os.path.isfile(pb_fname)
        assert os.path.isfile(PATH_TO_LABELS)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='') 
        try:        
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)
        except Exception as e:
            print(e)        
        try:           
            cap = cv2.VideoCapture(0)
        except Exception as e:
             print("Problem with your web cam please check and restart application")
             sys.exit(1)    
        
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                while True:
                    ret, image_np = cap.read()
                    gray_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
                    image_np_expanded = np.expand_dims(heatmap_img, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
                    cv2.imshow('LiveCam', cv2.resize(heatmap_img, (360, 240)))
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break
    
    @pyqtSlot()
    def image_input(self):
        print("function_called")
        try:
            test_record_fname = './data/annotations/test.record'
            train_record_fname = './data/annotations/train.record'
            label_map_pbtxt_fname = './data/annotations/label_map.pbtxt'
            pb_fname = './assets/inference_graphs/engie.pb'
            IMAGE_SIZE = (12, 8)
            PATH_TO_CKPT = pb_fname
            PATH_TO_LABELS = label_map_pbtxt_fname
            num_classes = self.get_num_classes(label_map_pbtxt_fname)
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
            image_path = self.file_path.text()
            image = Image.open(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = self.load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = self.run_inference_for_single_image(image_np, detection_graph)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=10) 
            image_np= cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)    
            cv2.imshow(image_path.split('/')[-1],image_np)
        except Exception as e:
            print("Please select another image because the current image gives an irregular array shape",e)
   
    

    @pyqtSlot()
    def process(self):
        if str(self.file_path.text()) =="LiveCam":
            self.live_cam()  
        else:
            self.image_input()          
            
                
                                                              

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


