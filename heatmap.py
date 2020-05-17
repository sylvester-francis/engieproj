import cv2,os

def get_file_directory():
    file_path = input('Enter the directory with images: ')
    return file_path

def read_images(file_path,save_directory):
    for directory,subdirectory,files in os.walk(file_path):
        for i,file in enumerate(files):
             process_images(directory+'/'+file,save_directory,i)

def process_images(img_path,save_directory,img_no):
    img = cv2.imread(img_path, 1)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
    filename = save_directory + 'Heat_map' + '_' + str(img_no) + '.jpg'
    cv2.imwrite(filename, heatmap_img)



if __name__ == '__main__':
    file_directory = get_file_directory()
    directory_to_save = input('Enter directory for file to be saved:')
    read_images(file_directory,directory_to_save)



