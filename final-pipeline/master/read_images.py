import cv2, os, numpy as np
import matplotlib.pyplot as plt

def read_prof_images():
    
    '''
    Return numpy arrays of (X, Y, class_labels), with the first dimension of X and Y being the sample_index.
    '''

    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'project-files', 'prof_samples', 'extra_samples_oct5')
    dirs = [dir for dir in os.listdir(root_dir) if dir.startswith('class_')]
    X = []
    Y = []
    class_labels = [label[-2:] for label in dirs]

    for ind, dir in enumerate(dirs):
        imageData = [cv2.cvtColor(cv2.imread(os.path.join(root_dir, dir, img)), cv2.COLOR_BGR2RGB)
                     for img in os.listdir(os.path.join(root_dir, dir))]
        X = X + imageData
        Y = Y + [ind] * len(imageData)

    return np.array(X), np.array(Y), class_labels

def read_stft_plots(folder='images_stft_old'):
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            'project-files', 'radar_data', '2021_10_13_data', folder)
    
    X = []
    Y = []
    class_labels = sorted(list(set([label[:-8] for label in os.listdir(root_dir)])))

    for ind, gesture in enumerate(class_labels):
        imagesPath = [label for label in os.listdir(root_dir) if label.startswith(gesture)]
        imageData = [cv2.cvtColor(cv2.imread(os.path.join(root_dir, img)), cv2.COLOR_BGR2RGB)[60:428, 112:480]
                     for img in imagesPath]
        # plt.imshow(imageData[229][60:428, 112:480])
        # plt.axis('off')
        # plt.show()
        X = X + imageData
        Y = Y + [ind] * len(imageData)
    return np.array(X), np.array(Y), class_labels

if __name__ == "__main__":
    X, Y, class_labels = read_stft_plots(folder='images_stft_old')
    print(X.shape, Y.shape, class_labels)
    # X, Y, class_labels = read_our_radar_data()
    # print(X.shape, Y.shape, class_labels)
