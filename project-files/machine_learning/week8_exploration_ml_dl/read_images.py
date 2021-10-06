import cv2, os, numpy as np

def read_prof_images():
    
    '''
    Return numpy arrays of (X, Y, class_labels), with the first dimension of X and Y being the sample_index.
    '''

    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prof_samples', 'extra_samples_oct5')
    dirs = [dir for dir in os.listdir(root_dir) if dir.startswith('class_')]

    X = []
    Y = []
    class_labels = [label[-2:] for label in dirs]

    for ind, dir in enumerate(dirs):
        imageData = [cv2.imread(os.path.join(root_dir, dir, img)) for img in os.listdir(os.path.join(root_dir, dir))]
        X = X + imageData
        Y = Y + [ind] * len(imageData)

    return np.array(X), np.array(Y), class_labels


def read_our_radar_data():
    
    '''
    Return numpy arrays of (X, Y, class_labels), with the first dimension of X and Y being the sample_index.
    '''

    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'radar_data', '2021_10_06_final_gestures')
    dirs = [dir for dir in os.listdir(root_dir) if dir.startswith('gesture_')]

    X = []
    Y = []
    class_labels = [label[8:] for label in dirs]

    for ind, dir in enumerate(dirs):
        radarData = [np.load(os.path.join(root_dir, dir, data))['sample'] for data in os.listdir(os.path.join(root_dir, dir)) if data.endswith('.npz')]
        X = X + radarData
        Y = Y + [ind] * len(radarData)

    return np.array(X), np.array(Y), class_labels



if __name__ == "__main__":
    X, Y, class_labels = read_prof_images()
    print(X.shape, Y.shape, class_labels)
    X, Y, class_labels = read_our_radar_data()
    print(X.shape, Y.shape, class_labels)
