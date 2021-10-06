import cv2, os, numpy as np

def read_prof_images(reshape=True):
    '''
    Return numpy arrays of (X, Y, class_labels), with the first dimension being the sample_index.
    '''

    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'prof_samples', 'extra_samples_oct5')
    dirs = [dir for dir in os.listdir(root_dir) if dir.startswith('class_')]

    X = []
    Y = []
    class_labels = [label[-2:] for label in dirs]

    for ind, dir in enumerate(dirs):
        X = X + [cv2.imread(os.path.join(root_dir, dir, img)) for img in os.listdir(os.path.join(root_dir, dir))]
        Y = Y + [ind] * 100

    return np.array(X), np.array(Y), class_labels


if __name__ == "__main__":
    X, Y, class_labels = read_prof_images()
    print(X.shape, Y.shape, class_labels)
