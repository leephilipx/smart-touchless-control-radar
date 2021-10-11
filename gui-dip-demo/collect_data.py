from os import chdir, system
from os.path import dirname, join

if __name__ == "__main__":
    
    chdir(join(dirname(dirname(__file__)), 'acconeer-python-exploration', 'gui'))
    system('python main.py')