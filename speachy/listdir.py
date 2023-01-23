import os

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    files = os.listdir(cur_dir)
    print(files)