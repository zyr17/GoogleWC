import numpy as np
import scipy.misc

def saveimg(mat, filename = 'img.png'):
    scipy.misc.imsave(filename, mat.reshape(28,28))

typename = ['airplane', 'ant', 'apple', 'bat', 'baseball', 'bee', 'bed', 'The Eiffel Tower', 'bicycle', 'axe']

def loadquickdrawdata(number = 100):
    filenames = typename
    folder = 'data/'
    resx = []
    resy = []
    for num, fname in enumerate(filenames):
        fname = folder + fname + '.npy'
        tmat = np.load(fname)[:number].reshape(-1, 1, 28, 28).astype('float') / 255 * 2 - 1
        resx += list(tmat)
        resy += [num] * len(tmat)
    return resx, resy
        





