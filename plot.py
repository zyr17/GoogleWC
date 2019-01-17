# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:48:22 2019

@author: 13236
"""

import matplotlib
matplotlib.use('Agg')

from IPython.display import clear_output
import matplotlib.pyplot as plt

#plt.subplot(131)

plt.figure(figsize=(20,10))
def plot(epoch, record, filename):
    clear_output(True)
    plt.clf()
    plt.title('epoch:{} total:{}'.format(epoch, len(record)))
    #plt.plot(rewards)
    plt.plot([w+1 for w in range(len(record))], record)
    plt.savefig("{}.png".format('imgs/' + filename))
    #plt.show()

if __name__=='__main__':

    plot(2,[3,2,5,4,7],'test')