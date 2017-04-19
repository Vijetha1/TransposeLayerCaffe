import caffe
import numpy as np

class transpose(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].data.shape[1], bottom[0].data.shape[0])

    def forward(self, bottom, top):
        top[0].data[...] = np.transpose(bottom[0].data[...])

    def backward(self, top, propagate_down, bottom):
        pass
