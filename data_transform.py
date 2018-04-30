import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    data_info = unpickle("cucumber-9-p2-python/batches.meta")
    print(data_info)
    writer = tf.python_io.TFRecordWriter("train.tfrecord")
    for i in range(1, 2):
        cucumber = unpickle("cucumber-9-p2-python/data_batch_%d" % i)
        print(cucumber)
        num = len(cucumber[b'data'])
        print(cucumber[b'data'][0])
        print(len(cucumber[b'data']))
        print(len(cucumber[b'filenames']))
    #     for each_cucumber in range(num):
    #         feature = cucumber[b'data'][each_cucumber].tostring()
    #         label = cucumber[b'labels'][each_cucumber]
    #         example = tf.train.Example(
    #             features=tf.train.Features(
    #                 feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    #                          'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
    #         }))
    #         writer.write(example.SerializeToString())
    # writer.close()
    #
    # cucumber = unpickle("cucumber-9-p2-python/data_batch_6")
    # num = len(cucumber[b'data'])
    # writer = tf.python_io.TFRecordWriter("test.tfrecord")
    # for each_cucumber in range(num):
    #     feature = cucumber[b'data'][each_cucumber].tostring()
    #     label = cucumber[b'labels'][each_cucumber]
    #     example = tf.train.Example(
    #         features=tf.train.Features(
    #             feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    #                      'sequence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
    #                      }))
    #     writer.write(example.SerializeToString())
    # writer.close()