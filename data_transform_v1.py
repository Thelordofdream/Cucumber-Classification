import tensorflow as tf
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


if __name__ == "__main__":
    data_info = unpickle("cucumber-9-python/batches.meta")
    batch_num = data_info[b'num_cases_per_batch']
    label_num = len(data_info[b'label_names'])
    writer = tf.python_io.TFRecordWriter("cc_v1/train.tfrecord")
    for i in range(1, 6):
        cucumber = unpickle("cucumber-9-python/data_batch_%d" % i)
        for each in range(batch_num):
            each_cucumber = np.array(cucumber[b'data'][each * 3:(each + 1) * 3], dtype=np.int32).flatten()
            feature = each_cucumber.tostring()
            label = cucumber[b'labels'][each]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                             'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
                             }))
            writer.write(example.SerializeToString())
    writer.close()

    cucumber = unpickle("cucumber-9-python/test_batch")
    writer = tf.python_io.TFRecordWriter("cc_v1/test.tfrecord")
    for each in range(batch_num):
        each_cucumber = np.array(cucumber[b'data'][each * 3:(each + 1) * 3], dtype=np.int32).flatten()
        feature = each_cucumber.tostring()
        label = cucumber[b'labels'][each]
        example = tf.train.Example(
            features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                         'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
                         }))
        writer.write(example.SerializeToString())
    writer.close()
