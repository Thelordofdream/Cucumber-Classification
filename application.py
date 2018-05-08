import tensorflow as tf
import model
import numpy as np


def test(model, x_test, y_test, sess):
    temp = np.zeros((9, 9), dtype=np.int)
    for i in range(2):
        test_data, test_label = sess.run([x_test, y_test])
        logits = sess.run([model.output], feed_dict={model.x: test_data, model.y: test_label})
        print(np.where(logits == np.max(logits)))
    print(temp)

if __name__ == "__main__":
    my_network = model.CNN(name="CC_1", depth=3, classes=9, batch_size=1)
    my_network.create_nerual_network()

    x_test, y_test = my_network.load_data_source(filename="./cc_v1/test.tfrecord")

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, "model_v1/model.ckpt")
        print("===== Test Training! =====")
        test(my_network, x_test, y_test, sess)
        coord.request_stop()
        coord.join(threads)