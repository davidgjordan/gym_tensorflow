
# 1 setup
import tensorflow
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

a_0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

middle = 30
w_1 = tf.Variable(tf.truncated_normal([784, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 10]))
b_2 = tf.Variable(tf.truncated_normal([1, 10]))

# 1.1.The sigmoid function


def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


# 1.2.The forward propagation
z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)

# 1.3.Difference
diff = tf.subtract(a_2, y)


# 1.4.The sigmoid prime function
def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


#########################

cost = tf.multiply(diff, diff)
step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


# 1.5.Backward propagation
#d_z_2 = tf.multiply(diff, sigmaprime(z_2))
#d_b_2 = d_z_2
#d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)
#
#d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
#d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
#d_b_1 = d_z_1
#d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)
#
#
# 1.6.Updating the network
#eta = tf.constant(0.5)
# step = [
#    tf.assign(w_1,
#              tf.subtract(w_1, tf.multiply(eta, d_w_1))), tf.assign(b_1,
#                                                                    tf.subtract(b_1, tf.multiply(eta,
#                                                                                                 tf.reduce_mean(d_b_1, axis=[0])))), tf.assign(w_2,
#                                                                                                                                               tf.subtract(w_2, tf.multiply(eta, d_w_2))), tf.assign(b_2,
#                                                                                                                                                                                                     tf.subtract(b_2, tf.multiply(eta,
#                                                                                                                                                                                                                                  tf.reduce_mean(d_b_2, axis=[0]))))
#]


# 1.7.Running and testing the training process
acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in xrange(10000):
    batch_xs, batch_ys = mnist.train.next_batch(3)
    sess.run(step, feed_dict={a_0: batch_xs,
                              y: batch_ys})
    if i % 1000 == 0:
        res = sess.run(acct_res, feed_dict={a_0: mnist.test.images[:1000],
                                            y: mnist.test.labels[:1000]})
        print batch_ys
        #print("i: ", i)
        print res

#cost = tf.multiply(diff, diff)
#step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
