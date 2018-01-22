import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# La imagenes tienen dimension de 28x28
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# none puede tomar el tamano de observciones q ingresemos
# imagen del numero descompuesta a un vector
x = tf.placeholder(tf.float32, [None, 784])
# Matriz de pesos, 784 para recibir la imagen, 10 por las posible salidas
P = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))  # Vector con bias
# La operacion que se hara en los nodos que reciben entradas
y = tf.matmul(x, P) + b
# Matriz con las etiquetas REALES del set de datos
yR = tf.placeholder(tf.float32, [None, 10])

# Definir la funcion de costo entropia cruzada (Cross Entropy) para poder medir el error. La salida sera con Softmax
softmax = tf.nn.softmax_cross_entropy_with_logits(labels=yR, logits=y)
costo = tf.reduce_mean(softmax)
optimizador = tf.train.GradientDescentOptimizer(0.5).minimize(costo)

# Correr la grafica computacional
# Nos da arreglo de booleanos para decirnos
prediccion = tf.equal(tf.argmax(y, 1), tf.argmax(yR, 1))
# cuales estan bien y cuales no
# Nos da el porcentaje sobre el arreglo de prediccion
accuracy = tf.reduce_mean(tf.cast(prediccion, tf.float32))
Produccion = tf.argmax(y, 1)
init = tf.global_variables_initializer()


# Entrenar algoritmo
# Funcion que usaremos para ver que tan bien va a aprendiendo nuestro modelo
def avance(epoca_i, sess, last_features, last_labels):
    costoActual = sess.run(
        costo, feed_dict={x: last_features, yR: last_labels})
    Certeza = sess.run(accuracy, feed_dict={
                       x: mnist.validation.images, yR: mnist.validation.labels})
    print(
        'Epoca: {:<4} - Costo: {:<8.3} Certeza: {:<5.3}'.format(epoca_i, costoActual, Certeza))


with tf.Session() as sess:
    sess.run(init)
    for epoca_i in range(1000):
        lotex, lotey = mnist.train.next_batch(3)
        #print ("epoca_i:", epoca_i)
        #print ("lotex:", lotex)
        print ("lotey:", lotey)
        # for data in lotex:
        #   print ("data XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX: ", data)
        # for data in lotey:
        #print ("data YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY: ", lotey)

        sess.run(optimizador, feed_dict={x: lotex, yR: lotey})
        if (epoca_i % 50 == 0):
            # pass
            avance(epoca_i, sess, lotex, lotey)
           #print('RESULTADO FINAL: ',sess.run(accuracy, feed_dict={x: mnist.test.images,yR: mnist.test.labels}))
           #print ('Resultado de una imagen',sess.run(Produccion,feed_dict={x: mnist.test.images[5].reshape(1,784)}))
''' Epoca: 0    - Costo: 1.8      Certeza: 0.111
Epoca: 50   - Costo: 0.382    Certeza: 0.875
Epoca: 100  - Costo: 0.407    Certeza: 0.898
Epoca: 150  - Costo: 0.316    Certeza: 0.896
Epoca: 200  - Costo: 0.34     Certeza: 0.908
Epoca: 250  - Costo: 0.313    Certeza: 0.912
Epoca: 300  - Costo: 0.286    Certeza: 0.907
Epoca: 350  - Costo: 0.391    Certeza: 0.914
Epoca: 400  - Costo: 0.509    Certeza: 0.909
Epoca: 450  - Costo: 0.478    Certeza: 0.914
Epoca: 500  - Costo: 0.506    Certeza: 0.914
Epoca: 550  - Costo: 0.215    Certeza: 0.917
Epoca: 600  - Costo: 0.273    Certeza: 0.914
Epoca: 650  - Costo: 0.227    Certeza: 0.917
Epoca: 700  - Costo: 0.224    Certeza: 0.919
Epoca: 750  - Costo: 0.177    Certeza: 0.922
Epoca: 800  - Costo: 0.324    Certeza: 0.919
Epoca: 850  - Costo: 0.248    Certeza: 0.924
Epoca: 900  - Costo: 0.258    Certeza: 0.923
Epoca: 950  - Costo: 0.162    Certeza: 0.918
RESULTADO FINAL:  0.92
Resultado de una imagen [1]
In [8]:
mnist.test.labels[5]
Out[8]:
array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
In [ ]: '''
