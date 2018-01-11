# #importamos la libreria
# import tensorflow as tf
# #importamos librerias adicionales
# #import numpy as np
# #import matplotlib.pyplot as plt
# #import matplotlib.cm as cm
# #import pandas as pd

# #%matplotlib inline


# # Creacion de Constantes
# # El valor que retorna el constructor es el valor de la constante.

# # creamos constantes a=2 y b=3
# a = tf.constant(2)
# b = tf.constant(3)

# # creamos matrices de 3x3
# matriz1 = tf.constant([[1, 3, 2],
#                        [1, 0, 0],
#                        [1, 2, 2]])

# matriz2 = tf.constant([[1, 0, 5],
#                        [7, 5, 0],
#                        [2, 1, 1]])


#                        # Realizamos algunos calculos con estas constantes
# suma = tf.add(a, b)
# mult = tf.mul(a, b)
# cubo_a = a**3

# # suma de matrices
# suma_mat = tf.add(matriz1, matriz2)

# # producto de matrices
# mult_mat = tf.matmul(matriz1, matriz2)


# # Todo en TensorFlow ocurre dentro de una Sesion

# # creamos la sesion y realizamos algunas operaciones con las constantes
# # y lanzamos la sesion
# with tf.Session() as sess: 
#     print("Suma de las constantes: {}".format(sess.run(suma)))
#     print("Multiplicacion de las constantes: {}".format(sess.run(mult)))
#     print("Constante elevada al cubo: {}".format(sess.run(cubo_a)))
#     print("Suma de matrices: \n{}".format(sess.run(suma_mat)))
#     print("Producto de matrices: \n{}".format(sess.run(mult_mat)))


import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[5.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

# Launch the default graph.
sess = tf.Session()
# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
result = sess.run(product)
print(result)
# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()