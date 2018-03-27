
import tensorflow as tf 

m1 =tf.constant([[3,3]]) #create constant var
m2 =tf.constant([[2],[3]]) #create constant var

product = tf.matmul(m1,m2) #mutplie m1 m2
print(product)
# define session which could run the graphic
with tf.Session() as sess: #use this method don`t have to close()
	sess = tf.Session()
	result =sess.run(product)# run product to trigger the op
	print(result)
#sess.close()