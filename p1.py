import tensorflow as tf 
import numpy as np 

x_data = np.random.rand(100)
y_data =x_data*0.1+0.2

#build linear model
b= tf.Variable(5.)
k= tf.Variable(0.)
y=k*x_data+b

#build a match
loss= tf.reduce_mean(tf.square(y_data-y))

#build optimizer
optimizer =tf.train.GradientDescentOptimizer(0.2)
#build  mini match

train =optimizer.minimize(loss)

init =tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(501):
		sess.run(train)
		if step%20==0:
			print(step,sess.run([k,b]))