# Programming assignment 0
# Prereqs: numpy, scipy, tensorflow, scikit-learn, and keras
import keras
import numpy as np
import scipy.stats as st
import sklearn as sk
import sys
import tensorflow as tf



# Your A# is going to be your major input to your calculations. 
# Please be careful about the digits. Here's your guide for digit indices:
# A12345678 being an example A#, 8 is the last digit, 7 is the second to last digit, 
# 6 is the third to the last digit, etc.

# Please implement what is needed at the TODO sections,
# do not modify the other parts of the code.

result_list = list()
result_list.append(sys.version)
result_list.append(keras.__version__)


#TODO: 1. Using the scipy.stats.norm.pdf function, calculate the probability density value of
# "the last digit of your A#", where the mean (i.e., loc) is 5, and standard deviation (i.e., scale) is 15.
# Store the result in a variable called value1.
# Do not change the following line. Obviously, your code needs to preceed the following line :).

anumber = "A20406657"
lastdigit = anumber[-1]
secondlastdigit = anumber[-2]
thirdlastdigit= anumber[-3]
fourthlastdigit=anumber[-4]
value1=st.norm.pdf(int(lastdigit), scale=15, loc=5)
result_list.append(value1)


#TODO: 2. Set the seed of Numpy using np.random.seed() function, and use "the second to the last digit of your A#" as the seed.
#         Then sample a value from the uniform distribution of [0, 1) (use numpy.random.rand()). 
#		  The result should be assigned to value2.

np.random.seed(int(secondlastdigit))
value2=np.random.rand()
result_list.append(value2)


#TODO: 3. Using the tf.random_normal() function, sample a random value from the normal distribution 
#		  with mean=-1 and standard deviaion=4, and use "the third digit to the last of your A#" as the seed.
#         The result should be assigned to value3.

value3=tf.random_normal(shape=[1,],mean=-1,stddev=4,seed=int(thirdlastdigit))
result_list.append(value3)


#TODO: 4. Using sklearn.utils.shuffle(), shuffle the array of [0, 1,...,11]. Set the seed to "the fourth to 
#         the last digit of your A#". 
#		  The result should be assigned to value4.

array4 = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
value4=sk.utils.shuffle(array4,random_state=int(fourthlastdigit))
result_list.append(value4)


# Once you run the code, it will generate a "result.txt" file. Do not modify the following code.
with open("result.txt", "w") as f:
	for v in result_list:
		f.write(str(v))
		f.write("\n")

