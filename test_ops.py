import myflow as mf
import numpy as np
a=np.asarray([[1,4,3],[7,9,0]])
b=np.asarray([[2,1,2],[3,3,2]])
A=mf.constant(a)
B=mf.constant(b)
c=A/B
sess=mf.Session()
c_val=sess.run([c])[0]
print(c_val)