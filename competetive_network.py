import numpy as np

data=[]
while len(data)<32:
    l=np.random.randint(0,2,5)
    if str(l) not in str(data):
        data.append(l)

#data=[[1,0,1],[1,0,0],[0,1,0],[0,1,1]]

def ANN(data,cluster,epochs,eta):
    w=np.abs(np.random.normal(size=(len(data[0]),cluster)))
    clusters={}
    w_ch=[]
    for cl in range(cluster):
        clusters['cluster-'+str(cl)]=[]
    for el in range(w.shape[1]):
        w[:,el]=w[:,el]/np.sum(w[:,el])
    first_w=np.copy(w)
    for epoch in range(epochs):
        error_sum=0
        np.random.shuffle(data)
        for li in data:
            if np.sum(li)==0:
                break
            net=np.dot(li,w)
            max_n=np.argmax(net)
            dw=eta*((li/np.sum(li))-w[:,max_n])
            w[:,max_n]=w[:,max_n]+ dw
            error_sum=error_sum+dw
        if epoch%10==0:
            w_ch.append(error_sum)
    for f_li in data:
        f_net=np.dot(f_li,w)
        f_max=np.argmax(f_net)
        clusters['cluster-'+str(f_max)].append(f_li)
    return first_w,w,clusters,w_ch


first_w,w,clusters,w_ch=ANN(data,2,200,0.1)
print('fisrt weight matrix: ')
print(first_w)
print('---------------------------------------')
print('final weight matrix: ')
print(w)
print('---------------------------------------')
print('weight change in epoch: ')
print(w_ch)
print('---------------------------------------')
print('clusters: ')
print(clusters)
print('---------------------------------------')


    

