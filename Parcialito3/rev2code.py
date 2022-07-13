#coding:utf-8
# Code for running the REV2 algorithm from the paper "REV2: Fraudulent User Prediction in Rating Platforms"
# Modified to:
#  work with networkx>=2.0
#  use 1 as initial values for R F and G instead of birdnest
#  pickle the graph result instead of using csvs

import sys

NETWORKNAME = sys.argv[1] #需要外部输入参数

alpha1 = int(sys.argv[2])
alpha2 = int(sys.argv[3])

beta1 = int(sys.argv[4])
beta2 = int(sys.argv[5])

gamma1 = int(sys.argv[6])
gamma2 = int(sys.argv[7])
gamma3 = int(sys.argv[8])

if gamma1 == 0 and gamma2 == 0 and gamma3 == 0:
        sys.exit(0)

import networkx as nx
import _pickle as cPickle
import numpy

print ("Loading %s network" % NETWORKNAME)
G = cPickle.load(open("./data/%s_network.pkl" % (NETWORKNAME), "rb"))

nodes = G.nodes()
edges = G.edges(data=True)
print("%s network has %d nodes and %d edges" %  (NETWORKNAME, len(nodes), len(edges)))

user_names = [node for node in nodes if "u" in node]
product_names = [node for node in nodes if "p" in node]
num_users = len(user_names)
num_products = len(product_names)
user_map = dict(zip(user_names, range(len(user_names))))
product_map = dict(zip(product_names, range(len(product_names))))

for node in user_names:
    G.nodes[node]["fairness"] = 1
for node in product_names:
    G.nodes[node]["goodness"] = 1
for edge in edges:
    G[edge[0]][edge[1]]["fairness"] = 1

iter = 0

yfg = []
ygood = []
xfg = []
du = 0
dp = 0
dr = 0

##### REV2 ITERATIONS START ######

while iter < 500:
    print ('-----------------')
    print ("Epoch number %d with du = %f, dp = %f, dr = %f, for (%d,%d,%d,%d,%d,%d,%d)" % (iter, du, dp, dr, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3))
    if numpy.isnan(du) or numpy.isnan(dp) or numpy.isnan(dr):
            break
    
    du = 0
    dp = 0
    dr = 0
    
    ############################################################

    print ('Updating goodness of product')

    currentgvals = []
    for node in nodes:
     if "p" not in node[0]:
        continue
     currentgvals.append(G.nodes[node]["goodness"])
    
    median_gvals = numpy.median(currentgvals) # Alternatively, we can use mean here, intead of median
                                              # 对应mu(g)公式

    for node in nodes:
        if "p" not in node[0]:
            continue
        
        inedges = G.edges(node,  data=True)
        ftotal = 0.0 # 对应In(p)公式
        gtotal = 0.0
        for edge in inedges:
            gtotal += edge[2]["fairness"]*edge[2]["weight"]
        ftotal += 1.0

        kl_timestamp = 1  #对应P(p)公式

        if ftotal > 0.0:
            mean_rating_fairness = (beta1*median_gvals + beta2* kl_timestamp + gtotal)/(beta1 + beta2 + ftotal) #对应公式G(p)
        else:
            mean_rating_fairness = 0.0
        
        x = mean_rating_fairness
        
        if x < -1.0:
            x = -1.0
        if x > 1.0:
            x = 1.0
        dp += abs(G.nodes[node]["goodness"] - x)
        G.nodes[node]["goodness"] = x
    
    ############################################################
    
    print ("Updating fairness of ratings")
    for edge in edges:
        rating_distance = 1 - (abs(edge[2]["weight"] - G.nodes[edge[1]]["goodness"])/2.0) #对应公式R(u,p)分子中间一项
        
        user_fairness = G.nodes[edge[0]]["fairness"] #对应F(u)
        ee = (edge[0], edge[1])
        kl_text = 1.0  #对应BA R(u,p)

        x = (gamma2*rating_distance + gamma1*user_fairness + gamma3*kl_text)/(gamma1+gamma2 + gamma3) #对应公式R(u,p)

        if x < 0.00:
            x = 0.0
        if x > 1.0:
            x = 1.0
        
        dr += abs(edge[2]["fairness"] - x)
        G[edge[0]][edge[1]]["fairness"] = x

    ############################################################
    
    currentfvals = []
    for node in nodes:
     if "u" not in node[0]:
        continue
     currentfvals.append(G.nodes[node]["fairness"])
    median_fvals = numpy.median(currentfvals) # Alternatively, we can use mean here, intead of median

    print ('updating fairness of users')
    for node in nodes:
        if "u" not in node[0]:
            continue
        
        outedges = G.edges(node, data=True)
        
        f = 0
        rating_fairness = []
        for edge in outedges:
            rating_fairness.append(edge[2]["fairness"])
        
        for x in range(0,alpha1):
            rating_fairness.append(median_fvals)

        kl_timestamp = 1.0

        for x in range(0, alpha2):
            rating_fairness.append(kl_timestamp)

        mean_rating_fairness = numpy.mean(rating_fairness)

        x = mean_rating_fairness #*(kl_timestamp)
        if x < 0.00:
            x = 0.0
        if x > 1.0:
            x = 1.0

        du += abs(G.nodes[node]["fairness"] - x)
        G.nodes[node]["fairness"] = x
        #print mean_rating_fairness, kl_timestamp
    
    iter += 1
    if du < 0.01 and dp < 0.01 and dr < 0.01:
        break

nx.write_gpickle(G, f"./results/{NETWORKNAME}_network.pkl")
