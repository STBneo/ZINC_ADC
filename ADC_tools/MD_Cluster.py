import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os,sys,glob,random,itertools,multiprocessing
import subprocess

from functools import partial
from time import time
from multiprocessing import Manager,Process
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.manifold import spectral_embedding
import matplotlib
matplotlib.use("Agg")
plt.switch_backend("agg")

def Check_LSAlign():
	if not os.path.exists('./ADC_tools/LSalign'):
		print(" This python program needs \'LSalign\'.")
		print(" Move the \'LSalign\' program at ADC_tools folder and re-excute this program. ")

		return -1
	else:
		return 1

def Run_LSAlign(Re_list,ZID_Dic,list_pairs,apair):

	if Check_LSAlign()==-1:
		sys.exit(1)

	t_id = apair[0]
	q_id = apair[1]
	t_smi = ZID_Dic[t_id]
	q_smi = ZID_Dic[q_id]

	ln_list_pairs = len(list_pairs)
	aidx=list_pairs.index(apair)
	print '     Processing....:'+str(aidx+1)+'/'+str(ln_list_pairs)+' ,mol:'+t_id+'\r',
	sys.stdout.flush()


	pid = os.getpid()

	t_fname=t_id+'_'+str(pid)+'.t.mol2'
	q_fname=q_id+'_'+str(pid)+'.q.mol2'

	arg='obabel -:"'+t_smi+'" --gen2D --addtotitle '+t_id+' -O '+t_fname
	FNULL = open(os.devnull, 'w')
	process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
	process.wait()

	arg='obabel -:"'+q_smi+'" --gen2D --addtotitle '+q_id+' -O '+q_fname
	FNULL = open(os.devnull, 'w')
	process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
	process.wait()

	arg='./ADC_tools/LSalign '+q_fname+' '+t_fname
    #arg='./LSalign '+q_fname+' '+t_fname+' -d0 0.1'
	align = subprocess.Popen(arg, shell=True, stdout=subprocess.PIPE)
	lines = align.stdout.readlines()

    #######################################
    # In case of there is no data in SDF_DB
	if len(lines)<3:
		pcscore=0
		return  pcscore
    #######################################
	try:
		str_test=lines[3]
	except Exception as e:
		print traceback.format_exc()
		print e.message, e.args
		print '\nq_id:',q_id,q_smi
		print lines
		pcscore=0
		return pcscore

	str_error='has a problem'
	pcscore=0
	try:
		if str_test.find(str_error) == -1:
			output = lines[3].split()
			pcscore = float(output[2])
			#pcscore = [float(output[2]),float(output[3]),float(output[7])]
		else:
			output = lines[-6].split()
			pcscore = float(output[2])
			#pcscore = [float(output[2]),float(output[3]),float(output[7])]
	except Exception as e:
		
		print traceback.format_exc()
		print e.message, e.args
		print zid,lines
		pcscore=0
		return pcscore

	os.unlink(q_fname)
	os.unlink(t_fname)

	tmp_list=[]
	tmp_list = [t_id,q_id,pcscore]
	Re_list.append(tmp_list)

	return
def Making_Pair(Main_SMI_list):

	tmp_list=[]
	for pair in itertools.product(Main_SMI_list, repeat=2):
		tmp_list.append(pair)

	set_tmp_list = set(tmp_list)
	re_list_set = set()

	for ele in tmp_list:
		if ele[0] == ele[1]:
			continue
		r_ele = tuple([ele[1],ele[0]])
		if (ele not in re_list_set) and (r_ele not in re_list_set):
			re_list_set.add(ele)

	return list(re_list_set)


def Make_Distance_Mat(Main_ID_list,Main_SMI_list,ZID_Dic):

	time_sp1=time()
	print '\nStart to making the distance matrix using multiprocessing'

	pcs_m = np.zeros((len(Main_SMI_list),len(Main_SMI_list)))
	list_pairs = Making_Pair(Main_ID_list)

	Re_list=Manager().list()

	n_cpu=multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=(n_cpu-1))
	func=partial(Run_LSAlign,Re_list,ZID_Dic,list_pairs)
	pool.map(func,list_pairs)
	pool.close()
	pool.join()

	idx=0
	for ele in Re_list:
		x,y =  Main_ID_list.index(ele[0]),Main_ID_list.index(ele[1])
		pcs_m[x][y]=float(ele[2])
		pcs_m[y][x]=float(ele[2])
		pcs_m[x][x]=1.0
		pcs_m[y][y]=1.0

	print ('  -> Done')

	time_sp2=time()
	print '  -> Processing time to make distance matrix: '+str(time_sp2-time_sp1)+'s for '+str(len(Main_ID_list))+' ZIDs'

	return pcs_m

def Dendrogram_Plot(a,X_cp,aa1):
	
	linked = linkage(X_cp,method="ward")
	
	dendrogram(linked,ax=aa1)

	aa1.set_title("Dendrogram: %s"%a,fontsize=30)
	aa1.set_yticks(np.arange(0,50,5))
	aa1.grid(True,axis="y",linestyle="--")


def DBSCAN_Plot(a,X_cp,aa1):

	clustering = DBSCAN(eps=0.2,min_samples=10).fit(X_cp)

	labels = clustering.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	core_samples_mask = np.zeros_like(labels,dtype=bool)
	core_samples_mask[clustering.core_sample_indices_] = True

	colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(set(labels)))]

	for k,col in zip(set(labels),colors):
		if k == -1:
			col = [0,0,0,1]
		class_member_mask = (labels==k)

		xy = X_cp[class_member_mask&core_samples_mask]
		aa1.plot(xy[:,0],xy[:,1],"o",markerfacecolor=tuple(col),markeredgecolor="k",markersize=20)

		xy = X_cp[class_member_mask& ~core_samples_mask]
		aa1.plot(xy[:,0],xy[:,1],"o",markerfacecolor=tuple(col),markeredgecolor="k",markersize=10)

	aa1.set_xlabel("Estimated number of clusters: %d"%n_clusters_,fontsize=20.0)
	aa1.set_title("Clustering by DBSCAN: %s"%a,fontsize=30)
	aa1.grid(True)

def Kmeans_Plot(a,X_cp,aa1):
	in_cluster = 5
	clustering = KMeans(n_clusters=in_cluster,init="k-means++").fit(X_cp)

	centers = clustering.cluster_centers_
	labels = pairwise_distances_argmin(X_cp,centers)

	colors = [plt.cm.Spectral(each) for each in np.linspace(0,1,len(set(labels)))]

	for k,col in zip(range(in_cluster),colors):
		class_member_mask = (labels==k)

		aa1.plot(X_cp[class_member_mask,0],X_cp[class_member_mask,1],"w",markerfacecolor=tuple(col),marker=".",markeredgecolor="k",markersize=20)
		aa1.plot(centers[k][0],centers[k][1],"o",markerfacecolor=tuple(col),markeredgecolor="k",markersize=30)

	aa1.set_xlabel("Clusters: %d"%in_cluster,fontsize=20.0)
	aa1.set_title("Clustering by KMeans: %s"%a,fontsize=30.0)
	aa1.grid(True)
def Make_ZID_dic(df,a):
	didi = {}
	if len(df) <= a:
		n_samples = len(df)
	else:
		n_samples = int(a)
	
	Main_SMI_list = []
	Main_ID_list = []
	for idx,line in df.iterrows():
		zid = line["ZID"]
		smi = line["SMILES"]
		didi[zid] = smi

	for i in random.sample(didi,n_samples):
		Main_SMI_list.append(didi[i])
		Main_ID_list.append(i)

	return Main_ID_list,Main_SMI_list,didi

def Calc_Dist(df):
	df = df[1:]
	a = 1000 # N.of samples
	Main_ID_list,Main_SMI_list,ZID_Dic = Make_ZID_dic(df,a)
	Dis_Mat = Make_Distance_Mat(Main_ID_list,Main_SMI_list,ZID_Dic)

	return Dis_Mat
if __name__ == "__main__":
	inf = sys.argv[1]
	X = np.load(inf) 
	X_cp = pd.DataFrame(spectral_embedding(X,n_components=2)).values

	X_cp = StandardScaler().fit_transform(X_cp)

	fig = plt.figure(figsize=(30,15))
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	fn = os.path.basename(inf).split(".")[0]

	Dendrogram_Plot(fn,X_cp,ax1)
	DBSCAN_Plot(fn,X_cp,ax2)

	fig.tight_layout()

	if not os.path.exists("%s_PNG"%fn):
		os.makedirs("%s_PNG"%fn)
	else:
		pass
	fig.savefig("%s_PNG/%s.png"%(fn,fn))
