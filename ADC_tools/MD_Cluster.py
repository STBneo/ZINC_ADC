import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os,sys,glob,random,itertools,multiprocessing
import subprocess
import networkx as nx
from functools import partial
from time import time
from multiprocessing import Manager,Process
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.manifold import spectral_embedding
from rdkit import Chem
import matplotlib
import argparse


matplotlib.use("Agg")
plt.switch_backend("agg")


def Check_LSAlign():
	if not os.path.exists('./LSalign'):
		print(" This python program needs \'LSalign\'.")
		print(" Rerun this program. ")

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

	#arg='./ADC_tools/LSalign '+q_fname+' '+t_fname
	arg='./LSalign '+q_fname+' '+t_fname
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

    if Check_LSAlign()==-1:
        sys.exit(1)
    
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

    Re_list = list(Re_list)
    #print(Re_list)

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



def Make_ZID_Dic(Main_ID_list,Main_SMI_list):

    tmp_Dic = {}

    idx=0
    for aID in Main_ID_list:
        tmp_Dic[aID] = Main_SMI_list[idx]
        idx+=1

    return tmp_Dic



def Start_Make_DisMat(Main_ID_list,Main_SMI_list,Mat_name,n_sam,rebuild):

    np_name=Mat_name+'.Dis_matrix.npy'

    if rebuild == 'y':
        ZID_Dic = Make_ZID_Dic(Main_ID_list,Main_SMI_list)
        #Dis_Mat = Make_Distance_Mat(Main_ID_list,Main_SMI_list,ZID_Dic)
        Dis_Mat = Make_Distance_Mat(Main_ID_list[0:n_sam],Main_SMI_list[0:n_sam],ZID_Dic)
        np.save('./'+np_name,Dis_Mat)

    if rebuild == 'n':
        if os.path.exists('./'+np_name):
            print ('Loading the distance matrix file\n')
            Dis_Mat = np.load('./'+np_name)
        else:
            print ('There is no Distance matrix file.\n Make Distance matrix file first')
            sys.exit(1)

    return Dis_Mat



##########################################################################

def Dendrogram_Plot(a,X_cp,aa1):
	
	linked = linkage(X_cp,method="ward")
	
	dendrogram(linked,ax=aa1)

	aa1.set_title("Dendrogram: %s"%a,fontsize=30)
	aa1.set_yticks(np.arange(0,50,5))
	#aa1.grid(True,axis="y",linestyle="--")

def Activate_DBSCAN(a,X_cp,nlist,slist,eps,min_s):
	xs = []
	ys = []
	n_cl = []

	temp_df = pd.DataFrame()
	finl_df = {}
	XY = pd.DataFrame()

	XY["X"] = X_cp[:,0]
	XY["Y"] = X_cp[:,1]
	XY["IDs"] = nlist
	XY["Scores"] = slist

	clustering = DBSCAN(eps=eps,min_samples=min_s).fit(X_cp)

	labels = clustering.labels_

	core_samples_mask = np.zeros_like(labels,dtype=bool)
	core_samples_mask[clustering.core_sample_indices_] = True

	for k in set(labels):
		class_member_mask = (labels==k)
		xy = X_cp[class_member_mask]
		for x,y in zip(xy[:,0],xy[:,1]):
			xs.append(x)
			ys.append(y)
			n_cl.append(int(k))
	temp_df["X"] = xs
	temp_df["Y"] = ys
	temp_df["N_Cluster"] = n_cl

	df = pd.merge(XY,temp_df,on=["X","Y"]) # cluster by id

	for i in df["N_Cluster"].drop_duplicates():
		ids = df[df["N_Cluster"]==i]["IDs"].tolist()
		avg = df[df["N_Cluster"]==i]["Scores"].mean()
		finl_df[i] = [len(ids),avg,','.join(ids)]

	finl_df = pd.DataFrame.from_dict(finl_df,orient="index").reset_index()
	finl_df.columns = ["N_Cluster","N.IDs","Avg.PCScore","IDs"]
	finl_df.sort_values(by="Avg.PCScore",inplace=True,ascending=False)
	finl_df.to_csv(a + ".DBSCAN.cluster.csv",index=False) # Extract Result of DBSCAN

	return clustering
		
def DBSCAN_Plot(a,X_cp,clustering,aa1):

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
	#aa1.grid(True)

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
	#aa1.grid(True)


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


def Reading_Input_File(in_file):
    
    fp_for_in = open(in_file,'r')
    head = fp_for_in.readline()
    skip_line = fp_for_in.readline()

    lines =fp_for_in.readlines()
    fp_for_in.close()


    Main_SMI_list=[]
    Main_ID_list=[]
    Main_Score_list = []
    idx=0
    for aline in lines:
        token = aline.split(',')
        Main_ID_list.append(token[0])
        Main_SMI_list.append(token[-1])
        Main_Score_list.append(np.float64(token[1]))

    return Main_ID_list,Main_SMI_list,Main_Score_list


def Cluster_DBSCAN(Mat_name,nlist,slist,epsilon,min_s):

    time_sp1=time()

    print('\nStarting clustering........')

    dis_matrix = Mat_name+'.Dis_matrix.npy'
    
    X = np.load(dis_matrix) 
    X = np.ones((len(X),len(X))) - X
    
    G = nx.from_numpy_matrix(X)
    pos = nx.spring_layout(G,seed=1)
    X_cp = pd.DataFrame(pos.values())
    X_cp = StandardScaler().fit_transform(X_cp)
    
    fig = plt.figure(figsize=(30,15))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    
    fn = os.path.basename(dis_matrix).split(".")[0]
    
    Dendrogram_Plot(fn,X_cp,ax1)
    clustering = Activate_DBSCAN(fn,X_cp,nlist,slist,epsilon,min_s)
    DBSCAN_Plot(fn,X_cp,clustering,ax2)
    
    fig.tight_layout()
    
    if not os.path.exists("%s_PNG"%fn):
        os.makedirs("%s_PNG"%fn)
    else:
        pass
    
    fig.savefig("%s_PNG/%s.png"%(fn,fn))
    print('  -> End clustering........')
    
    time_sp2=time()
    print '  -> Processing time to make distance matrix: '+str(time_sp2-time_sp1)+'s for '+str(len(X))+' ZIDs\n'

    return
def Scaffold_Clustering(CF_dic,finl_dic,remain_smiles):
	if not remain_smiles:
		return

	first = set()
	asmi = list(remain_smiles)[0]
	for j in remain_smiles:
		comparison_substruct(CF_dic,first,asmi,j)
	finl_dic[asmi] = list(first)
	remain_smiles = set(remain_smiles) - first
	Scaffold_Clustering(CF_dic,finl_dic,remain_smiles)
def comparison_substruct(CF_dic,tlist,x,y):
	rm1 = Chem.MolFromSmiles(CF_dic[x])
	rm2 = Chem.MolFromSmiles(CF_dic[y])
	
	gsm1 = rm1.GetSubstructMatch(rm2)
	gsm2 = rm2.GetSubstructMatch(rm1)
	
	if gsm1 and gsm2:
		tlist.add(y)
	else:
		pass

if __name__ == "__main__":


    parser=argparse.ArgumentParser()
    parser.add_argument('-infile',required=True, help='input csv file')
    #parser.add_argument('-re_mat',required=True, choices=['y','n'], help='input csv file')
    parser.add_argument("-eps",default=0.1,type=float,help="DBSCAN epsilon dtype is float")
    parser.add_argument("-min_s",default=4,type=int,help="DBSCAN minimum samples to define cluster dtype is int")
    parser.add_argument("-n_sam",default=1000,type=int,help="Number of Samples to Analysis default is 1000")
    args=parser.parse_args()

    Input_CSV = args.infile 
    epsilon = args.eps
    min_sam = args.min_s
    n_sam = args.n_sam
    re_mat='y'
    Mat_name = Input_CSV #'X'
	#scores = [np.float64(line.split(","[1]) for line in open(Input_CSV).readlines() if not line.startswith("ZID")][:n_sam]
    Main_ID_list,Main_SMI_list,Main_Score_list = Reading_Input_File(Input_CSV)
    #Main_ID_list,Main_SMI_list = Reading_Input_File(Input_CSV,id_idx,smi_idx)
    #print(Main_ID_list, Main_SMI_list)
    dis_mat = Start_Make_DisMat(Main_ID_list,Main_SMI_list,Mat_name,n_sam,re_mat)

    Cluster_DBSCAN(Mat_name,Main_ID_list[:n_sam],Main_Score_list[:n_sam],epsilon,min_sam)

    '''
	inf = sys.argv[1]
	X = np.load(inf) 
	X = np.ones((len(X),len(X))) - X

	G = nx.from_numpy_matrix(X)

	pos = nx.spring_layout(G,seed=1)

	X_cp = pd.DataFrame(pos.values())

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

    '''



