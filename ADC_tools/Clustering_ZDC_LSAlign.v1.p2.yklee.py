from os import walk
import sys, os
import string
from time import time
#import time
import datetime
from shutil import copy
import shutil
import json
import numpy as np
#from pyRMSD.utils.proteinReading import Reader
#from pyRMSD.matrixHandler import MatrixHandler
#import pyRMSD.RMSDCalculator
import subprocess
import commands
#from Bio import pairwise2
#from Bio.Seq import Seq
#from Bio.SubsMat import MatrixInfo as matlist
import multiprocessing
from functools import partial
from multiprocessing import Process, Manager
import shutil
import yaml
import operator
import pickle
import cPickle
import copy
import glob
import itertools
from sklearn.cluster import DBSCAN

import argparse

#from multiprocessing import shared_memory

# 2019.09.02 11:32
# usage prog. target_ligand_list file target_receptor_list file
# the numnber of ligand and the number of receptor shuld be same!!!


# V5: frist execute tanimoto and make filter pid list  by produced tanimoto
# usage: python Clustering_Input_ligand.DBSCAN.v3.py y|n e min_cluster
# y: rebuild the distance matrix, n: load the saved distance matrix

def Run_Babel(mylist,apair):

    #print apair
    #print apair[0], apair[1]

    tmplist1=[]

    obabel_re = subprocess.check_output(['obabel',apair[0],apair[1],'-ofpt'],stderr=subprocess.STDOUT)
    #print obabel_re
    token=obabel_re.split('\n')
    
    re=''
    for atoken in token:
        if 'Tanimoto from' in atoken:
            re=atoken
            break
    token=re.split()
    tani_on_off=float(token[5])
    #print token[5]
    tmplist1.append(apair[0])
    tmplist1.append(apair[1])
    tmplist1.append(token[5])
    #print tmplist1
    mylist.append(tmplist1)

    return


def Re_build_M(tani_m,l_list,l_dic_re):

    time_sp1=time()
    print '\nStart to making the distance matrix using multiprocessing'

    tmp_list=[]
    for pair in itertools.product(l_list, repeat=2):
        tmp_list.append(pair)

    #'''
    mylist=Manager().list()
    #print multiprocessing.cpu_count()
    n_cpu=multiprocessing.cpu_count()
    #sys.exit(1)
    #pool = multiprocessing.Pool(processes=int(sys.argv[3]))
    pool = multiprocessing.Pool(processes=(n_cpu-1))
    func=partial(Run_Babel,mylist)
    pool.map(func,tmp_list)
    pool.close()
    pool.join()
    #'''

    #print mylist

    for ele in mylist:
        #print ele,'\n'
        x,y = l_dic_re[ele[0]],l_dic_re[ele[1]]
        #print x,y 
        tani_m[x][y]=1-float(ele[2])
        tani_m[y][x]=1-float(ele[2])
    
    #print tani_m

    time_sp2=time()
    print 'Processing time to make distance matrix: '+str(time_sp2-time_sp1)+'\n\n'

    return tani_m


def Clustering_ligand_BAK():

    #python Clustering_Input_ligand.v2.py 0.1 1 

    time_sp1=time()

    l_dic={}
    l_dic_re={}
    Cut_Off=0.8

    Input_path='../Data/Input'
    l_list=glob.glob(Input_path+'/*.sdf')
    a_size=len(l_list)

    tani_m=np.zeros((len(l_list),len(l_list)))
    tani_m_1=np.zeros((len(l_list),len(l_list)))

    #shm=shared_memory.SharedMemory(create=True,size=tani_m.nbytes)
    #b=np.ndarray(tani_m.shape,dtype=a.dtype,buffer=shm.buf)

    w,h=a_size,a_size;
    #tani_m=[[0 for x in range(w)] for y in range(h)] 
    #print tani_m

    ele_count=0

    for ele in l_list:
        l_dic[ele_count]=ele
        l_dic_re[ele]=int(ele_count)
        #print ele,ele_count
        ele_count+=1

    #print l_dic_re
    C_path='./'
    np_file='Input.save.npy'


    #################################
    ### 1. Making distance Matrix ###

    re_build=sys.argv[1]
    if(re_build=='y'):
        tani_m=Re_build_M(tani_m,l_list,l_dic_re)
        np.save('./Input.save.npy',tani_m)
    else:
        path_file=os.path.join(C_path,np_file)
        if os.path.exists(path_file):
            print 'Loading the input tanimoto matrix\n'
            tani_m=np.load('./Input.save.npy')
        else:
            print 'There is no Input.save.npy file!'
            sys.exit(1)


    #print tani_m
    #sys.exit(1)

    ##########################
    ### 2. Clustering part ###
    eps=float(sys.argv[2])
    min_samples=int(sys.argv[3])

    #print 
    Clustering = DBSCAN(eps, min_samples,metric="precomputed").fit(tani_m)
    #print Clustering
    #print 
    print Clustering.labels_
    print 'The number of element :',len(Clustering.labels_)
    print 'The number of cluster :',len(set(Clustering.labels_))

    '''
    cluster_dic={}
    cluster_dic_count={}
    for x_i in range(1,w):
        tmp_list=[]
        for y_i in range(0,h):
            if x_i!=y_i:
                #print x_i,y_i
                if tani_m[x_i][y_i]>0.7:
                    l_idx=l_dic[x_i]
                    tokenx=l_idx.split('/')
                    rex=tokenx[-1].split('.')

                    l_idy=l_dic[y_i]
                    tokeny=l_idy.split('/')
                    rey=tokeny[-1].split('.')

                    if not rex[0] in cluster_dic:
                        tmp_list.append(rey[0])
                        cluster_dic[rex[0]]=tmp_list
                        cluster_dic_count[rex[0]]=1
                        #print x_i,y_i,tani_m[x_i][y_i]
                    else:
                        cluster_dic[rex[0]].append(rey[0])
                        cluster_dic_count[rex[0]]+=1
            else:
                break

    l_keys=cluster_dic.keys()
    print l_keys
    '''
    #print cluster_dic
    #print cluster_dic_count
    
    cluster_list={}
    cluster_count={}

    ##########################
    ##### for print part #####

    #print type(Clustering.labels_)
    index=0
    for ele in Clustering.labels_:
        l_id=l_dic[index]
        token=l_id.split('/')
        re=token[-1].split('.')
        #print re[0]
        #print ele
        if not ele in cluster_list:
            tmp=[]
            tmp.append(re[0])
            cluster_list[ele]=tmp
            cluster_count[ele]=1
        else:
            cluster_list[ele].append(re[0])
            cluster_count[ele]+=1
        index+=1
    cluster_count_sorted=sorted(cluster_count.items(),key=lambda x:x[1],reverse=True)

    #print cluster_list
    #print cluster_count
    #print cluster_count_sorted
    print
    for ele in cluster_count_sorted:
        print int(ele[0])+1,'\t',ele[1],'\t',cluster_list[ele[0]]
        #print 
    #print cluster_count_sorted
    
    time_sp2=time()
    print '\nProcessing the clustering: '+str(time_sp2-time_sp1)+'\n\n'


def Make_dic_input(AFile_dic,in_keys):

    Input_path='./Data/Input'
    Input_dic_count={}
    #in_keys=sys.argv[1:]
    #in_keys=ms_list
    tmp_list=[]

    #print 'Reading A file lib......'
    #AFile_dic=Read_lib('1D_lib_F.ADP.pickle')

    f_list=glob.glob(Input_path+'/*.sdf')
    #print f_list
    ln_f_list=len(f_list)

    for al in f_list:
        token=al.split('/')
        t=token[-1].split('.')
        tmp_list.append(t[0])

    f_list=tmp_list
    #print f_list
    #return 

    for akey in in_keys:
        for af in f_list:
            af_f=AFile_dic[af]['f']
            #print af, af_f
            if akey in af_f:
                if not akey in Input_dic_count:
                    Input_dic_count[akey]=1
                else:
                    Input_dic_count[akey]+=1
            else:
                pass
    
    #print Input_dic_count
    return ln_f_list,Input_dic_count



def Read_lib(lib_file):

    l_base='./Data/Lib'
    chembl_dic={}

    path_file=os.path.join(l_base,lib_file)
    if not os.path.exists(l_base):
        pass
    else:
	exists=os.path.isfile(path_file) 
	if exists:
            with open(path_file,'rb') as fp:
                chembl_dic=pickle.load(fp)
	else:
            print 'There is no Dic.picklefile'

    return chembl_dic


def Write_summary(akey,file_trace,ln_Input,Input_Dic):

    O_base='./Data/1D_Out'
    f_name_in='Summary.'+akey+'.csv'
    f_name_out='Summary.'+akey+'.with.frq.csv'
    path_file_in=os.path.join(O_base,f_name_in)
    path_file_out=os.path.join(O_base,f_name_out)

    fp_for_in=open(path_file_in,'r')
    fp_for_out=open(path_file_out,'w')

    head_in=fp_for_in.readline()
    fp_for_out.write(head_in)

    lines=fp_for_in.readlines()
    fp_for_in.close()

    #print file_trace
    #print ln_Input
    #print Input_Dic

    main_freq=Input_Dic[akey]

    for line in lines:
        #print line
        token=line.split(',')
        if token[0] in file_trace:
            key_list=file_trace[token[0]]
            for key in key_list:
                sub_freq=Input_Dic[key]
                #print line
                #print token[1],main_freq,key,sub_freq
                arg1=','.join(token[0:2])
                arg2=','+str(main_freq)+','+str(key)+','+str(sub_freq)+','
                arg3=','.join(token[2:])
                arg=arg1+arg2+arg3
                #print arg
                fp_for_out.write(arg)
                #printst='-'.join(in_keys) 
        else:
            fp_for_out.write(line)

    fp_for_out.close()


def Sub_clustering(AFile_dic,akey,cid_list):
#def Sub_clustering(akey,cid_list):

    time_sp1=time()
    Key_ln=5
    Top_rank=5

    #print 'Reading A file lib......'
    #AFile_dic=Read_lib('1D_lib_F.ADP.pickle')

    cid_list_tmp=[]
    # v2 version for files
    '''
    for acid in  cid_list:
        token=acid.split('/')
        t=token[-1].split('.')
        cid_list_tmp.append(t[0])
    '''
    #for v3 version for list
    for acid in cid_list:
        acid=acid.strip()
        token=acid.split('.')
        cid_list_tmp.append(token[0])

    cid_list=cid_list_tmp
    #print cid_list
    #return

    item=['A','D','P']
    candidate=list(map(''.join,itertools.product(item,repeat=int(Key_ln))))

    dic_count={}
    dic_id={}
    #print cid_list

    for acid in cid_list:
        feat=AFile_dic[acid]['f']
        #print acid,feat, len(feat)
        
        check_list=[]
        for c_tmp in candidate:
	    if c_tmp in feat:
	    #if feat in c_tmp:
		    #check_list.append(name)
		    if not c_tmp in dic_count:
			dic_id[c_tmp]=[acid]
			dic_count[c_tmp]=1
		    else:
			dic_id[c_tmp].append(acid)
			dic_count[c_tmp]+=1
	    else:
		pass

    dic_count_sorted=sorted(dic_count.items(),key=lambda x:x[1],reverse=True)
    #print dic_count
    #print dic_id
    #print dic_count_sorted

    s_path='./Data/1D_Out/'
    s_path_file=os.path.join(s_path,akey)
    #print s_path_file
    d_list=glob.glob(s_path_file+'/*.sdf')
    #print d_list

    refile_list=[]
    '''
    for ele in d_list:
        token=ele.split('/')
        id_token=token[-1].split('.')
        refile_list.append(id_token[0])
    #print refile_list
    '''

    # for v3 version
    refile_list=cid_list

    print 'Copying the result files to sub_cluster.....'
    l_count=0
    file_trace={}

    top5_f_list=[]
    for ele in dic_count_sorted:
        #top5_f_list.append(ele[0])
        #if ele[0]==akey:
            #continue

        #print ele[0]
        t_path=os.path.join(s_path_file,ele[0])
        #print 'Main: ',akey,'Sub_key: ',ele[0]
        #print t_path
        if not os.path.exists(t_path):
            os.mkdir(t_path)
        else:
            pass

        f_name=akey+'.'+ele[0]+'.'+'Cid_list.txt'
        t_path_file=os.path.join(t_path,f_name)
        fp_for_out=open(t_path_file,'w')

        for afile in refile_list:
            afile_f=AFile_dic[afile]['f']
            #print afile,afile_f
            if ele[0] in afile_f:
                pass
                #arg='mv '+s_path_file+'/'+afile+'.sdf'+' '+t_path
                arg='cp '+s_path_file+'/'+afile+'.sdf'+' '+t_path
                #print arg
                #os.system(arg)

                fp_for_out.write(afile+'\n')

                # making file trace dictionary
                if not afile in file_trace:
                    tmp_list=[ele[0]]
                    file_trace[afile]=tmp_list
                else:
                    file_trace[afile].append(ele[0])
            else:
                pass

        l_count+=1
        # set the top rank
        if(l_count==Top_rank):
            break
        fp_for_out.close()

    for ele in dic_count_sorted:
        top5_f_list.append(ele[0])


    #print '\nfile trace file.....' 
    #print file_trace 
    msf_list=list(set(top5_f_list+sys.argv[1:]))
    #print msf_list
    ln_Input,Input_Dic=Make_dic_input(AFile_dic,msf_list)
    Write_summary(akey,file_trace,ln_Input,Input_Dic)
    #print Input_Dic
    #key_freq=Input_Dic[akey]
    #re_f=str(key_freq)+'/'+str(ln_Input)

    #for(path, dir, files) in os.walk(each_dir):

    print             
    print('Target Key : ',akey)
    print('key length : ', Key_ln)
    print('Num of compounds : ', len(set(refile_list)))
    print('Num of Keys : ' ,len(dic_count.keys()))
    print 
    #print(dic_count_sorted)
    print('****************************************************************')

    time_sp2=time()
    print '\nProcessing the 1DScan with keys: '+str(time_sp2-time_sp1)+'\n\n'

def Sub_partition(AFile_dic):

    #print 'Reading A file lib......'
    #AFile_dic=Read_lib('1D_lib_F.ADP.pickle')

    print '\nStarting sub_clustering........'
    O_base='./Data/1D_Out'
    in_keys=sys.argv[1:]

    '''
    # for Sub clustering part
    sub_f_name='Sub_Clust.txt'
    sub_f_path=os.path.join(O_base,sub_f_name)
    fp_for_sub=open(sub_f_path,'r') 
    lines=fp_for_sub.readlines()
    sub_dic={}

    # making sub_dic for sub_clustering
    for aline in lines:
        tmp_list=[]
        aline=aline.strip()
        token=aline.split()
        ln_token=len(token)
        key=token[0]
        tmp_list=token[1:]
        sub_dic[key]=tmp_list
        #print key,tmp_list

    #print sub_dic
    '''

    for akey in in_keys:
        '''
        sub_f_path=os.path.join(O_base,akey)
        cid_list=glob.glob(sub_f_path+'/*.sdf')
        print sub_f_path
        '''

        sub_Mkey_path=os.path.join(O_base,akey)
        Mkey_file=akey+'.Cid_list.txt'
        sub_Mkey_path_file=os.path.join(sub_Mkey_path,Mkey_file)
        fp_for_in=open(sub_Mkey_path_file,'r')
        cid_lines=fp_for_in.readlines()

        '''
        print cid_list
        print cid_lines
        return
        '''

        #print akey,sub_f_path,d_list
        #print cid_list
        #if(len(cid_list)>=1):
        if(len(cid_lines)>=1):
            #Sub_clustering(AFile_dic,akey,cid_list)
            Sub_clustering(AFile_dic,akey,cid_lines)
            #Sub_clustering(akey,cid_list)
        else:
            pass

#def Cal_freq(key):
def Cal_freq(AFile_dic,key):
   
    #print 'Reading A file lib......'
    #AFile_dic=Read_lib('1D_lib_F.ADP.pickle')

    O_base='./Data/1D_Out'
    key_path=os.path.join(O_base,key)
    f_list=glob.glob(key_path+'/*.sdf')
    ln_f_list=len(f_list)

    #print key_path,f_list
    #print ln_f_list
    #return

    refile_list=[]
    for ele in f_list:
        token=ele.split('/')
        id_token=token[-1].split('.')
        refile_list.append(id_token[0])
    #print refile_list

    count=0
    for afile in refile_list:
        f_afile=AFile_dic[afile]['f']
        if key in f_afile:
            count+=1
        else:
            pass
    #print count,ln_f_list
    #return
    str_re=str(count)+'/'+str(ln_f_list)
    #print str_re
    #print count,ln_f_list
    #return count,ln_f_list
    return str_re 

def Scan_Only_Key(AFile_dic):
    
    time_sp1=time()

    Max_Copy_Num=1000
    Max_Copy_Num=10

    # Reading library filesa
    #print '\n### Starting Reading files ###\n'
    print 'Reading key lib.........'
    Key_dic=Read_lib('1D_key_lib_5.pickle')
    print 'Reading All lib.........'
    #All_dic=Read_lib('1D_lib_A.ADP.pickle')
    #print 'Reading A file lib......'
    #AFile_dic=Read_lib('1D_lib_F.ADP.pickle')
    print 'Reading Chemble lib.....'
    Chembl_dic=Read_lib('CHEMBL25-chembl_molecule.all.tsv.dic.pickle')
    #Chembl_RO5_dic=Read_lib('CHEMBL25-chembl_molecule.RO5.tsv.dic.pickle')
    print '### Ending reading Chemble lib ###\n'

    O_base='./Data/1D_Out'
    print 'Starting 1Dscan........'
    in_keys=sys.argv[1:]
    #print in_key

    # for timestamp
    ts = time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    t_token=st.split()
    str_time=t_token[0]+'_'+t_token[1]
    key_list='-'.join(in_keys)
    #fp_for_summary=open('./Data/1D_Out/Summary.'+str_time+'.'+key_list+'.txt','w')
    #head_str='Chembl_ID\tKey_ID\tMw\tAlogP\tHBA Lipinski\tHBD Lipinski\t#RO5 Violations'
    #head_str='Chembl_ID\tKey_ID\tMw\tAlogP\tHBA Lipinski\tHBD Lipinski\n'
    #fp_for_summary.write(head_str)

    #0:Mw, 1:AlogP,2:HBA Lipinski,3:Lipinski,4:RO5 Violations

    #ln_Input,Input_Dic=Make_dic_input(AFile_dic)

    #print Input_Dic
    #return

    for akey in in_keys:
        #key_freq=Input_Dic[akey]
        #re_f=str(key_freq)+'/'+str(ln_Input)
        #re_f=Cal_freq(AFile_dic,akey)
        #print re_f
        #return
        #fp_for_summary=open('./Data/1D_Out/Summary.'+akey+'.txt','w')
        #for CSV
        fp_for_summary=open('./Data/1D_Out/Summary.'+akey+'.csv','w')
        #head_str='Chembl_ID\tKey_ID\tKey_freq\tMw\tAlogP\tHBA Liipinski\tHBD Lipinski\t#RO5 Violations\n'
        #head_str='Chembl_ID,Key_ID,Key_freq,Mw,AlogP,HBA Liipinski,HBD Lipinski,#RO5 Violations\n'
        head_str='Chembl_ID,Main_Key_ID,Main_Key_freq,Sub_Key,Sub_Key_freq,Mw,AlogP,HBA Liipinski,HBD Lipinski,#RO5 Violations\n'
        #head_str='Chembl_ID\tKey_ID\tKey_freq\tMw\tAlogP\tHBA Lipinski\tHBD Lipinski\n'
        fp_for_summary.write(head_str)

        print '\nKey searching:'+akey+'.........'
        #print akey
        pid_list=[]
        pid_list2=[]
        per_list=[]
        tmp_dic={}

        pid_list=Key_dic[akey]
        for ele in pid_list:
            if ele.startswith('CH'):
                pid_list2.append(ele)
                #per_list=Chembl_dic[ele]
                tmp_dic[ele]=Chembl_dic[ele]
        pid_list=pid_list2
        #print tmp_dic
        #print len(pid_list)

        #0:Mw, 1:AlogP,2:HBA Lipinski,3:Lipinski,4:RO5 Violations

        # value is a element
        # sorted_x= sorted(myDict.items(), key=lambda e: e[1][2])

        # value is list
        # [('item2', [8, 2, 3]), ('item1', [7, 1, 9]), ('item3', [9, 3, 11])]

        #sorted_x = sorted(tmp_dic.items(), key=lambda e: e[1][4],reverse=True)
        sorted_x = sorted(tmp_dic.items(), key=lambda e: e[1][4])
        #print sorted_x
        for s_e in sorted_x:
            #print s_e[0],'\t',akey,'\t',s_e[1][0],'\t',s_e[1][1],'\t',s_e[1][2],'\t',s_e[1][3]
            #tmp_str=s_e[0]+'\t'+akey+'\t'+re_f+'\t'+s_e[1][0]+'\t'+s_e[1][1]+'\t'+s_e[1][2]+'\t'+s_e[1][3]+'\t'+s_e[1][4]+'\n'
            # For CSV
            #tmp_str=s_e[0]+','+akey+','+re_f+','+s_e[1][0]+','+s_e[1][1]+','+s_e[1][2]+','+s_e[1][3]+','+s_e[1][4]+'\n'
            tmp_str=s_e[0]+','+akey+','+s_e[1][0]+','+s_e[1][1]+','+s_e[1][2]+','+s_e[1][3]+','+s_e[1][4]+'\n'
            fp_for_summary.write(tmp_str)
            tmp_str=''

        # Making the result fold and copy result file
        # delete the previous reuslts
        '''
        C_path=os.getcwd()
        rm_path='./Data/1D_Out/'
        os.chdir(rm_path)

        listOfDir= filter(os.path.isdir, os.listdir(os.getcwd()))
        #dirs = [fi for fi in listOfFiles if fi.endswith('.txt')]
        print listOfDir
        for adir in listOfDir:
            rm_path=os.path.join(rm_path,adir)
            print rm_path
            #os.rmdir(rm_path)
            #print rm_path
            arg='rm -r '+rm_path
            print arg
            os.system(arg)
            rm_path='./Data/1D_Out/'
        
        os.chdir(C_path)
        '''

        print 'Copy the result files.....'
        S_path='./Data/Lib/sdf_lib/'
        key_path=os.path.join(O_base,akey)

        #print key_path
        if not os.path.exists(key_path):
            os.mkdir(key_path)
        else:
            arg='rm '+key_path+'/*'
            #print arg
            os.system(arg) 
            pass

        c_count=0
        Mkey_file=akey+'.Cid_list.txt'
        key_path_file=os.path.join(key_path,Mkey_file)
        #print 
        #print key_path_file
        #print 
        fp_for_Mkey_out=open(key_path_file,'w')
        for afile in pid_list:
            f_name=afile+'.sdf'
            path_file=os.path.join(S_path,f_name)
            #print path_file,key_path,akey
            arg='cp '+path_file+' '+key_path
            #os.system(arg)
            c_count+=1
            '''
            if c_count==Max_Copy_Num:
                break
            if c_count%1000==0:
                print c_count
            '''
            fp_for_Mkey_out.write(f_name+'\n')
        fp_for_Mkey_out.close()
           
    fp_for_summary.close()

    return 

    time_sp2=time()
    print '\nProcessing the 1DScan with keys: '+str(time_sp2-time_sp1)

def Add_M_weight():


    #2019.11.27
    #python prog input_file(1st col: chembl_id) lib_file(1st col:chembl_id)

    time1=time()

    chembl_dic={}
    l_base='./'
   
    time_sp1=time()

    in_f=sys.argv[1]
    lib_f=sys.argv[2]
    out_f=in_f+'.A.Mw.txt'

    fp_for_out=open(out_f,'w')
    
    fp_for_in=open(in_f,'r')
    lines=fp_for_in.readlines()
    fp_for_in.close()


    if not os.path.exists(l_base):
        pass
    else:
	# remove the existing library file
	# exists=os.path.isfile('./1D_lib_A.ADP.json') 
	exists=os.path.isfile(lib_f) 
	if exists:
            with open(lib_f,'rb') as fp:
                chembl_dic=pickle.load(fp)
	else:
            print 'There is no Dic.picklefile'
    
    l_count=0
    f_count=0
    str_tmp=''
    for line in lines:
        line=line.strip()
        token=line.split()
        #print toke
        if token[0] in chembl_dic:
            mw=chembl_dic[token[0]]
            print token[0],mw
            str_tmp=token[0]+'\t'+mw+'\t'+line+'\n'
            fp_for_out.write(str_tmp)
            f_count+=1
        else:
            pass

        l_count+=1
        '''
        if l_count==20:
            fp_for_out.close()
            return
        '''
    fp_for_out.close()
    print l_count,f_count

    time_sp2=time()
    print 'Write and copying time : '+str(time_sp2-time_sp1)

    return


def Cal_M_weight():


    #2019.11.27
    #python prog input_file(1st col: chembl_id)

    #print 'start_PHscan_1D'
    time1=time()

    chembl_dic={}

    l_base='./'
   
    time_sp1=time()

    fp_for_out=open('result.txt','w')

    if not os.path.exists(l_base):
        pass
    else:
	# remove the existing library file
	# exists=os.path.isfile('./1D_lib_A.ADP.json') 
	exists=os.path.isfile('./Chembl_Dic.pickle') 
	if exists:
            with open('Chembl_Dic.pickle','rb') as fp:
                chembl_dic=pickle.load(fp)
	else:
            print 'There is no Chembl_Dic.picklefile'
    
    fp_for_in=open(sys.argv[1],'r')
    lines=fp_for_in.readlines()
    fp_for_in.close()

    l_count=0
    f_count=0
    str_tmp=''
    for line in lines:
        line=line.strip()
        token=line.split()
        #print toke
        if token[0] in chembl_dic:
            mw=chembl_dic[token[0]]
            print token[0],mw
            str_tmp=token[0]+'\t'+mw+'\n'
            fp_for_out.write(str_tmp)
            f_count+=1
        else:
            pass

        l_count+=1
        '''
        if l_count==20:
            fp_for_out.close()
            return
        '''
    fp_for_out.close()
    print l_count,f_count

    time_sp2=time()
    print 'Write and copying time : '+str(time_sp2-time_sp1)

    return

def anal_1DScan():

    #mylist = Manager().list()
    #mylist.append(1)
    #print mylist
    #sys.exit(1)

    head_tmp='Bin,Num.of.target,Num.of.C\n'
    #for s_weight in np.arange(1,1,0.1): 
    for s_weight in np.arange(1,2,1): 
        #print '\nparameter: '+str(s_weight)
        PHscan_1D(s_weight)
        #anal_result_files()
        '''
        #all_bin_t_ac_avg,all_bin_nt_ac_avg=anal_result_files()

        #print len(all_bin_t_ac_avg), all_bin_t_ac_avg
        tmp_f='All_Graph_P'+str(s_weight)+'.csv'
        fp_for_out=open(tmp_f,'w')
        e_index=0
        tmp_str=head_tmp
        for ele in all_bin_t_ac_avg:
            tmp_str=tmp_str+str(e_index)+','+str(ele)+','+str(all_bin_nt_ac_avg[e_index])+'\n'
            e_index+=1
        fp_for_out.write(tmp_str)
        fp_for_out.close()
        '''

# add two dictionary and return result dictionary
def sum_dic(dic_score,dic_score_r):

    dic_result={}
    dic_score_id_f=dic_score.keys()
    #print dic_score_id_f
    score_f=0
    score_r=0
    score_s=0

    for ele in dic_score_id_f:
        score_f=dic_score[ele]
        if ele in dic_score_r:
            score_r=dic_score_r[ele]
        else:
            pass
        score_s=score_f+score_r
        dic_result[ele]=score_s
        #print score_f,score_r
        #print

    return dic_result

def anal_result_files():
  
    print 'analyzing the result.......'
    c_path=os.getcwd()
    #print c_path
    t_path='/data/1D_Out/'
    t_path=c_path+t_path
    #print t_path
    #sys.exit(1)
    os.chdir(t_path)
    #print os.getcwd()
    #sys.exit(1)

    increment=1
    bin_size=100/increment
    bin_size=int(bin_size)
    #print bin_size
    #sys.exit(1)

    # select original Complex
    listOfFiles= filter(os.path.isfile, os.listdir(os.getcwd()))
    files = [fi for fi in listOfFiles if fi.endswith('.txt')]
    ln_files=len(files)
    #print ln_files
    #print files
    #sys.exit(1)
    #os.chdir(c_path)

    all_bin_t=[0]*bin_size
    all_bin_nt=[0]*bin_size

    # for pdb 1st rank
    pdbs_rank=[]*ln_files

    # Max rank2 Count
    rank2_count=5000
    pdbs_rank2=[0]*rank2_count

    rank_target_count=[0]*rank2_count
    rank_ntarget_count=[0]*rank2_count

    file_count=1
    for a_file in files:

        # for 5% increament
        k=[0]*bin_size
        nk=[0]*bin_size
        index=[0]*bin_size
        apdb_rank=[]

        # Max rank is 100
        pdb_rank=[0]*rank2_count

        # tmp ------
        rank_target_t=[0]*rank2_count
        rank_ntarget_t=[0]*rank2_count
        # ----------

        fp_for_in=open(a_file,'r')
        lines=fp_for_in.readlines()
        ln_afile=len(lines)
        #print ln_afile
        
        # Make_index
        i=increment
        a_index=0
        while(i<101):            
            r_index=(float(i)/float(100))*ln_afile
            r_index=int(r_index)
            #print i,r_index
            index[a_index]=r_index

            a_index=a_index+1
            i=i+increment
            
        #print 'processing pdb:'+a_file
        #print index
        #print 
        t_ligand= 'l_'+a_file[:-4]
        #print t_ligand


        rank_target=[]
        count_t=0

        rank_ntarget=[]
        count_nt=0

        ln_lines=len(lines)
        #print ln_lines
        
        # Filling the count array
        l_count=0
        rank_index=0
        tmp_t=[]
        tmp_nt=[]
        check_flag=0

        while(l_count<ln_lines-1):

            ctoken=lines[l_count].split('\t')   # parsing current line
            ntoken=lines[l_count+1].split('\t') # parsing next line

            #print l_count,lines[l_count]

            if(ctoken[1]==ntoken[1]):
                # the line is about kinase
                if ctoken[0].startswith('l_'):
                    if not ctoken[0] in tmp_t:
                        tmp_t.append(ctoken[0])
                        count_t+=1
                    # for check the t_ligand rank
                    if (ctoken[0]==t_ligand and check_flag==0):
                        #print rank_index
                        pdb_rank[rank_index]+=1
                        check_flag=1
                        
                # the line is about CJ
                if not ctoken[0].startswith('l_'):
                    if not ctoken[0] in tmp_nt:
                        tmp_nt.append(ctoken[0])
                        count_nt+=1
            if(ctoken[1]!=ntoken[1] or l_count==ln_lines-2):
                # the line is about kinase
                if ctoken[0].startswith('l_'):
                    tmp_t.append(ctoken[0])
                    count_t+=1

                    # for check the t_ligand rank
                    if (ctoken[0]==t_ligand and check_flag==0):
                        #print rank_index
                        pdb_rank[rank_index]+=1
                        check_flag=1

                rank_target.append(tmp_t)
                #rank_target_count.append(count_t)
                #print rank_index
                rank_target_t[rank_index]=count_t
                rank_target_count[rank_index]=rank_target_count[rank_index]+count_t
                tmp_t=[]
                count_t=0

                # the line is about CJ
                if not ctoken[0].startswith('l_'):
                    tmp_nt.append(ctoken[0])
                    count_nt+=1
                rank_ntarget.append(tmp_nt)
                #rank_ntarget_count.append(count_nt)
                rank_ntarget_t[rank_index]=count_nt
                rank_ntarget_count[rank_index]=rank_ntarget_count[rank_index]+count_nt
                tmp_nt=[]
                count_nt=0
                
                rank_index+=1

            '''
            if(l_count==ln_lines-2):
                if (check_flag==0):
                    print 'No find PDB'+a_file
            '''

            l_count+=1

        #print  str(file_count)+' '+a_file
        file_count+=1
        #print  pdb_rank
        pdbs_rank2=[sum(x) for x in zip(pdbs_rank2,pdb_rank)] 
        #print pdbs_rank2
        #print  rank_target
        #print  rank_ntarget
        '''
        print rank_target_t
        print rank_ntarget_t
        print 
        print rank_target_count
        print rank_ntarget_count
        print 
        '''

    #print 
    #print pdbs_rank2
    #print sum(pdbs_rank2) 
    #return float("{0:.3f}".format(score))
    #rank_target_count=[float("{0:.0f}".format(x/ln_files)) for x in rank_target_count] 
    #print rank_target_count
    rank_target_count=[int(x/742) for x in rank_target_count] 
    #print rank_target_count
    #print
    #print rank_ntarget_count
    #rank_ntarget_count=[int(x/742) for x in rank_ntarget_count] 
    #rank_ntarget_count=[int(x/ln_files) for x in rank_ntarget_count] 
    #print rank_ntarget_count
    #return 
    
    '''
        # Filling the count array
        l_count=0
        for line in lines:
            token=line.split('\t')
            #print token
            # the ligand is kinase
            if token[0].startswith('l_'):
                #print token[0]
                #check toke[0] is receptor's ligand
                if token[0]==t_ligand:
                    #print l_count,t_ligand
                    apdb_rank.append(t_ligand)
                    apdb_rank.append(l_count)
                    #sys.exit(1)

                a_index=0
                for ele in index:
                    if l_count<ele:
                        k[a_index]=k[a_index]+1
                        break
                    a_index+=1
            # the ligand is CJ
            else:
                a_index=0
                for ele in index:
                    if l_count<ele:
                        nk[a_index]=nk[a_index]+1
                        break
                    a_index+=1
            #sys.exit(1)a
            l_count+=1

        #print k
        #print 
        #print nk
        #print '\n'

        pdbs_rank.append(apdb_rank)

        # add the result to global variable
        all_bin_t=[sum(x) for x in zip(all_bin_t,k)] 
        all_bin_nt=[sum(x) for x in zip(all_bin_nt,nk)]


    # accmulate the result to the *_ac array
    index=0
    all_bin_t_ac=all_bin_t[:]
    all_bin_nt_ac=all_bin_nt[:]
    while(index<bin_size-1):
        all_bin_t_ac[index+1]=all_bin_t_ac[index]+all_bin_t_ac[index+1]
        all_bin_nt_ac[index+1]=all_bin_nt_ac[index]+all_bin_nt_ac[index+1]
        index+=1

    # average per bin
    all_bin_t_avg=[int(x/ln_files) for x in all_bin_t] 
    all_bin_nt_avg=[int(x/ln_files) for x in all_bin_nt]

    # average acculate
    all_bin_t_ac_avg=[int(x/ln_files) for x in all_bin_t_ac] 
    all_bin_nt_ac_avg=[int(x/ln_files) for x in all_bin_nt_ac]
    

    print 
    print '----- Average bin per bin-----'
    print 'Kinase bin:'
    print all_bin_t_avg
    print 'CJ bin:'
    print all_bin_nt_avg
    print 


    print '----- Average bin acculation -----'
    print 'Kinase bin:'
    print all_bin_t_ac_avg
    print 'CJ bin:'
    print all_bin_nt_ac_avg
    print 

    print 
    print '------ PDB ligand rank -------'
    print pdbs_rank


    total_2count=0
    total_1count=0
    dir_toward=0
    dir_reverse=0
    dir_all=0

    for eles in pdbs_rank:
        ln_eles=len(eles)
        #print ln_eles,eles
        if ln_eles==4:
            total_2count=total_2count+2
            #print eles[1],eles[3]
            #print eles[1],
            dir_toward=dir_toward+int(eles[1])
            dir_reverse=dir_toward+int(eles[3])
            dir_all=dir_all+int(eles[1])+int(eles[3])
        if ln_eles==2:
            total_1count=total_1count+1   
            dir_toward=dir_toward+int(eles[1])
            dir_all=dir_all+int(eles[1])
            #print str(eles[1])

    pid_avg_rank=float(dir_all)/float((total_2count+total_1count))
    pid_avg_to_rank=float(dir_toward)/float(((total_2count/2)+total_1count))
    pid_avg_re_rank=float(dir_reverse)/float((total_2count/2))

    print '\nPDB average rank: '+str(int(pid_avg_rank))
    print 'PDB average toward rank: '+str(int(pid_avg_to_rank))
    print 'PDB average reverse rank: '+str(int(pid_avg_re_rank))
    print '\n\n'
    ''' 
    os.chdir(c_path)
    print c_path

    #return all_bin_t_ac_avg, all_bin_nt_ac_avg 
    return


def check_l_r_pdb_list(q_type,pdb_list):
    
    #print 'start_check_l_r_pdb_list'
    tmp_list=[]
    #print pdb_list

    for ele in pdb_list:
        #print ele
        tmp_type=check_l_r_apdb(ele)

        # check if there is info in PHscan_lib1
        if tmp_type!='E':
            #print q_type, tmp_type,
            # select ohter type L->R, R->L
            if tmp_type!=q_type:
                #print ele,tmp_type, q_type,
                tmp_list.append(ele)
            else:
                pass
        else:
            print 'There is no file in PHscan_lib1'
            #sys.exit(1)
            return 'E'

    #print tmp_list
    #print 'end_check_l_r_pdb_list'
    return tmp_list


# Scoring the each libray 
def check_l_r_apdb(a_pdb):
  
    #print 'start_check_l_r_apdb'
    a_pdb=a_pdb.strip()
    c_path=os.getcwd()
    #print c_path
    l_base='/data/PHscan_Lib1/'
    path=''
    for a in a_pdb:
        path=path+a+'/'
        #print a
    #print path
    path=c_path+l_base+path[:-1]
    #print path

    if os.path.exists(path):
        os.chdir(path)
    else:
        print a_pdb
        print 'There is no file in PHscan_lib1 : check_l_r_apdb'
        #sys.exit(1)
        return 'E'

    # select original Complex
    listOfFiles= filter(os.path.isfile, os.listdir(os.getcwd()))
    files = [fi for fi in listOfFiles if fi.endswith('.pdb')]
    #print os.listdir(os.getcwd())
    #print a_pdb,files
    os.chdir(c_path)
    #sys.exit(1)

    #print 'end_check_l_r_apdb'
    # a_pdb is ligand if there is no pdb files
    if(len(files)==0):
        return 'L'
    else:
        return 'R'


# Scoring the each libray 
def sub_scoring(s_weight,target_f,lib_f,ele,a_pdb,pos):
    
    #print 'start_sub_scoring'
    #print ele, a_pdb, pos
    #print target_f
    #print lib_f

    #print 'in sub_scoring'
    #print s_weight
    #sys.exit(1)

    ln_target_f=len(pos)
    #print ln_target_f
    zero_count=pos.count(0)
    #print ln_target_f-zero_count
    sim=float(ln_target_f-zero_count)/float(ln_target_f)
    #print weight
    sum_f=sum(pos)
    #print sum_f
    score=s_weight*sim*sum_f
    #print 'score: '+str(score)
    #return score
    #print 'end_sub_scoring'
    return float("{0:.3f}".format(score))



def sub_scoring_2(weight,target_f,lib_f,ele,a_pdb,pos):
    
    #print 'start_sub_scoring'
    #print ele, a_pdb, pos
    #print target_f
    #print lib_f

    #print 'in sub_scoring'
    #print s_weight
    #sys.exit(1)

    ln_target_f=len(pos)
    Max_target_f_s=ln_target_f*3
    zero_count=pos.count(0)

    s_weight=0.5-weight
    f_weight=0.5+weight


    s_weight=weight
    f_weight=1-weight

    sim=float(ln_target_f-zero_count)/float(ln_target_f)
    #print weight
    sum_f=sum(pos)
    #print sum_f
    score=float(s_weight*sim)+(float(sum_f)/float(Max_target_f_s))*f_weight
    #print 'score: '+str(score)
    #return score
    #print 'end_sub_scoring'
    return float("{0:.3f}".format(score))


def sub_scoring_3(weight,target_f,lib_f,ele,a_pdb,pos,pos_l):
    
    #print 'start_sub_scoring'
    #print ele, a_pdb, pos,target_f,lib_f
    #print target_f
    #print lib_f

    #print 'in sub_scoring'
    #print s_weight
    #sys.exit(1)

    matrix = matlist.blosum62
    gap_open = -1
    gap_extend = -0.1

    alignments = pairwise2.align.localds(target_f,lib_f,matrix,gap_open, gap_extend)
    top_aln=alignments[0]
    m_pdb_x, a_pdb_y, score, begin, end = top_aln
    #print 'Local'
    print ele+'\n'+a_pdb+'\n'+m_pdb_x+'\n'+a_pdb_y+'\n'+str(score)
    print pos
    print pos_l
    print
    #print str(score)+'\n'

    #print
    #print pos
    pos.append(0)
    pos.append(0)
    #print pos

    ln_target=len(pos)
    #print ln_target
    pos_fill=[0]*(ln_target)
    total_f=0

    #for index in range(0,5):
    for index in range(0,ln_target-1):
        #------------------------
        # for pos_fill cacualtion
        if(pos[index]!=0 and pos[index+1]!=0):
            pos_fill[index]=1

            num_f=pos[index]
            total_f=total_f+3-2+(num_f-1)*3

        if(pos[index]!=0 and pos[index+1]==0 and pos[index+2]!=0):
            pos_fill[index]=1
            pos_fill[index+1]=1

            num_f=pos[index]
            total_f=total_f+3-1+(num_f-1)*3

        if(pos[index]!=0 and pos[index+1]==0 and pos[index+2]==0):
            pos_fill[index]=1
            pos_fill[index+1]=1
            pos_fill[index+2]=1

            num_f=pos[index]
            total_f=total_f+3+(num_f-1)*3
    '''
    print 'pos_fill'
    print pos_fill
    print 'total f'
    print total_f
    print 
    print
    '''
    sum_pos=sum(pos_fill)
   
    #return sum_pos
    #return score*sum_pos #standard function
    return score*sum_pos*total_f
    #return float("{0:.3f}".format(score))



def sub_scoring_4(weight,target_f,lib_f,ele,a_pdb,pos,pos_l):
    
    #print 'start_sub_scoring'
    #print ele, a_pdb, pos,target_f,lib_f
    #print target_f
    #print lib_f

    #print 'in sub_scoring'
    #print s_weight
    #sys.exit(1)

    ##################################################################################
    ############################# alignment part #####################################
    matrix = matlist.blosum62
    gap_open = -0.5
    gap_extend = -0.5

    # forword alignment
    #alignments = pairwise2.align.localds(target_f,lib_f,matrix,gap_open, gap_extend)
    #alignments = pairwise2.align.globalds(target_f,lib_f,matrix,gap_open, gap_extend)
    #top_aln=alignments[0]

    '''
    # reverse alignment
    target_f_r=target_f[::-1]
    alignments = pairwise2.align.localds(target_f_r,lib_f,matrix,gap_open, gap_extend)
    top_aln_r=alignments[0]
    
    toalyy_aln=top_aln+top_aln_r
    ''' 

    #m_pdb_x, a_pdb_y, al_score, begin, end = top_aln
    #print 'Local'
    #print ele+'\n'+a_pdb+'\n'+m_pdb_x+'\n'+a_pdb_y+'\n'+'align score:'+str(al_score)
    #print pos
    #print pos_l
    #print
    #print str(score)+'\n'
    #ln_match=len(pos)
    ########################## end of alignment ######################################

    #print
    #print pos
    pos.append(0)
    pos.append(0)
    #print pos
    ln_match=len(pos)

    ln_target=len(pos)
    ln_ligand=len(pos_l)
    #print ln_target
    pos_fill=[0]*(ln_target)
    pos_l_fill=[0]*(ln_ligand)
    total_f=0
    total_l=0

    #print '################\n\n'
    #print pos
    #print ln_target
    #print pos_l
    #print ln_ligand

    #fill_count_f=ln_target-pos.count(0)
    #fill_count_l=ln_ligand-pos_l.count(0)
    
    fill_count_f=sum(pos) 
    fill_count_l=sum(pos_l)

    #print
    #print fill_count_f
    #print fill_count_l

    #if(fill_count_f+2==ln_target):
    #    print 'Full match'

    ##################################################################
    # for caculation '1,0,0' type in position list
    # for caculation '1,0,1' type in position list
    #################################################################
    t_end_type_count=0
    for index in range(0,ln_target-2):
        #print index,':',pos[index],pos[index+1],pos[index+2]
        if(pos[index]==1 and pos[index+1]==0 and pos[index+2]==0):
            t_end_type_count+=2
        if(pos[index]==1 and pos[index+1]==0 and pos[index+2]==1):
            t_end_type_count+=1
            #print 'find'
    #print 'end:',t_end_type_count

    #print 'ligand_part'
    l_end_type_count=0
    repeat_tar=0
    for index in range(0,ln_ligand-2):
        #print index,':',pos_l[index],pos_l[index+1],pos_l[index+2]
        if(pos_l[index]==1 and pos_l[index+1]==0 and pos_l[index+2]==0):
            l_end_type_count+=2
        if(pos_l[index]==1 and pos_l[index+1]==0 and pos_l[index+2]==1):
            l_end_type_count+=1
            #print 'find'
        #print target_f
        #print lib_f
     
        #print 'index;',index
        if(len(target_f)<=len(lib_f)):
            r_index=index
            m_flag=0
            if(r_index+ln_match<=ln_ligand):
                for e_index in range(0,ln_target):
                    #print r_index,e_index,ln_target,ln_match
                    if(target_f[e_index]==lib_f[r_index]):
                        m_flag=1
                    else:
                        m_flag=0
                        break
                    r_index+=1
                if m_flag==1:
                    repeat_tar+=1
        else:
            pass
            #print 'target is longer than ligand'
                
    #print 'end:',l_end_type_count
    ##################### end of pattern search ####################
    #print 'match_ln:',ln_match
    #print 'num. of. repeat:',repeat_tar

    w1=1
    w2=1
    #repeat_tar=1

    fill_count_f=fill_count_f+t_end_type_count
    fill_count_l=fill_count_l+l_end_type_count

    #print '################\n\n'
    #print fill_count_f
    #print fill_count_l
    #print repeat_tar

    #score=(w1*(float(fill_count_f)/float(ln_target)))+((w2+repeat_tar)*(float(fill_count_l)/float(ln_ligand)))
    #score=w1*(float(fill_count_f))+(w2+repeat_tar)*(float(fill_count_l))
    score=w1*(float(fill_count_f))+(w2)*(float(fill_count_l))
    #score=w1*(float(fill_count_f))
    #print 'feature score:',score
    #print '#############\n'

   
    #return sum_pos
    #return score*sum_pos #standard function
    #return score*sum_pos*total_f
    #return score*al_score
    return score
    #return float("{0:.3f}".format(score))


# savee the reuslt of 1Dscan to disk
def save_1D_ranking_result(OneD_lib_F,sorted_x,dic_pos,target_f,ele):
    
    #print 'start_save_1D_ranking_result'
    # make 1D result  folder
    l_base='./data/1D_Out'
    if not os.path.exists(l_base):
        os.makedirs(l_base)
    else:
        arg='rm '+l_base+'/*'
        #os.system(arg)

    #print '\n' 
    #print sorted_x
    #print dic_pos
    #print target_f
    #print ele

    #sys.exit(1)

    # For processing a ranked a library stored in sorted_x which composed of two elements
    for_out={}
    for pair in sorted_x:
        #print ele,pair[0] # pair[0] is pid
        a_pos=dic_pos[pair[0]] # a_pos is position list for pdb
        #print a_pos

        index_pos=0

        # for feature_set a ranked pdb
        feature_set=[]

        # the postion of a ranked pdb
        for pos_index in a_pos:
            if pos_index!=0:
                #print index_pos, pos_index
                #print target_f,target_f[index_pos:index_pos+3]
                l_pid=OneD_lib_F[pair[0]]
                #print l_pid
                l_pid_f_indexs= l_pid[target_f[index_pos:index_pos+3]]
                #print l_pid_f_indexs
                for a_index in l_pid_f_indexs:
                    #print a_index
                    feature_set.append(a_index)
                #print 
            index_pos+=1
        #print feature_set
        #print 
        for_out[pair[0]]=feature_set
        feature_set=[]

        cwd=os.getcwd()
        path=cwd+'/'+l_base[2:]

        #print for_out
        #sys.exit(1)

        # print ele,path
        #with open(path+'/'+ele+'.1DRe.pickle','wb') as fp:
	    #pickle.dump(for_out,fp,protocol=pickle.HIGHEST_PROTOCOL)
	    #pickle.dump(for_out,fp,protocol=0)
	    #cPickle.dump(for_out,fp)

        # for test
        #with open(path+'/'+ele+'.1DRe.json','w') as fp:
	    #json.dump(sorted_x,fp)
    #print  for_out

    #print 'end_save_1D_ranking_result'
    return

def filtering_list_with_key(tmp_list,OneD_lib_F,f_key_list):
 
    tmp_list_t=[]

    #print 'befor filtering: ',len(tmp_list)
    #print f_key_list
    for akey in f_key_list:
        for apid in tmp_list:
            afe=OneD_lib_F[apid]['f']
            #print afe
            if akey in afe:
                tmp_list_t.append(apid)
            else:
                pass
    #print tmp_list
    tmp_list=tmp_list_t
    #print 'after filtering: ',len(tmp_list),'\n'

    return tmp_list


def filtering_list_with_tani(tmp_list,pid_list_tani):
 
    tmp_list_t=[]
    #print tmp_list
    #print pid_list_tani
    #print 'befor filtering: ',len(tmp_list)
    #print f_key_list
    for apid in tmp_list:
        if apid in pid_list_tani:
            tmp_list_t.append(apid)
        else:
            pass
    #print tmp_list
    tmp_list=tmp_list_t
    #print tmp_list_t
    #print 'after filtering: ',len(tmp_list),'\n'

    return tmp_list
    

def Make_pdb_list(q_type,t_feature,OneD_lib_A,OneD_lib_F,f_key_list,pid_list_tani):
    
    #print 'start_Make_pdb_list'
    # Pass 1
    # Make pdb list for a INPUT
    pdb_list=[]
    ln_ADP=len(t_feature)
    #print ln_ADP
    l_index=0
    end_of_ADP=ln_ADP-3

    while(l_index<=end_of_ADP):
        #print l_index
        #print t_feature[l_index:l_index+3]
        tmp_list=OneD_lib_A[t_feature[l_index:l_index+3]]
        #print tmp_list
        print 'Original: ',len(tmp_list)
        tmp_list=filtering_list_with_key(tmp_list,OneD_lib_F,f_key_list)
        tmp_list_b=tmp_list
        print 'Filtering with Key: ',len(tmp_list)

        # V52: remove pids with pid_list_tani
        tmp_list=filtering_list_with_tani(tmp_list,pid_list_tani)
        print 'Filtering with tani: ',len(tmp_list)
        #print tmp_list

        #'''
        if(len(tmp_list)==0):
            print 'By tanimoto, remaining 0 pid'
            tmp_list=tmp_list_b
        #'''
        # select the oppsite type: if a query is ligand, select the receptors
        # select the oppsite type: if a query is receptor, select the ligands
        #print tmp_list
        tmp_list= check_l_r_pdb_list(q_type,tmp_list)
        #print tmp_list
        #print len(tmp_list)
        #print f_key_list
        #print q_type
        #sys.exit(1)

        if (len(tmp_list)==0 or tmp_list=='E' ):
            print 'No pdb_list for some feature!'
            #sys.exit(1)
            return 'E'

        pdb_list.append(tmp_list)
            
        l_index+=1

    # V52: remove pids with pid_list_tani
    #tmp_list=filtering_list_with_tani(tmp_list,OneD_lib_F,pid_list_tani)

    #print 'end_Make_pdb_list'
    #print pdb_list
    #print len(pdb_list)
    return pdb_list


def Make_dic_score_pos(s_weight,pdb_list,OneD_lib_F,target_f,ele,dic_tani):

    #print 'start_Make_dic_score_pos'
    #Pass2

    dic_score={}
    dic_pos={}

    tmp_all=[]
    for tmp_pdb_list in pdb_list:
        tmp_pdb_list=list(set(tmp_pdb_list))
        tmp_all=tmp_all+tmp_pdb_list
    #print tmp_all
    pdb_set_list=tmp_all
    '''
    print 'pdb_set_list'
    print pdb_set_list  #[a,,b,c,d,f,e] the set for pdb_list
    print 
    print 'pbd_list'    #[[a,b,c],[a,b,d],[e,f]] the pdbs set for a receptor index
    print pdb_list

    return
    '''

    #print pdb_list
    #print dic_tani
    #return


    # print 
    # Making count arrary for a candidate pdb
    # Check a candidate pdb 

    # for caculation of ligand pos
    lib_target=OneD_lib_F[ele]
    lib_target_f=lib_target['f']

    al_pos_count=0
    al_pos_l_count=0

    for a_pdb in pdb_set_list:
        pos=[] 
        total_pos_count=0
        total_pos_l_count=0
        # check a each index for cadidate pdb
        #####################################
        for pdbs in pdb_list:
            #print ele, a_pdb, pdbs
            num_of_pdb=pdbs.count(a_pdb)
            total_pos_count=total_pos_count+num_of_pdb
            pos.append(num_of_pdb)
            #print num_of_pdb
        #print ele, pos, a_pdb
        #return

        # for position array of ligand
        lib_ligand=OneD_lib_F[a_pdb]
        lib_ligand_f=lib_ligand['f']
        #print ele,':',lib_target_f,a_pdb,':',lib_ligand_f
        ln_ligand_f=len(lib_ligand_f)

        pos_l=[0]*ln_ligand_f

        #print lib_ligand_f
        for index in range(0,ln_ligand_f-2):
            feature_3s=lib_ligand_f[index:index+3]
            if feature_3s in lib_target_f:
                pos_l[index]=pos_l[index]+1
                total_pos_l_count+=1
            else:
                pass 
        #return
        #print pos_l
        #print ele,':',lib_target_f,pos,a_pdb,':',lib_ligand_f,pos_l
        #print total_pos_count,total_pos_l_count

        if(total_pos_count<=1): 
            al_pos_count+=1
            #print 'pos: ',ele,a_pdb
            #print 'pos: ',ele,':',lib_target_f,pos,a_pdb,':',lib_ligand_f,pos_l
            continue
        if(total_pos_l_count<=1):
            al_pos_l_count+=1
            #print 'pos_l: ',ele,a_pdb
            #print 'pos_l: ',ele,':',lib_target_f,pos,a_pdb,':',lib_ligand_f,pos_l
            continue

        #print a_pdb,dic_tani
        #raw_input("Press Enter to continue...")
        #######################################
        # Tanimoto part
        #######################################
        if a_pdb in dic_tani:
            tani_score=dic_tani[a_pdb]
            #print tani_score
            #raw_input("Press Enter to continue...")
        else:
            tani_score=1
        ######################################

        lib=OneD_lib_F[a_pdb]
        lib_f=lib['f']
        #print lib
        ln_lib=len(lib_f)

        # Sort dictionary_1
        # Import operator
        # x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
        # sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)

        # Sort dictionary_2
        # x=sorted(d.items(), key=lambda x: x[1],reverse=True)
        score=sub_scoring_4(s_weight,target_f,lib_f,ele,a_pdb,pos,pos_l)

        # In case of adoption of tanimoto
        score=score*tani_score
        #score=tani_score
        #score=score

        #score=sub_scoring_2(s_weight,target_f,lib_f,ele,a_pdb,pos)
        #score=sub_scoring(s_weight,target_f,lib_f,ele,a_pdb,pos)
        dic_score[a_pdb]=score
        dic_pos[a_pdb]=pos
        #print ele, a_pdb, pos
        #return
        #print target['f']
        #print lib['f']
        #print ''
        #print dic_scorea

    #print 'end_Make_dic_score_pos'
    #print al_pos_count,al_pos_l_count
    return dic_score, dic_pos


def takeSecond(elem):
    return elem[1]


def make_Tanimoto_dic(ob_lines):

    dic_tani={}

    #print ob_lines

    for ob_line in ob_lines:
        #print ob_line
        #print len(ob_line)
        if(len(ob_line)>0):
            # skip the comment: 'Possible superstructure of....'
            if(ob_line[0]=='>'):
                token=ob_line.split()
                # skip the frist line
                if(len(token)>2):
                    pass
                    #print token[0][1:],token[5],token
                    dic_tani[token[0][1:]]=float(token[5])*10
                else:
                    pass
            else:
                pass

    #print dic_tani
    return dic_tani


def split_line(line):
    
    line=line.strip()
    #tmp_token=line.split('\t')
    tmp_token=line.split()
    #print tmp_token
    r_value=[]

    #print tmp_token,len(tmp_token)

    tpid=tmp_token[0]
    if len(tmp_token)>=2:
        f_key=tmp_token[1:]
        r_value.append(tpid)
        r_value.append(f_key)
        return r_value
        
    if len(tmp_token)==1:
        r_value.append(tpid)
        return r_value

def copy_candidate(ligands,tpid,t_key):

    # set the max
    Max_candidate=10
    #Max_candidate=1000

    l_path='./data/PHscan_Lib1/'
    Out_path='./data/1D_Out/'

    t_path=Out_path+tpid+'/'+t_key
    #print t_path
    path=''
    tmp_path=''


    # clean the directory
    d_list=glob.glob(t_path+'/*.sdf')
    for x in d_list:
        os.remove(x) 

    ligands=ligands.strip()
    ligands=ligands.split('\n')
    #print ligands

    candidate_count=0
    for ligand in ligands:
        token=ligand.split()
        for a in token[0]:
            path=path+a+'/'
        tmp_path=l_path+path

        arg='cp '+tmp_path+token[0]+'.sdf.zip '+Out_path+tpid+'/'+t_key
        #print arg
        os.system(arg)

        FNULL = open(os.devnull, 'w')
        #zp = subprocess.call(['7z', 'e', '-p=swshin', '-y', Out_path+tpid+'/'+t_key+'/*.zip','-o'+Out_path+tpid+'/'+t_key],stdout=FNULL,stderr=subprocess.STDOUT)
        zp = subprocess.call(['7z', 'e', '-p=swshin', '-y', t_path+'/*.zip','-o'+t_path],stdout=FNULL,stderr=subprocess.STDOUT)
        
        tmp_path=''
        path=''

        # return only max 1000 candidate
        candidate_count+=1
        if(candidate_count==Max_candidate):
            # clean the directory
            d_list=glob.glob(t_path+'/*.zip')
            for x in d_list:
                os.remove(x) 
            return

    # clean the directory
    d_list=glob.glob(t_path+'/*.zip')
    for x in d_list:
        os.remove(x) 

    return



def tani_ids(ob_lines):

    tmp=[]
    for line in ob_lines:
        token=line.split()
        #print token[0][1:]
        tmp.append(token[0][1:])
    return tmp


def core_f(s_weight,OneD_lib_A,OneD_lib_F,Key_dic_list,line):

    #print 'start_core_f'

    # Check time a pdb target
    start_time=time()

    print '\nstarting core_f...'
    ts = time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print st


    # Check a type: ligand or receptor
    # if a query type is 'L', search receptors
    # if a query type is 'R', search ligands

    time_sp1=time()
    #print line
    #tpid,f_key=split_line(line)
    r_value=split_line(line)
    f_flag=0
    tpid=''
    f_key_list=[]
    #print Key_dic_list
    #print r_value
    #return

    if(len(r_value)==2):
        tpid=r_value[0]
        f_key_list=r_value[1]
        #print f_key_list
        ln_key_list=len(f_key_list)
        f_flag=1

    if(len(r_value)==1):
        tpid=r_value[0]

    print '\nProcessing: '+tpid
    # key element and key length
    # if pattern is used
    if(f_flag):
        if(check_key(f_key_list)==0):
            print '\n\n'
            print f_key_list
            print '\nInvalid key length. Key length must be within 4 and 6.\n'
            print 'Or invalid key character. Key must be consisted of three letter:\'A\' or \'D\' or \'P\''
            print 'Try agin!\n\n'
            return

    #print Key_dic_list[0]
    #return
    
    q_type=check_l_r_apdb(tpid)
    # L or R
    #print q_type
    if(q_type=='E'):
        print 'there is no info in PHscan_lib1 for query:'+tpid
        #s.exit(1)
        return 'E'
    #sys.exit(1)
    #print 'Processing..... target PDB :'+ele
    time_sp2=time()
    print tpid+': '+'Check key time : '+str(time_sp2-time_sp1)

    pdb_list=[]
    pdb_list_r=[]
    dic_score={}
    dic_pos={}
    dic_score_r={}
    dic_pos_r={}
    dic_tani={}
    pid_list_tani=[]

    #print ele

    #######################################################
    # for tanimoto caculation  2019.09.30
    #######################################################
    target_smi='l_'+tpid+'.smi'
    #print target_smi
    c_path=os.getcwd()
    os.chdir('./data/sdf_lib')

    time_sp1=time()
    #FNULL = open(os.devnull, 'w')
    #zp = subprocess.call(['7z', 'e', '-p=swshin', '-y', t_path+'/*.zip','-o'+t_path],stdout=FNULL,stderr=subprocess.STDOUT)
    #obabel_re = subprocess.check_output(['obabel','ligand.lib.fs','-S',target_smi, '-ofpt','-at0.2'],stdout=FNULL,stderr=subprocess.STDOUT)
    obabel_re = subprocess.check_output(['obabel','ligand.lib.fs','-S',target_smi, '-ofpt','-at0.1'])
    obabel_re =obabel_re.strip()
    ob_lines=obabel_re.split('\n')
    #print ob_lines

    os.chdir(c_path)

    #print ele,ob_lines[0][1:]
    t_token=ob_lines[0].split()
    #print t_token

    tani_flag=0

    if 'bits' in t_token:
        #print 'Yes'
        tani_flag=1
    else:
        #print 'No'
        tani_flag=0
    #return 

    # new V5 function
    pid_list_tani=tani_ids(ob_lines)

    #'''
    #check receptor name and ligand name
    if tani_flag==0:
        if(tpid==ob_lines[0][3:]):
            #print 'Names are matched!:',ele,'and',ob_lines[0][1:]
            dic_tani=make_Tanimoto_dic(ob_lines)
        else:
            print 'Names are not matched!',ele,'and',ob_lines[0][1:]
            dic_tani=make_Tanimoto_dic(ob_lines)
    #return
    ################# End of tanimoto #####################

    time_sp2=time()
    print tpid+': '+'Tanimoto searching time : '+str(time_sp2-time_sp1)


    # extract dictionary information for ele(target pdb)
    target_att=OneD_lib_F[tpid]

    # extract the target feature and feature indexes
    t_feature=target_att['f']
    f_index=target_att['i']

    # reverse the t_feature and f_index
    #t_feature_r=t_feature[::-1]
    #f_index_r=f_index[::-1]

    # Make pdb list for a Input
    # Pass I -> OK
    pdb_list_f=Make_pdb_list(q_type,t_feature,OneD_lib_A,OneD_lib_F,f_key_list,pid_list_tani)
    #------------------------------------
    #pdb_list_r=Make_pdb_list(q_type,t_feature_r,OneD_lib_A)
    #sys.exit(1)

    
    ##########################################################################################
    # Scoring part
    ##########################################################################################
    dic_score,dic_pos=Make_dic_score_pos(s_weight,pdb_list_f,OneD_lib_F,t_feature,tpid,dic_tani)
    #------------------------------------
    #dic_score_r,dic_pos_r=Make_dic_score_pos(s_weight,pdb_list_r,OneD_lib_F,t_feature_r,tpid,dic_tani)
    #sys.exit(1)
    #print dic_score

    time_sp1=time()
    #sum_dic_score=sum_dic(dic_score,dic_score_r) # use only forward
    sum_dic_score=dic_score
    #return
        
    # forward -----------------------------------
    #sorted_x   = sorted(dic_score.items(),   key=operator.itemgetter(1), reverse=True)
    # reverse -----------------------------------
    #sorted_x_r = sorted(dic_score_r.items(), key=operator.itemgetter(1), reverse=True)

    # all(forwar and reverse)-------------------
    sorted_a    = sorted(sum_dic_score.items(),key=operator.itemgetter(1),reverse=True)

    #print sorted_x
    #print 
    #print sorted_x_r
    #return 

    #----------------------------------
    #all_result=sorted_x+sorted_x_r
    #print all_result

    #all_result=sorted_x

    all_result=sorted_a
    all_result.sort(key=takeSecond,reverse=True)

    # for scoring test
    # -----------------------
    tmp_str=''
    #print 'writing....'+ele

    #return
    time_sp2=time()
    print tpid+': '+'Scoring time : '+str(time_sp2-time_sp1)


    ########################################################
    # writing the result and copy the canidate files
    #######################################################
    # for pattern is uesed!!
    # prepare the filter_pid
    # print Key_dic_list
    # print ln_key
    # return 

    time_sp1=time()
    path='./data/1D_Out/'+tpid
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        pass

    # for timestamp
    ts = time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    t_token=st.split()
    str_time=t_token[0]+'_'+t_token[1]

    fp_for_summary=open(path+'/'+tpid+'.'+str_time+'.summary.txt','w')
    str_summary='Key Pattern:\tThe Num. of file\n'

    # filtering the result by multiple key pattern
    if(f_flag):
        
        k_order=0
        union=[]
        i_key_count=0
        for f_key in f_key_list:
            a_key_count=0
            p_path=path+'/'+f_key
            #print p_path
            #return
            if not os.path.exists(p_path):
                os.mkdir(p_path)
            else:
                for f in glob.glob(p_path+'/*.*'):
                    os.remove(f)

            # len(f_key)-4: the index of length '4' is '0'
            key_dic_filter=Key_dic_list[len(f_key)-4]
            filter_pid=key_dic_filter[f_key]
            #print filter_pid
            #print f_key
            #print len(f_key)-4

            # for result out as file
            for e in all_result:
                #print ele,e,t_feature,OneD_lib_F[e[0]]['f']
                #print e[0],e[1]
                if e[0] in filter_pid:
                    #print 'Find!',e[0]
                    tmp_str=tmp_str+e[0]+'\t'+str(e[1])+'\t'+t_feature+'\t'+OneD_lib_F[e[0]]['f']+'\n'
                    # for union
                    if k_order==0:
                        #print e[0],OneD_lib_F[e[0]]['f']
                        union.append(e[0])
                    a_key_count+=1
                else:
                    pass

            if k_order>=1:
                tmp_u=[]
                for e in union:
                    #print f_key,OneD_lib_F[e]['f']
                    if f_key in OneD_lib_F[e]['f']:
                        tmp_u.append(e)
                        #print f_key,OneD_lib_F[e]['f']
                        #print 'find!!!'
                union=tmp_u

            #fp_for_out=open('./data/1D_Out/'+ele,'w')
            #print tpid,f_key
            #fp_for_out=open('./data/1D_Out/'+tpid+'.'+f_key+'.txt','w')
            #print path
            fp_for_out=open(p_path+'/'+tpid+'.'+f_key+'.txt','w')
            fp_for_out.write(tmp_str)
            fp_for_out.close()
            copy_candidate(tmp_str,tpid,f_key)
            #return
            k_order+=1

            str_summary=str_summary+f_key+':\t'+str(a_key_count)+'\n'

        ##########################
        # Print the intersection set
        ##########################
        tmp_str=''
        #print union
        # write the union files
        for e in all_result:
            #print ele,e,t_feature,OneD_lib_F[e[0]]['f']
            #print e[0],e[1]
            if e[0] in union:
                #print 'Find!',e[0]
                tmp_str=tmp_str+e[0]+'\t'+str(e[1])+'\t'+t_feature+'\t'+OneD_lib_F[e[0]]['f']+'\n'
                i_key_count+=1

        # make all directory
        p_path=path+'/Intersection'
        #print p_path
        #return
        if not os.path.exists(p_path):
            os.mkdir(p_path)
        else:
            for f in glob.glob(p_path+'/*.*'):
                os.remove(f)

        # write all file
        fp_for_out=open(p_path+'/'+tpid+'.Intersection.txt','w')
        fp_for_out.write(tmp_str)
        fp_for_out.close()
        copy_candidate(tmp_str,tpid,'intersection')
        str_summary=str_summary+'Intersection:\t'+str(i_key_count)+'\n'
        # end of print the union set
        ##########################

    # No filter is uesed!
    else:
        a_key_count=0
        p_path=path+'/no_filter'
        #print p_path
        #return
        if not os.path.exists(p_path):
            os.mkdir(p_path)
        else:
            for f in glob.glob(p_path+'/*.*'):
                os.remove(f)
        
        for e in all_result:
            tmp_str=tmp_str+e[0]+'\t'+str(e[1])+'\t'+t_feature+'\t'+OneD_lib_F[e[0]]['f']+'\n'
            #fp_for_out=open('./data/1D_Out/'+tpid+'.no_f.txt','w')
            fp_for_out=open(p_path+'/'+tpid+'.no_filter.txt','w')
            fp_for_out.write(tmp_str)
            fp_for_out.close()
            copy_candidate(tmp_str,tpid,f_key)
            a_key_count+=1
        str_summary=str_summary+'No_fitler:\t'+str(a_key_count)+'\n'

    fp_for_summary.write(str_summary)
    fp_for_summary.close()
        
    ################ end the writing and copying ###############
    ###########################################################

    # save result for 2Dscan input
    #save_1D_ranking_result(OneD_lib_F,sorted_x,dic_pos,t_feature,ele)
    #save_1D_ranking_result(OneD_lib_F,sorted_x_r,dic_pos_r,t_feature_r,ele)

    time_sp2=time()
    print tpid+': '+'Write and copying time : '+str(time_sp2-time_sp1)


    end_time=time()
    psid=os.getpid()

    print '\nEnding core_f.....'
    ts = time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print st


    print '\nProcess ID: '+str(psid)+' for '+tpid
    #print '.',
    #print 'Total processing time : '+str(end_time-start_time)+'\n\n'
    #print 'end_core_f'
    
    return


def check_key(key_list):

    # define 0 : error
    # define 1 : correct

    chars=set('ADP')
    e_flag=0
    #print key_list

    for akey in key_list:
        akey=akey.upper()
        #print key

        # check the pattern char
        if any((c not in chars) for c in akey):
            e_flag=0
            #return 0
        else:
            e_flag=1
            #return 1

        # check the pattern length: 4<=len<=6
        ln_key=len(akey)    
        if(4<=ln_key and ln_key<=6):
            e_flag=1
        else:
            e_flag=0
            
    if e_flag==1:
        # No error
        return 1
    else:
        # having error
        return 0



def PHscan_1D(s_weight):

    #print 'start_PHscan_1D'
    time1=time()

    #mylist = Manager().list()
    #mydict = Manager().dict()  

    OneD_lib_A={}
    OneD_lib_F={}
    Key_dic_list=[]

    i_base='./input'
    l_base='./'
   
    time_s1=time()

    if not os.path.exists(l_base):
        pass
    else:
	# remove the existing library file
	# exists=os.path.isfile('./1D_lib_A.ADP.json') 
	exists=os.path.isfile('./1D_lib_A.ADP.pickle') 
	if exists:
            #js = open('./1D_lib_A.ADP.json').read()
            #OneD_lib_A= json.loads(js)
            #OneD_lib_A= json.dumps(js)
            #OneD_lib_A= yaml.safe_load(js)
            with open('1D_lib_A.ADP.pickle','rb') as fp:
                OneD_lib_A=pickle.load(fp)
	else:
            print 'There is no 1D_lib_A.ADP.pickle json file'
    #print OneD_lib_A
    #return

    
    # for 1D_lib_F.ADP file
    if not os.path.exists(l_base):
        pass
    else:
	# remove the existing library file
	# exists=os.path.isfile('./1D_lib_F.ADP.json') 
	exists=os.path.isfile('./1D_lib_F.ADP.pickle') 
	if exists:
            #js = open('./1D_lib_F.ADP.json').read()
            #OneD_lib_F= json.loads(js)
            #OneD_lib_F= yaml.safe_load(js)
            #OneD_lib_F= json.dumps(js)
            #OneD_lib_F= yaml.safe_load(js)
            with open('1D_lib_F.ADP.pickle','rb') as fp:
                OneD_lib_F=pickle.load(fp)
	else:
            print 'There is no 1D_lib_F.ADP.json file'
    #print OneD_lib_F
    #return
  

    # for pattern file 4-6
    # example: 1D_key_lib_4.pickle
    if not os.path.exists(l_base):
        pass
    else:
	# remove the existing library file
	# exists=os.path.isfile('./1D_key_lib_4.pickle') 
        for f_index in range(4,7):
            exists=os.path.isfile('./1D_key_lib_'+str(f_index)+'.pickle') 
	    if exists:
                #js = open('./1D_lib_F.ADP.json').read()
                #OneD_lib_F= json.loads(js)
                #OneD_lib_F= yaml.safe_load(js)
                #OneD_lib_F= json.dumps(js)
                #OneD_lib_F= yaml.safe_load(js)
                with open('./1D_key_lib_'+str(f_index)+'.pickle','rb') as fp:
                    key=pickle.load(fp)
                    Key_dic_list.append(key)
    	    else:
                print 'There is no 1D_key_lib_'+str(f_index)+'.pickle file'
    #print Key_dic_list

    time_s2=time()
    print '\nReading library files time : '+str(time_s2-time_s1)

    # read the input ligand list 
    lines_l=[]
    exists=os.path.isfile(sys.argv[1]) 
    if exists:
        fp_for_in=open(sys.argv[1],'r')
        lines_l=fp_for_in.readlines()
    else:
        print 'There is no such a input list'
    #print lines_l


    '''
    # mulitprocessing part using pool and mapa

    pool = multiprocessing.Pool(processes=int(sys.argv[2]))
    #passing pararmeter like this and need 2 steps
    #func=partial(sub_make_complex,base_path)
    func=partial(sub_enva_analysis,fun_type,base_path,mylist)
    #map have only iteratable value
    fail_pdb=pool.map(func,to_be_process)
    fail_pids.append(fail_pdb)
    pool.close()
    pool.join()
    '''

    # This part
    #'''
    pool = multiprocessing.Pool(processes=int(sys.argv[2]))
    func=partial(core_f,s_weight,OneD_lib_A,OneD_lib_F,Key_dic_list)
    pool.map(func,lines_l)
    pool.close()
    pool.join()
    #'''
    # End part


    '''
    # For each Target
    for ele in lines_l:

        # Check time a pdb target
        start_time=time()

        # Check a type: ligand or receptor
        # if a query type is 'L', search receptors
        # if a query type is 'R', search ligands
        ele=ele.strip()
        q_type=check_l_r_apdb(ele)
        print q_type

        if(q_type=='E'):
            print 'there is no info in PHscan_lib1 for query:'+ele
            sys.exit(1)
            continue
        #sys.exit(1)
        print 'Processing..... target PDB :'+ele

        pdb_list=[]
        pdb_list_r=[]
        dic_score={}
        dic_pos={}
        dic_score_r={}
        dic_pos_r={}

        #print ele

        # extract dictionary information for ele(target pdb)
        target_att=OneD_lib_F[ele]

        # extract the target feature and feature indexes
        t_feature=target_att['f']
        f_index=target_att['i']

        # reverse the t_feature and f_index
        t_feature_r=t_feature[::-1]
        f_index_r=f_index[::-1]

        # Make pdb list for a Input
        # Pass I -> OK
        pdb_list_f=Make_pdb_list(q_type,t_feature,OneD_lib_A)
        pdb_list_r=Make_pdb_list(q_type,t_feature_r,OneD_lib_A)
        #sys.exit(1)

        # OK
        dic_score,dic_pos=Make_dic_score_pos(pdb_list_f,OneD_lib_F,t_feature,ele)
        dic_score_r,dic_pos_r=Make_dic_score_pos(pdb_list_r,OneD_lib_F,t_feature_r,ele)
        #sys.exit(1)
        
        sorted_x   = sorted(dic_score.items(),   key=operator.itemgetter(1), reverse=True)
        sorted_x_r = sorted(dic_score_r.items(), key=operator.itemgetter(1), reverse=True)

        all_result=sorted_x+sorted_x_r
        all_result.sort(key=takeSecond,reverse=True)

        for e in all_result:
            print e
        #print all_result
        #sys.exit(1)

        save_1D_ranking_result(OneD_lib_F,sorted_x,dic_pos,t_feature,ele)
        save_1D_ranking_result(OneD_lib_F,sorted_x_r,dic_pos_r,t_feature_r,ele)

        end_time=time()
        print '\nProcessing time for a pdb('+ele+'): '+str(end_time-start_time)+'\n'
    '''

    time2=time()
    print '\n\nTotal processing time : '+str(time2-time1)+'\n\n'
    #print 'end_PHscan_1D'
    return


def select_sample_random():

    # make ligand folder
    l_base='./Select_PDB_Ligand'
    if not os.path.exists(l_base):
        os.makedirs(l_base)
    else:
        arg='rm '+l_base+'/*.pdb'
        #print arg
        os.system(arg)
        #shutil.rmtree(l_base)

    # make receptor folder
    r_base='./Select_PDB_Receptor'
    if not os.path.exists(r_base):
        os.makedirs(r_base)
    else:
        arg='rm '+r_base+'/*.pdb'
        #print arg
        os.system(arg)
        #shutil.rmtree(r_base)

    i_base='./Select_PDB'
    if not os.path.exists(i_base):
        os.makedirs(i_base)
    else:
        arg='rm '+i_base+'/*.pdb'
        #print arg
        os.system(arg)
        #shutil.rmtree(i_base)

    target_list=[]
    cluster_list=[]
    fail_pdb=[]
    num_cluster=0

    fp_for_in=open(sys.argv[1],'r')
    head=fp_for_in.readline()
    lines=fp_for_in.readlines()
    ln_lines=len(lines)
    fp_for_in.close()
    #print ln_lines
    #sys.exit(1)

    tmp_pdb=[]
    fp_for_in2=open(sys.argv[2],'r')
    lines2=fp_for_in2.readlines()
    ln_lines2=len(lines)
    fp_for_in2.close()
    
    for line in lines2:
        fail_pdb.append(line.strip())
    #print fail_pdb
    
    l_count=0
    c_set=[]
    for i in range(0,ln_lines-1):
        token1=lines[i].split(',')
        token2=lines[i+1].split(',')
        if(token1[0]!=token2[0]):
            num_cluster+=1
            cluster_list.append(c_set)
            c_set=[]
        else:
            #print token1[1]
            c_set.append(token1[1])

    #print num_cluster
    #print cluster_list
    #sys.exit(1)

    t_count=0
    s_count=0
    for c in cluster_list:
        t_count+=1
        #print str(t_count)+':'+c[0]
        #print c
        f_count=0
        for sub_c in c:
            #print sub_c
            tmp_pdb=sub_c+'.pdb'
            #print tmp_pdb    
            #print fail_pdb
            #sys.exit(1)
            
            if sub_c in fail_pdb:
                print tmp_pdb+' is in fail list'
                break
            else:
                #sys.exit(1)    
                target_list.append(c[0])
                #print tmp_pdb

	        # check pdb_file_format 
	        exists=os.path.isfile(tmp_pdb) 
	        if exists:
                    HETATM_flag=0
                    atom_count=0
                    fp_for_pdbin=open(tmp_pdb)
                    lines=fp_for_pdbin.readlines()
                    fp_for_pdbin.close()

                    # 2 pass processing
                    receptor_str=''
                    ligand_str=''
                    atom_flag=0
                    hatom_flag=0
                    for line in lines:
                        # processing 'ATOM'
                        if(line.startswith('ATOM')):
                            atom_count+=1
                            receptor_str=receptor_str+line
                            atom_flag=1
                            # max atom count:8000
                            if(atom_count>=8000):
                                break
                        # for write the receptor
                        if(atom_flag==1 and not line.startswith('ATOM')):
                            receptor=receptor_str+'TER\n'+'END\n'
                            fp_for_out=open(r_base+'/'+sub_c+'_r.pdb','w')
                            fp_for_out.write(receptor)
                            fp_for_out.close()

                        # processing 'HETATM'
                        if(line.startswith('HETATM')):
                            HETATM_flag=1
                            '''
                            arg='cp '+tmp_pdb+' ./Select_PDB'
                            os.system(arg)
                            s_count+=1
                            f_count+=1
                            '''
			    if(line[12:17].strip()!='H'):
                                hatom_flag=1
                                ligand_str=ligand_str+line

			if(line.startswith('CONECT') or line.startswith('MASTER') ):
                            ligand_str=ligand_str+line

                        if(hatom_flag==1 and not line.startswith('HETATM') and not line.startswith('CONECT') and not line.startswith('MASTER')):
                            ligand=ligand_str+'TER\n'+'END\n'
                            fp_for_out=open(l_base+'/'+sub_c+'_l.pdb','w')
                            fp_for_out.write(ligand)
                            fp_for_out.close()
                            arg='cp '+tmp_pdb+' ./Select_PDB'
                            os.system(arg)
                            s_count+=1
                            f_count+=1

                    if HETATM_flag and f_count==2:
                        f_count=0
                        break
	        else:
		    break
    print len(target_list)
    print s_count
    #print fail_pdb
    #print target_list

def parsing_enva(*farg):

	p_dic = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K','ILE': 'I', 'PRO': 'P','THR': 'T', 'PHE': 'F', 'ASN': 'N', 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R','TRP': 'W', 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

	type=01
	
	if(len(farg)==1):
		pdb_name=farg[0]
		type=1
	if(len(farg)==2):
		pdb_name=farg[0]
		t_pdb_p=farg[1]
		type=2

	fp_for_in=open(pdb_name,'r')
	head=fp_for_in.readline()
	lines=fp_for_in.readlines()
	fp_for_in.close()
	#print pdb_name
	
	m_pdb_f=[0]*9
	m_pdb_p=[]
	m_pdb_a=[]
	m_pdb_seq=''

	# M = np.empty((0,9),float)
	M = np.empty((0,8),float)

	#print pdb_name
	#sys.exit(1)

	c_count=0
	for line in lines:
		if(not line.startswith('END')):
			line=line.strip()
			token=line.split()
			if(token[29]!='---'):
				c_count+=1
				#print token
				#print line
				m_pdb_p.append(token[5])
				m_pdb_a.append(p_dic[token[3][-3:]])
				m_pdb_seq+=p_dic[token[3][-3:]]
				m_pdb_f[0]=m_pdb_f[0]+int(token[10]) # B-turn-1,else-0 **
				m_pdb_f[1]=m_pdb_f[1]+int(token[11]) # Acc
				m_pdb_f[2]=m_pdb_f[2]+int(token[12]) # Pol
				m_pdb_f[3]=m_pdb_f[3]+int(token[13]) # Phi **
				m_pdb_f[4]=m_pdb_f[4]+int(token[14]) # Psi
				m_pdb_f[5]=m_pdb_f[5]+int(token[15]) # Dep(0:surface,10:out)
				m_pdb_f[6]=m_pdb_f[6]+int(token[16]) # HF(0-10)
				m_pdb_f[7]=m_pdb_f[7]+int(token[21]) # Ac(water accessible) **
				m_pdb_f[8]=m_pdb_f[8]+int(token[22]) # Polarity **
				#print m_pdb_f
				#sys.exit(1)

				#tmp_a=np.array([[float(token[10]),float(token[11]),\
                                #float(token[12]),float(token[13]),float(token[14]),float(token[15]),float(token[16]),float(token[21]),float(token[22])]])
				# remove B-turn
				if(len(farg)==1):
					tmp_a=np.array([[float(token[11]),float(token[12]),\
                                        float(token[13]),float(token[14]),float(token[15]),float(token[16]),float(token[21]),float(token[22])]])
					M=np.vstack((M,tmp_a))
					#print token[11],token[12],token[13],token[14],token[15],token[16],token[21],token[22]
				if(len(farg)==2):
					if(token[5] in t_pdb_p):
						#print t_pdb_p
						#print token
						#print 'Find'
						#print token[5]
						tmp_a=np.array([[float(token[11]),float(token[12]),\
                                                float(token[13]),float(token[14]),float(token[15]),float(token[16]),float(token[21]),float(token[22])]])
						#print token
						#print token[11],token[12],token[13],token[14],token[15],token[16],token[21],token[22]
						M=np.vstack((M,tmp_a))
						#sys.exit(1)
					else:
						#print 'No'
						pass
					#sys.exit(1)
				#M=np.vstack((M,tmp_a))

        if(c_count==0):
            print pdb_name
            arg='enva_error'
            return arg
	#print m_pdb_f
	#if(type==2):
		#sys.exit(1)
	m_pdb_f[0]=float(m_pdb_f[0])/c_count
	m_pdb_f[1]=float(m_pdb_f[1])/c_count
	m_pdb_f[2]=float(m_pdb_f[2])/c_count
	m_pdb_f[3]=float(m_pdb_f[3])/c_count
	m_pdb_f[4]=float(m_pdb_f[4])/c_count
	m_pdb_f[5]=float(m_pdb_f[5])/c_count
	m_pdb_f[6]=float(m_pdb_f[6])/c_count
	m_pdb_f[7]=float(m_pdb_f[7])/c_count
	m_pdb_f[8]=float(m_pdb_f[8])/c_count
	#print m_pdb_f
	#sys.exit(1)
	#print m_pdb_f
	#print m_pdb_p
	#print m_pdb_a
	#print m_pdb_seq

	#Normalize M
	if(sys.argv[1]=='3'):
		#print 'Hi'
		#print M
		Max_M=M.max(axis=0)
		#print Max_M
		Min_M=M.min(axis=0)
		#print Min_M
		#for index in range(0,9):
		for index in range(0,8):
			#print index
			M[:,index]=(M[:,index]-Min_M[index])/(Max_M[index]-Min_M[index])
		#print M
		M_avg=np.average(M,axis=0)
		#print M_avg
		#print m_pdb_p
		#sys.exit(1)

        # return the sequence
	if(sys.argv[1]=='1'):
		return m_pdb_seq
        # return the feature
	if(sys.argv[1]=='2'):
		return m_pdb_f
        # return normalized value and total average
	if(sys.argv[1]=='3'):
		return M_avg, m_pdb_p


def enva_analysis():

    start=time()

    cpu_num=multiprocessing.cpu_count()
    if(len(sys.argv)!=3):
        print '\nUsage: program_name input_parameter num_of_cpu!!\n'
        return

    if(int(sys.argv[2])>cpu_num):
        print 'The number of cpu should be less than '+str(cpu_num)+'\n'
        return
    
    # for data commnuication
    mylist = Manager().list()
    base_path=os.getcwd()
    tar_dir=[]
    to_be_process=[]
    fail_pids=[]
    fun_type=sys.argv[1]

    fp_for_in=open('log.enva.analysis.txt','r')
    passed_id=fp_for_in.readlines()
    fp_for_in.close()

    listOfFiles=filter(os.path.isdir, os.listdir(os.getcwd()))
    listOfFiles.sort(reverse = True)

    #sys.exit(1)
    num_of_pid=len(listOfFiles)
    for d in listOfFiles:
        token=d.split('.')
        if(len(token)>=2):
            tar_dir.append(d)
    print len(tar_dir)
    print tar_dir
    #sys.exit(1)

    print '\n'
    for td in tar_dir:
        td_tmp=td
        td=td+'\n'
        if td not in passed_id:
            to_be_process.append(td_tmp)
        else:
            print td_tmp+': is already processed!\n'

    print to_be_process
    #sys.exit(1)

    # mulitprocessing part using pool and map
    pool = multiprocessing.Pool(processes=int(sys.argv[2]))
    #passing pararmeter like this and need 2 steps
    #func=partial(sub_make_complex,base_path)
    func=partial(sub_enva_analysis,fun_type,base_path,mylist)
    #map have only iteratable value
    fail_pdb=pool.map(func,to_be_process)
    fail_pids.append(fail_pdb)
    pool.close()
    pool.join()

    #print mylist

    fp_for_in=open('log.enva.analysis.txt','a')
    for l in mylist:
        fp_for_in.write(l+'\n')
    fp_for_in.close()
   
    end=time()
    t_s=end-start
    t_m=t_s/60
    t_d=t_m/24
    print 'Execution Time: day:'+str(t_d)+' min.:'+str(t_m)+' second: '+str(t_s)

    print 'failed_pid lists: '
    print fail_pids

    return


def sub_enva_analysis(fun_type,base_path,mylist,target_dir):


        #print fun_type,base_path,mylist,target_dir
        #return
        proc = os.getpid()

        t_path=base_path+'/'+target_dir
        #Binding PDB file and comformers
        os.chdir(t_path)
        listOfFiles= filter(os.path.isfile, os.listdir(os.getcwd()))
        enva_files = [fi for fi in listOfFiles if fi.endswith('.env')]
        enva_files.sort(reverse = True)
        #print enva_files


        # select original Complex
        listOfFiles= filter(os.path.isfile, os.listdir(os.getcwd()))
        files = [fi for fi in listOfFiles if fi.endswith('.pdb')]
        files.sort(reverse = True)

        pdb_orig=''
        fmin=1000
        for f in files:
            f_size=len(f)
            if f_size<fmin:
                fmin=f_size
                pdb_orig=f

        if(len(pdb_orig[:-4])!=5):
            print pdb_orig
            print 'check original pdb! there are no original PDB file!'
            return pdb_orig[:5]

        chain_id=pdb_orig[4:-4]
        print pdb_orig,chain_id

	# ENVA code
	arg2='../SP2.enva_ub -e '+pdb_orig+' '+chain_id
	print arg2
	os.system(arg2)
        #return 


        '''
	id_dic={}

	fp_for_pdb_list=open(sys.argv[1],'r')
	lines=fp_for_pdb_list.readlines()
	fp_for_pdb_list.close()

	for line in lines:
		line=line.strip()
		token=line.split('\t')
		if(len(token)==2):
			id_dic[token[0]]=token[1]
		if(len(token)==1):
			pass
	#print id_dic
        '''

	token_in=sys.argv[1].split('_')
	m_pdb=token_in[0]+'pdb.env'
        m_pdb=pdb_orig+'.env'
        #print m_pdb
        #return
       
        #print sys.argv[1],sys.argv[2]
        #return


	if(sys.argv[1]=='1' or sys.argv[1]=='2'):
		m_pdb_seq=parsing_enva(m_pdb)
	if(sys.argv[1]=='3'):
		m_pdb_nf,m_pdb_p=parsing_enva(m_pdb)
		m_pdb_nf[np.isnan(m_pdb_nf)]=0
		#print m_pdb_nf
		#print m_pdb_p

	#print m_pdb+':'+m_pdb_seq
	#sys.exit(1)
        #return

	matrix = matlist.blosum62
	gap_open = -1.5
	gap_extend = -1.5

	listOfFiles = os.listdir('./')  
	listOfFiles= filter(os.path.isfile, os.listdir(os.getcwd()))
	files = [fi for fi in listOfFiles if fi.endswith('.env')]
	# print listOfFiles
	#count=0
	tmp_str_l=''
	tmp_str_g=''
	tmp_str=''
	for afile in files:
		token=afile.split('.')
		if(len(token)!=5):
			continue
		else:
			env_id=token[2]

		if(sys.argv[1]=='1'):
			a_pdb_seq=parsing_enva(afile)
                        #return
			print 'ID:'+env_id
			print m_pdb+':'+m_pdb_seq
			print afile+':'+a_pdb_seq
                        #return
			#alignments = pairwise2.align.globalxx(seq1, seq2)
			alignments = pairwise2.align.localds(m_pdb_seq,a_pdb_seq,matrix,gap_open, gap_extend)
			top_aln=alignments[0]
			m_pdb_x, a_pdb_y, score, begin, end = top_aln
			print 'Local'
			print m_pdb_x+'\n'+a_pdb_y
			print str(score)+'\n'
			tmp_str_l=tmp_str_l+str(env_id)+'\t'+str(score)+'\n'


			alignments = pairwise2.align.globalds(m_pdb_seq,a_pdb_seq,matrix,gap_open, gap_extend)
			gtop_aln=alignments[0]
			gm_pdb_x, ga_pdb_y, gscore, gbegin, gend = gtop_aln
			print 'Global'
			print gm_pdb_x+'\n'+ga_pdb_y
			print str(gscore)+'\n\n'
			tmp_str_g=tmp_str_g+str(env_id)+'\t'+str(gscore)+'\n'

		if(sys.argv[1]=='2'):
			a_pdb_f=parsing_enva(afile)
			tmp_str=tmp_str+str(env_id)+'\t'
			for ele in a_pdb_f:
				tmp_str=tmp_str+str(ele)+'\t'
			tmp_str=tmp_str+'\n'
			#print tmp_str

		if(sys.argv[1]=='3'):
			t_pdb_nf,t_pdb_p=parsing_enva(afile,m_pdb_p)
                        print afile
			print m_pdb_nf
			#print t_pdb_nf
			t_pdb_nf[np.isnan(t_pdb_nf)]=0
			print t_pdb_nf
			print 'Difference:'
			result=m_pdb_nf-t_pdb_nf
			result_avg=np.average(result)
			print result
			print 'Average:'
			print result_avg
			print '\n\n'
			#if(env_id==76):
				#sys.exit(1)
			tmp_str=tmp_str+str(env_id)+'\t'+str(result_avg)+'\t'
			for ele in np.nditer(result):
				tmp_str=tmp_str+str(ele)+'\t'
			tmp_str=tmp_str+'\n'
			#print tmp_str

			#sys.exit(1)

	if(sys.argv[1]=='1'):
		fp_for_out=open('env_aligment_l_score.csv','w')
		fp_for_out.write(tmp_str_l)
		fp_for_out.close()
		fp_for_out=open('env_aligment_g_score.csv','w')
		fp_for_out.write(tmp_str_g)
		fp_for_out.close()

	if(sys.argv[1]=='2'):
                print target_dir
                print tmp_str
		fp_for_out=open('env_aligment_feature.csv','w')
		fp_for_out.write(tmp_str)
		fp_for_out.close()
	
	if(sys.argv[1]=='3'):
		fp_for_out=open('env_normal_feature.csv','w')
		fp_for_out.write(tmp_str)
		fp_for_out.close()
		pass

def make_complex():

	final_result=''
	
	f=[]
	f.append(sys.argv[1])
	base_pdb=''
    
	fn_token=sys.argv[1].split('.')
	mpdb_name=fn_token[0]
	chain_ID=mpdb_name[-1]
	#print chain_ID
	
	fp_for_in=open(sys.argv[1])
	lines=fp_for_in.readlines()
	
	for line in lines:
		#if(not line.startswith('HETATM')):
		if(line.startswith('ATOM')):
			base_pdb=base_pdb+line
	fp_for_in.close()
	#print base_pdb
	#sys.exit(1)
	
	listOfFiles = os.listdir('./')  
	# print listOfFiles
	listOfFiles= filter(os.path.isfile, os.listdir(os.getcwd()))
	#files = [fi for fi in listOfFiles if not fi.endswith('.pdb')]
	files = [fi for fi in listOfFiles if fi.endswith('.pdb')]
	# print listOfFiles
	count=0
	ln_files=len(files)
	for f in files:
		tmp_pdb=''
		#print f
		token_pdb=f.split('.')
		if len(token_pdb)!=3:
			continue 
        
		count+=1
		print str(count)+'/'+str(ln_files)+':'+f
		token_t=token_pdb[1].split('_')
		pdb_num=token_t[2]
		#print pdb_num
		pdb_tmp_name=token_pdb[0]+'.C'+pdb_num+'.pdb'
		fp_for_pdb=open(f,'r')
		lines_pdb=fp_for_pdb.readlines()
		fp_for_pdb.close()
		for line_pdb in lines_pdb:
			if(line_pdb.startswith('HETATM')):
				if(line_pdb[12:17].strip()!='H'):
					#print line_pdb.strip()
					#print line_pdb[12:17]
					tmp_pdb=tmp_pdb+line_pdb

			if(line_pdb.startswith('CONECT') or line_pdb.startswith('MASTER') ):
				tmp_pdb=tmp_pdb+line_pdb
				#print tmp_pdb
		#print tmp_pdb
		tmp_pdb_f=''
		tmp_pdb_f=base_pdb+tmp_pdb+'END\n'
		#print tmp_pdb_f
		tmp_pdb_name=mpdb_name+'.L.'+pdb_num+'.pdb'
		print tmp_pdb_name
		fp_out_tmp=open(tmp_pdb_name,'w')
		fp_out_tmp.write(tmp_pdb_f)
		fp_out_tmp.close()
		arg='python ../calculate_rmsd3 '+tmp_pdb_name+' ' +sys.argv[1]
		#print arg
		result = commands.getoutput(arg)
		#print result
		final_result=final_result+pdb_num+'\t'+str(result)+'\n'
		
		# ENVA code
		arg2='./enva -e '+tmp_pdb_name+' '+chain_ID
		print arg2
		os.system(arg2)

		os.unlink(tmp_pdb_name)
		#sys.exit(1)

	#os.unlink('tmp_pdb.pdb')
	
	#print final_result
	fp_for_out=open('./Final_RMSD_result.csv','w')
	fp_for_out.write(final_result)
	fp_for_out.close()

def calcal_rmsd():


    #print each_dir,pdb_id
    #exit(1)

    #print sys.argv[1]

    files=[]
    files.append(sys.argv[1])
    files.append(sys.argv[2])

    time1=time()
    M = np.empty((0,3),float)
    f_count=0
    atom_count=[] 
    
    #print files
    #sys.exit(1)

    for filename in files:
        ext=os.path.splitext(filename)[-1]
        if(ext=='.pdb'):
            f_count+=1
            #print filename
            #print path
            #M = np.empty((0,3),float)
            fp_for_in=open('./'+filename,'r')
            lines=fp_for_in.readlines()
            #print lines
            a_count=0
            for line in lines:
                if(line[0:4]=='ATOM'):
                    a_count+=1
                    line=line.strip()
                    x_c=float(line[30:39])
                    y_c=float(line[38:47])
                    z_c=float(line[46:55])
                    tmp_a=np.array([[x_c,y_c,z_c]])
                    #print tmp_a
                    #M=np.concatenate((M,tmp_a),axis=0)
                    M=np.vstack((M,tmp_a))

                if(line[0:6]=='HETATM'):
                    a_count+=1
                    line=line.strip()
                    x_c=float(line[30:39])
                    y_c=float(line[38:47])
                    z_c=float(line[46:55])
                    tmp_a=np.array([[x_c,y_c,z_c]])

                    #print tmp_a
                    #M=np.concatenate((M,tmp_a),axis=0)
                    M=np.vstack((M,tmp_a))
            print str(f_count)+'th '+filename+' atom count is '+str(a_count)
            atom_count.append(a_count)
            #print M
            #print 
            #print M.shape
            #sys.exit(1)
    #print f_count,filename
    #print f_count,path
    #print M.shape
    print atom_count
    #sys.exit(1)

    #pdb_id=pdb_id+'_'+path+'.npy'
    #token=path.split('_')

    #t_pdb_id=pdb_id+token[1][0:2]+'.npy'
    #print t_pdb_id
    #np.save('./'+t_pdb_id,M)
    #M = np.empty((0,3),float)
    #sys.exit(1)
    #print M
    #sys.exit(1)
    #print path

    #print f_count, a_count
    #M=M.reshape(f_count,a_count,3)
    #print M
    #print M.shape
    #pdb_id=pdb_id+'.npy'
    #np.save('./'+pdb_id,M)
    
    #M=merge_matrix(pdb_id)
    M=M.reshape(f_count,a_count,3)
    print M
    print M.shape
    # sys.exit(1)
    #mHandler = MatrixHandler()
    #matrix = mHandler.createMatrix(M,'KABSCH_SERIAL_CALCULATOR')
    #inner_data = rmsd_matrix.get_data() 

    #rmsd_matrix = MatrixHandler().createMatrix(M,'KABSCH_SERIAL_CALCULATOR')
    #inner_data = rmsd_matrix.get_data() 
    #print inner_data
    calculator = pyRMSD.RMSDCalculator.RMSDCalculator(M,'KABSCH_SERIAL_CALCULATOR')
    rmsd = calculator.pairwiseRMSDMatrix()
    rmsd_matrix = CondensedMatrix(rmsd)
    inner_data = rmsd_matrix.get_data()


    # Save the matrix to 'to_this_file.bin'
    # mHandler.saveMatrix(pdb_id)
    # Load it from 'from_this_file.bin'
    # mHandler.loadMatrix("from_this_file")
    # Get the inner CondensedMatrix instance
    # rmsd_matrix = mHandler.getMatrix()
    # print rmsd_matrix

    # rmsd_matrix = MatrixHandler().createMatrix(M, 'KABSCH_SERIAL_CALCULATOR')
    #print M
    #print M.shape
    #pdb_id=pdb_id+'.npy'
    #np.save('./'+pdb_id,M)

    time2=time()
    print time2-time1
    #sys.exit(1)



def extract_coordinate(line):
    
    coordinate=[]

    x_c=line[30:39]
    y_c=line[38:47]
    z_c=line[46:55]

    return


def make_all_pdb():

    #listOfFiles = os.listdir('.')  
    listOfFiles= filter(os.path.isdir, os.listdir(os.getcwd()))
    #print listOfFiles
    #exit(1)
    p_count=0
    for pdb_id in listOfFiles:
        print 'Processing........'+pdb_id
        path='./'+pdb_id
        #print path
        #exit(1)
        make_npy_data(path,pdb_id)
        p_count+=1
        if(p_count==10):
            break
        #merge_matrix(pdb_id)

    return

def merge_matrix(pdb_id):
    
    listOfFiles= filter(os.path.isfile, os.listdir(os.getcwd()))
    files = [ fi for fi in listOfFiles if fi.startswith(pdb_id) ]

    M = np.empty((0,3),float)
    for ele in files:
        #print ele
        test_data=np.load(ele)
        #print test_data.shape
        M=np.vstack((M,test_data))
        os.remove(ele)

    return M
        

def make_npy_data(each_dir,pdb_id):

    #print each_dir,pdb_id
    #exit(1)

    time1=time()
    M = np.empty((0,3),float)
    f_count=0
    #for(path, dir, files) in os.walk("./"):
    for(path, dir, files) in os.walk(each_dir):
        if not dir:
            #print path
            ln_files=len(files)
            #print ln_files
            if ln_files>0:
                for filename in files:
                    ext=os.path.splitext(filename)[-1]
                    if(ext=='.pdb'):
                        f_count+=1
                        #print filename
                        #print path
                        #M = np.empty((0,3),float)
                        fp_for_in=open(path+'/'+filename,'r')
                        lines=fp_for_in.readlines()
                        #print lines
                        a_count=0
                        for line in lines:
                            if(line[0:4]=='ATOM'):
                                a_count+=1
                                line=line.strip()
                                x_c=float(line[30:39])
                                y_c=float(line[38:47])
                                z_c=float(line[46:55])
                        
                                tmp_a=np.array([[x_c,y_c,z_c]])
                                #print tmp_a
                                #M=np.concatenate((M,tmp_a),axis=0)
                                M=np.vstack((M,tmp_a))
                        #print M
                        #print 
                        #print M.shape
                        #sys.exit(1)
                    #print f_count,filename
                #print f_count,path
                #print M.shape
                #pdb_id=pdb_id+'_'+path+'.npy'
                token=path.split('_')

                t_pdb_id=pdb_id+token[1][0:2]+'.npy'
                print t_pdb_id
                np.save('./'+t_pdb_id,M)
                M = np.empty((0,3),float)
                #sys.exit(1)
                #print M
                #sys.exit(1)
        #print path

    #print f_count, a_count
    #M=M.reshape(f_count,a_count,3)
    #print M
    #print M.shape
    #pdb_id=pdb_id+'.npy'
    #np.save('./'+pdb_id,M)
    
    M=merge_matrix(pdb_id)
    M=M.reshape(f_count,a_count,3)
    print M.shape
    mHandler = MatrixHandler()
    matrix = mHandler.createMatrix(M,'KABSCH_SERIAL_CALCULATOR')
    # Save the matrix to 'to_this_file.bin'
    mHandler.saveMatrix(pdb_id)
    # Load it from 'from_this_file.bin'
    # mHandler.loadMatrix("from_this_file")
    # Get the inner CondensedMatrix instance
    # rmsd_matrix = mHandler.getMatrix()

    #rmsd_matrix = MatrixHandler().createMatrix(M, 'KABSCH_SERIAL_CALCULATOR')
    #print M
    #print M.shape
    #pdb_id=pdb_id+'.npy'
    #np.save('./'+pdb_id,M)

    time2=time()
    print time2-time1
    #sys.exit(1)


def make_Pariwise_matrix():

    time1=time()
    coordinates=np.load('1rbp.npy')
    #rmsd_matrix = MatrixHandler().createMatrix(coordinates, 'QCP_SERIAL_CALCULATOR')
    rmsd_matrix = MatrixHandler().createMatrix(coordinates, 'KABSCH_SERIAL_CALCULATOR')
    inner_data = rmsd_matrix.get_data() 
    print inner_data
    time2=time()
    print time2-time1


def add_file_dic_file(tlig_dict,a_file_loc):

	#print a_file_loc
	token=a_file_loc.split('/')
	#print token[-1]
	e_z_name=token[-1]
	#exit(1)
	token1=e_z_name.split('_')
	#print token1[0]
	lig = STBLigand.LigFV(token1[0])
	lig.reload_ligfv(a_file_loc) 
	ln_fseq_count=len(lig.l_fseq)

	l_index=0
	while(l_index<ln_fseq_count):
		alig_dic = {}
		index_A = []
		index_D = []
		index_P = []
		seq = ''.join(lig.l_fseq[l_index])

		index = 0
    		for atom in seq:
    			if atom=='A':
				index_A.append(index)
			if atom=='D':
				index_D.append(index)
			if atom=='P':
				index_P.append(index)
			index+=1
		alig_dic['A']=index_A
		alig_dic['D']=index_D
		alig_dic['P']=index_P
		l_name=lig.title+'.'+str(0)

		tlig_dict[l_name]=alig_dic
		l_index += 1
	return tlig_dict

def make_lib_fold():

	print 'Warning!!!: the exsiting library fold will be deleted! and New library fold will be created'
	check=raw_input("Press Enter to continue...or Press \'n\' to quit: ")
	if(check=='n'):
		sys.exit(1)

	
        # remove the existing library folder
	try:
		shutil.rmtree('./PHscan_Lib1')
	except OSError as e:
		#print ("Error: %s - %s." % (e.filename, e.strerror))
		print '\n\nThere is no \'PHscan_Lib1i\'  but we will create!\n\n'
		#sys.exit(1)


	# remove the existing library file
	exists=os.path.isfile('./ligand.lib.json') 
	if exists:
		os.remove('./ligand.lib.json')
	else:
		pass

	# remove the existing loc. file
	exists=os.path.isfile('./ligand.loc.json') 
	if exists:
		os.remove('./ligand.loc.json')
	else:
		pass
	
	path_base=os.getcwd()
	target='/input'
        i_base='./input'


	input_dir_base=path_base+'/input' 
	#print input_dir_base
	#print path_base+target
 
	if not os.path.exists(i_base):
		print '\nThere is no \'input\' fold!'
    		print 'Make a \'input\' fold!! and put the pdb list!!\n'
    		sys.exit(1)

	#input_fold=os.chdir(path_base+target)
	lines=os.listdir(i_base)
	#print lines
	#exit(1)
	nf_count=0
	tlig_dict={}
	loc_dict={}
	for line in lines:
		tmp_pos='./PHscan_Lib1'
		line=line.strip()
		for c_char in line:
        		tmp_pos=tmp_pos+'/'+c_char
    		#tmp_pos = path_base + tmp_pos
		#print tmp_pos
		#exit(1)
		if not os.path.exists(tmp_pos):
			nf_count+=1
			print "path doesn't exist. trying to make"
			os.makedirs(tmp_pos)
			#source_dir=input_dir_base+'/'+line
			source_dir=i_base+'/'+line
                        #print source_dir
			source_files=os.listdir(source_dir)
                        #print source_files

			loc_dict[line]=tmp_pos
                        #print loc_dict
	
			#exit(1)
			for a_file in source_files:
				if(a_file.find('_conf.')==-1):
					a_file_loc=source_dir+'/'+a_file
					copy(a_file_loc,tmp_pos)
				else:
					a_file_loc=source_dir+'/'+a_file
					z_name=source_dir+'/'+a_file+'.zip'
					#print z_name
					#print a_file_loc
					#sys.exit(1)

					if(a_file.find('_conf.dump')!=-1):
						tlig_dict=add_file_dic_file(tlig_dict,a_file_loc)
						#sys.exit(1)
					#sys.exit(1)

					zp = subprocess.call(['7z', 'a', '-p=swshin', '-y', z_name] + [a_file_loc])
					#a_file_loc_zip=a_file_loc+'.zip'
					#print a_file_loc
					z_file_loc=a_file_loc+'.zip'
					#print a_file_loc,z_file_loc,tmp_pos
					copy(z_file_loc,tmp_pos)
					os.remove(z_file_loc)
					#sys.exit(1)
				#sys.exit(1)
			#sys.exit(1)
		#sys.exit(1)
		else:
			lines=os.listdir(tmp_pos)
			num_files=len(lines)
			#print num_files
			if(num_files==8):
				print '\nPdb \''+line+'\' is already processed and exists!\n'
			else:
				print 'Some file is omitted! Check fold:'+tmp_pos+'\n'
				sys.exit(1)
		#sys.exit(1)
		#print tmp_pos

	with open('./ligand.loc.json','w') as fp1:
		json.dump(loc_dict,fp1)

	with open('./ligand.lib.json','w') as fp2:
		json.dump(tlig_dict,fp2)
	print '\n\n\nTotal '+str(nf_count)+' id(s) is(are) created and made a new library fold!\n\n\n'


def add_lib_fold():

        i_base='./input'
        l_base='./PHscan_Lib1'

        if not os.path.exists(l_base):
            pass
        else:
	    ## Loading existing lib file
    	    js = open('./ligand.lib.json').read()
	    tlig_dict= json.loads(js)

	    ## Loading existing lib loc. file
	    js = open('./ligand.loc.json').read()
	    loc_dict= json.loads(js)


	path_base=os.getcwd()
	target='/input'
        
	input_dir_base=path_base+'/input' 
	#print input_dir_base
	#print path_base+target


	if not os.path.exists(i_base):
		print '\nThere is no \'input\' fold!'
    		print 'Make a \'input\' fold!! and put the pdb list!!\n'
    		sys.exit(1)

	#input_fold=os.chdir(path_base+target)
	lines=os.listdir(i_base)
	#print lines
	#exit(1)

	af_count=0
	ef_count=0
	for line in lines:
		tmp_pos='./PHscan_Lib1'
		line=line.strip()
		for c_char in line:
        		tmp_pos=tmp_pos+'/'+c_char
    		#tmp_pos = path_base + tmp_pos
		#print tmp_pos
		#exit(1)
		if not os.path.exists(tmp_pos):
			af_count+=1
			print 'path doesn\'t exist. trying to make'
			os.makedirs(tmp_pos)
			#source_dir=input_dir_base+'/'+line
			source_dir=i_base+'/'+line
			source_files=os.listdir(source_dir)

			loc_dict[line]=tmp_pos

			#exit(1)
			for a_file in source_files:
				if(a_file.find('_conf.')==-1):
					a_file_loc=source_dir+'/'+a_file
					copy(a_file_loc,tmp_pos)
				else:
					a_file_loc=source_dir+'/'+a_file
					z_name=source_dir+'/'+a_file+'.zip'
					#print z_name
					#print a_file_loc
					#sys.exit(1)

					if(a_file.find('_conf.dump')!=-1):
						tlig_dict=add_file_dic_file(tlig_dict,a_file_loc)
						#sys.exit(1)

					zp = subprocess.call(['7z', 'a', '-p=swshin', '-y', z_name] + [a_file_loc])
					#a_file_loc_zip=a_file_loc+'.zip'
					#print a_file_loc
					z_file_loc=a_file_loc+'.zip'
					#print a_file_loc,z_file_loc,tmp_pos
					copy(z_file_loc,tmp_pos)
					os.remove(z_file_loc)
					#sys.exit(1)
				#sys.exit(1)
			#sys.exit(1)
		#sys.exit(1)
		else:
			ef_count+=1
			lines=os.listdir(tmp_pos)
			num_files=len(lines)
			#print num_files
			if(num_files==8):
				print '\nPdb \''+line+'\' is already processed and exists!\n'
			else:
				print 'Some file is omitted! Check fold:'+tmp_pos+'\n'
				sys.exit(1)
		#sys.exit(1)
		#print tmp_pos

	## Dump location to file
	with open('./ligand.loc.json','w') as fp1:
		json.dump(loc_dict,fp1)

	## Dump dictionary to file
	with open('./ligand.lib.json','w') as fp2:
		json.dump(tlig_dict,fp2)

	print '\n\n\n'
	print 'Total '+str(af_count)+' id(s) is(are) added to library fold!'
	print 'Total '+str(ef_count)+' id(s) exist(s) and skipped!\n\n'


def del_lib_fold():

	print 'Warning!!!: the exsiting library fold(s) will be deleted!'
	check=raw_input("Press Enter to continue...or Input \'n\' to quit: ")
	if(check=='n'):
		sys.exit(1)

        l_base='./PHscan_Lib1'
        if not os.path.exists(l_base):
            pass
        else:
	    ## Loading existing lib file
    	    js = open('./ligand.lib.json').read()
	    tlig_dict= json.loads(js)

	    ## Loading existing lib loc. file
	    js = open('./ligand.loc.json').read()
	    loc_dict= json.loads(js)


	path_base=os.getcwd()
	target='/input'

	pdb_id_lists=[]
        ln_sys_argv=len(sys.argv)
        for ele in range(2,ln_sys_argv):
            pdb_id_lists.append(sys.argv[ele])
	#target_id=sys.argv[2]
	#print target_id+'HIHIHI'
	#exit(1)
	if(len(pdb_id_lists)==0):
		show_usage()
		sys.exit(1)

	## Check if argv is file or not.
	exists=os.path.isfile('./'+pdb_id_lists[0]) 
	if exists:
		fp_for_in=open('./'+ipdb_id_lists[0],'r')
		pdb_id_lists=fp_for_in.readlines()
                fp_for_in.close()
	
	input_dir_base=path_base+'/input' 

	df_count=0
	print 

        # if case is 'All'
        if(pdb_id_lists[0]=='All' or pdb_id_lists[0]=='all'):
            if not os.path.exists(l_base):
                print 'There is no PHscan_lib1 folds\n'
                sys.exit(1)
            else:
                ## Remove the all library fold
                shutil.rmtree(l_base)
                print 'All PHscan_lib1 folds are removed!\n'

		## Remove the existing library file
		exists=os.path.isfile('./ligand.lib.json') 
		if exists:
		    os.remove('./ligand.lib.json')
		else:
		    pass

		## Remove the existing loc file
		exists=os.path.isfile('./ligand.loc.json') 
		if exists:
		    os.remove('./ligand.loc.json')
		else:
		    pass
                sys.exit(1)
        else:
            tmp_pos='./PHscan_Lib1'
            for line in pdb_id_lists:
	        line=line.strip()

		## Making path with pdb id
		for c_char in line:
        		tmp_pos=tmp_pos+'/'+c_char
    		#tmp_pos = path_base + tmp_pos
		#print tmp_pos
		#exit(1)
		
                if not os.path.exists(tmp_pos):
			print 'PDB id:'+line+' is not found! and check the id again!\n\n'
			exit(1)
		else:
			# remove the existing library folder
			try:
				print 'Delelting the '+line+' fold and fiels!'
				shutil.rmtree(tmp_pos)
				del tlig_dict[line+'.0']
				del loc_dict[line]
				df_count+=1
			except OSError as e:
				print ("Error: %s - %s." % (e.filename, e.strerror))
				os.exit(1)

	## Dump location to file
	with open('./ligand.loc.json','w') as fp1:
		json.dump(loc_dict,fp1)

	## Dump dictionary to file
	with open('./ligand.lib.json','w') as fp2:
		json.dump(tlig_dict,fp2)

	print '\n\n\nTotal '+str(df_count)+' id(s) is(are) deleted\n\n\n'

        sys.exit(1)

        '''
	for line in pdb_id_lists:
		tmp_pos='/PHscan_Lib1'
		line=line.strip()
		
		# if case is 'All'
		if (line=='All' or line=='all'):
			if not os.path.exists('.'+tmp_pos):
				sys.exit(1)
			else:
				## Remove the all library fold
				shutil.rmtree('.'+tmp_pos)
				## Remove the existing library file
				exists=os.path.isfile('./ligand.lib.json') 
				if exists:
					os.remove('./ligand.lib.json')
				else:
					pass
				## Remove the existing loc file
				exists=os.path.isfile('./ligand.loc.json') 
				if exists:
					os.remove('./ligand.loc.json')
				else:
					pass
				sys.exit(1)
	
		## Making path with pdb id
		for c_char in line:
        		tmp_pos=tmp_pos+'/'+c_char
    		tmp_pos = path_base + tmp_pos
		print tmp_pos
		#exit(1)

		if not os.path.exists(tmp_pos):
			print 'PDB id:'+line+' is not found! and check the id again!'
			exit(1)
		else:
			# remove the existing library folder
			try:
				print 'Delelting the '+line+' fold and fiels!'
				shutil.rmtree(tmp_pos)
				del tlig_dict[line+'.0']
				del loc_dict[line]
				df_count+=1
			except OSError as e:
				print ("Error: %s - %s." % (e.filename, e.strerror))
				os.exit(1)
	
	## Dump location to file
	with open('./ligand.loc.json','w') as fp1:
		json.dump(loc_dict,fp1)

	## Dump dictionary to file
	with open('./ligand.lib.json','w') as fp2:
		json.dump(tlig_dict,fp2)
	print '\n\n\nTotal '+str(df_count)+' id(s) is(are) deleted\n\n\n'
        '''

def show_lib_fold_info():

	# Read the lib file as global variable
	ligand_dict={}
	exists = os.path.isfile('./ligand.lib.json') 
	if exists:
		js= open('./ligand.lib.json').read()
		ligand_dict = json.loads(js)
		icount=0
		for key, value in ligand_dict.iteritems():
			print key, value
			icount+=1
		print 'Total '+str(icount)+' id exists\n'
	else:
		print 'There are no \'ligand.lib.json\' file'
		print 'Creat the \'ligand.lib.json\' file first! using optiion -c'


def show_usage():
	print 'Usage: command \'-c\' to creat a new indexing file'
	print '               \'-a\' to add a new id to indexing file'
	print '               \'-d\' id or id_list_file  to delete a id from indexing file'
	print '               \'-s\' to show indexing file'


#############################################################################################

def Reading_Input_File(in_file):
    
    fp_for_in = open(in_file,'r')
    head = fp_for_in.readline()
    skip_line = fp_for_in.readline()

    lines =fp_for_in.readlines()
    fp_for_in.close()


    '''
    in_dir='./Input_files' 
    if not os.path.exists(in_dir):
        os.mkdir(in_dir)
    else:
        arg='rm '+in_dir+'/*'
        os.system(arg) 
    '''

    Main_SMI_list=[]
    Main_ID_list=[]

    idx=0
    for aline in lines:
        token = aline.split(',')
        #print (idx,token[0],token[-2])
        Main_ID_list.append(token[0])
        #Main_SMI_list.append(token[-2])
        Main_SMI_list.append(token[-3])

    return Main_ID_list,Main_SMI_list
    


def Making_Pair(Main_SMI_list):

    tmp_list=[]
    for pair in itertools.product(Main_SMI_list, repeat=2):
        tmp_list.append(pair)

    #print (len(tmp_list))
    set_tmp_list = set(tmp_list)
    re_list_set = set()

    for ele in tmp_list:
        if ele[0] == ele[1]:
            continue
        r_ele = tuple([ele[1],ele[0]])
        if (ele not in re_list_set) and (r_ele not in re_list_set):
            re_list_set.add(ele)
    #print (len(re_list_set))

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

    #print Re_list

    #return

    #print mylist

    idx=0
    for ele in Re_list:
        #print ele,'\n'
        x,y =  Main_ID_list.index(ele[0]),Main_ID_list.index(ele[1])
        #print x,y 
        pcs_m[x][y]=float(ele[2])
        pcs_m[y][x]=float(ele[2])
        pcs_m[x][x]=1.0
        pcs_m[y][y]=1.0
    
    #print (pcs_m)
    print ('  -> Done')

    time_sp2=time()
    print '  -> Processing time to make distance matrix: '+str(time_sp2-time_sp1)+'s for '+str(len(Main_ID_list))+' ZIDs'

    return pcs_m



def Check_LSAlign():
    if not os.path.exists('./LSalign'):
        print ' This python program needs \'LSalign\'.'
        print ' Move the \'LSalign\' program at this fold and re excute this program.'
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

    #print(t_id,t_smi,q_id,q_smi)
    # For Check '.' in smiles for disconnection
    #q_smi= Check_SMI_Dot(q_smi)

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



def Make_ZID_Dic(Main_ID_list,Main_SMI_list):

    tmp_Dic = {}

    idx=0
    for aID in Main_ID_list:
        tmp_Dic[aID] = Main_SMI_list[idx]
        idx+=1

    return tmp_Dic



def Clustering_ligand(Main_ID_list,Main_SMI_list,re_mat,in_file):
    fn = in_file.split(".")[0]

    ZID_Dic = Make_ZID_Dic(Main_ID_list,Main_SMI_list)

    if re_mat == 'y':
        #Dis_Mat = Make_Distance_Mat(Main_ID_list,Main_SMI_list,ZID_Dic)
        Dis_Mat = Make_Distance_Mat(Main_ID_list[0:1000],Main_SMI_list[0:1000],ZID_Dic)
        print ('  -> Saving the distance matrix file as \'Dis_matrix.save.npy\'....')
        #np.save('./Dis_matrix.save.npy',Dis_Mat)
        np.save("./%s.Dist_matrix.save.npy"%fn,Dis_Mat)
    else:
        if os.path.exists('./Dis_matrix.save.npy'):
            print ('Loading the distance matrix file\n')
            Dis_Mat = np.load('./Dis_matrix.save.npy')
        else:
            print ('There is no Distance matrix file.\n Make Distance matrix file first')
            sys.exit(1)

    return


def main():

    time_sp1=time()

    parser=argparse.ArgumentParser()
    parser.add_argument('-infile',required=True, help='input csv file')
    parser.add_argument('-re_mat',required=True, choices=['y','n'], help='input csv file')
    args=parser.parse_args()

    Input_CSV = args.infile 
    Rebuild   = args.re_mat

    Main_ID_list,Main_SMI_list = Reading_Input_File(Input_CSV)
    Clustering_ligand(Main_ID_list,Main_SMI_list,Rebuild,Input_CSV)


    #Add_M_weight()
    #Cal_M_weight()
    #anal_1DScan()
    #PHscan_1D()
    #select_sample_random()
    #enva_analysis()
    #make_complex()
    # calcal_rmsd()
    # make_all_pdb()
    # make_npy_data()
    # make_Pariwise_matrix()


    time_sp2=time()
    print '\nProcessing the clustering: '+str(time_sp2-time_sp1)+'\n\n'



if __name__=="__main__":

    main()
