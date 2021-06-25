import glob 
import sys,os
from time import time 
import argparse
import glob 
import os
import subprocess
import pickle
from datetime import datetime
from os import path 

from rdkit import RDConfig
from rdkit import Chem 
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw 
from rdkit.Chem import ChemicalFeatures

import multiprocessing
from multiprocessing import Process, Manager
from functools import partial
import json 
import copy 
import re
import pandas as pd 
import traceback
import numpy as np

from pybel import *

# For Database
import sqlite3
from sqlite3 import Error
import apsw 
import glob 

# For Zipfile
import zipfile
from io import BytesIO
from zipfile import ZipFile
from StringIO import StringIO

from multiprocessing import Process, current_process
import shutil

#For call user define function
from MD_Backbone import *
from MD_DB import *
from MD_Align import *
from MD_Output import *

def Fetch_BB_Rings(ring,up_mw,low_mw,DB_Path):
	didi = {}
	try :
		#conn = apsw.Connection(DB_Path + '/ZINC_BB.db')
		#conn = apsw.Connection(DB_Path + '/ZINC_BB_idx.db') # ZINC_BB_idx.db is indexed DB
		conn = apsw.Connection(DB_Path + "ZINC_PBB_idx.db")
	except:
		print("DB Connection Error")
		sys.exit(1)
	cursorObj = conn.cursor()
	try :
		#cursorObj.execute("SELECT BB FROM BB_Table WHERE Ring=? LIMIT 500",(ring,))
		cursorObj.execute("SELECT BB FROM PBB_Table WHERE Ring=? AND MW>=? AND MW<=?",(ring,np.float64(low_mw),np.float64(up_mw),))
		#cursorObj.execute("SELECT BB FROM BB_Table WHERE Ring=?",(ring,))
	except:
		print("Query Error")
		sys.exit(1)
		
	print("DB Fetch Start")
	rows = cursorObj.fetchall()
	for i,j in zip(rows,range(len(rows))):
		#didi[str(j)] = unicode.encode(i[0],'utf-8')
		didi[str(j)] = readstring("smi",unicode.encode(i[0],'utf-8')).write("smi").strip()
	print("Length Of Backbones : " + str(len(didi)))
	print("DB Fetch End")

	lenid = len(didi)//19 # define save point
	
	di_keys = list(divide_list(didi.values(),lenid))
	di_ids = list(divide_list(didi.keys(),lenid))
	
	return di_keys,di_ids,didi

def divide_list(ls,n):
	if n == 0:
		num = 1
	else:
		num = n
	for i in range(0,len(ls),num):
		yield ls[i:i+num]
def multi_file_remove_func(a):
	# using multirprocessing for removing faster 
	os.remove(a)
def query_parameters(smiles,mw_percent):
	InputCP = Extract_CP(smiles)
	Input_Num_Ring = InputCP["Ring"]
	Input_MW = InputCP["MW"] #np.float64(InputCP["MW"])
	per_MW= Input_MW*np.float64(mw_percent)/np.float64(100)
	up_MW = Input_MW + per_MW
	low_MW = Input_MW - per_MW
	return Input_Num_Ring,up_MW,low_MW
def Fetch_BB_smis_old(smiles,DB_Path):
    print("""
    ##############################################################
    # Extract SMILES by Chemical Properties using ZINC_BB_idx.db #
    ##############################################################
    """)
    smis = []
    didi = {}
    roro = set()
    re = Extract_CP(smiles)
    idid = []
    try :
        conn = sqlite3.connect(DB_Path + "./ZINC_PBB_idx.db")
    except:
        print("ZINC BB DB Connection Error")
        sys.exit(1)
    cursorObj = conn.cursor()

    try :
        cursorObj.execute("SELECT BB FROM PBB_Table WHERE Ring=? AND MW=? AND LogP=? AND TPSA=?",(re["Ring"],re["MW"],re["LogP"],re["TPSA"],))
    except:
        print("Error %s"%smiles)
    rows = cursorObj.fetchall()
    for i in rows :
        for ids in set(unicode.encode(i[0],'utf-8').split(',')):
            roro.add(ids)
    roro = list(roro)
    for i,j in zip(roro,range(0,len(roro))):
        didi[str(j)] = i
    df1 = AlignM3D('titi',smiles,didi.keys(),didi.values()) # Align
    if df1 is None:
        return
    else:
		
        for i in df1["Query"]:
            smis.append(didi[i])
        df1["SMILES"] = smis
        exp_df = df1[df1["PCScore"]<0.98]
        out_smiles = df1[df1["PCScore"]>=0.98]["SMILES"] # self Align cutoffa
        return out_smiles
def Fetch_BB_smis(smiles,smi_set,DB_Path):
	smis = []
	didi = {}
	roro = set()
	re = Extract_CP(smiles)
	idid = []
	entities = (re["Ring"],re["MW"],re["LogP"],re["TPSA"])
	try:
		conn = sqlite3.connect(DB_Path + "./ZINC_PBB_idx.db")
	except:
		print("PBB DB Connection Error")
		sys.exit(1)
	cursorObj = conn.cursor()

	try:
		cursorObj.execute("SELECT BB FROM PBB_Table WHERE Ring=? AND MW=? AND LogP=? AND TPSA=?",entities)
	except:
		print("Query Error : %s"%smiles)
		sys.exit(1)

	rows = cursorObj.fetchall()

	if len(rows) == 1:
		out_smiles = unicode.encode(rows[0][0],"utf-8")
		return [out_smiles]
	else:
		for i in rows:
			for ids in set(unicode.encode(i[0],"utf-8").split(",")):
				if ids in smi_set:
					pass
				else:
					smi_set.add(ids)
					roro.add(ids)
		roro = list(roro)
		if len(roro) == 1:
			return roro
		elif not roro:
			return
		for i in roro:
			j = roro.index(i)
			didi[str(j)] = i
		df1 = AlignM3D("temp_align",smiles,didi.keys(),didi.values())
		if df1 is None:
			return 
		else:
			for i in df1["Query"]:
				smis.append(didi[i])
			df1["SMILES"] = smis
			exp_df = df1[df1["PCScore"]< 0.98]
			out_smiles = df1[df1["PCScore"] >= 0.98]["SMILES"]
			return out_smiles
def Final_Annot(idid,id_BB,pcscore,DB_Path):
	id_df = []

    ##################
    # Add Annotation #
    ##################
	for ii in idid:
		iidf = Fetch_CP_Annot(ii,DB_Path)
		iiidf = Fetch_Purch_Annot(ii,DB_Path)
		
		if iiidf is None:
			pass
		else:
			midf = pd.merge(iidf,iiidf,on="ZID")
			midf["Backbone"] = [id_BB[ii]]
			id_df.append(midf)
	if id_df:
		id_df = pd.concat(id_df)
		id_df["PCScore"] = pcscore
		return id_df
	else:
		return
def Fetch_Purch_Annot(ids,DB_Path):
	try :
		conn = sqlite3.connect(DB_Path + "./ZINC_P_ID120M.db")
	except:
		print("DB Connection Error")
		sys.exit(1)

	cursorObj = conn.cursor()
	try :
		cursorObj.execute("SELECT * FROM ID_Table WHERE ZID=?",(ids,))
	except:
		print("Error %s"%ids)
		sys.exit(1)
	
	rows = cursorObj.fetchall()
	idf = pd.DataFrame(rows).rename(columns={0:"ZID"})
	if "CHEMBL" in ids or "STK" in ids:
		idf = pd.DataFrame([ids,"Unknown"]).T.rename(columns={0:"ZID",1:"Purchasability"})
		return idf
	elif idf.empty :
		return
	else:
		idf["Purchasability"] = ["Purchasable"]
		return idf
def Fetch_CP_Annot(ids,DB_Path):
    ##################################################
    # Extract Annotation by ZID using ZINC_CP_idx.db #
    ##################################################
    ids = ids.strip('\'')
    try :
        conn = sqlite3.connect(DB_Path + "./ZINC_PCP_idx.db")
    except:
        print("DB Connection Error")
        sys.exit(1)
    cursorObj = conn.cursor()

    try :
        cursorObj.execute("SELECT * FROM PCP WHERE ZID=?",(ids,))
    except:
        print("Error %s"%ids)
        sys.exit(1)
    rows = cursorObj.fetchall()
    idf = pd.DataFrame(rows)
    idf.columns = ['ZID','SMILES','MW','LogP','TPSA','RotatableB','HBD','HBA','Ring','Total_Charge','HeavyAtoms','CarBonAtoms','HeteroAtoms','Lipinski_Violation','VeBer_Violation','Egan_Violation','Toxicity']
    return idf
def Fetch_BB_IDs(smiles,idid,id_BB,DB_Path):
    try:
        conn = sqlite3.connect(DB_Path + "./ZINC_PBB_idx.db")
    except:
        print("DB Connection Error")
        sys.exit(1)

    cursorObj = conn.cursor()
    try:
        cursorObj.execute("SELECT IDs FROM PBB_Table WHERE BB=?",(smiles,))
    except:
        print("Error %s"%smiles)
        sys.exit(1)
    rows = cursorObj.fetchall()
    for i in rows:
        for ids in set(unicode.encode(i[0],'utf-8').split(',')):
            idid.append(ids.strip('\''))
            id_BB[ids.strip("\'")] = smiles
def Fetch_SMILES_by_ID(zid,id_smi,DB_Path):
	try:
		conn = sqlite3.connect(DB_Path + "./ZINC_PCP_idx.db")
	except:
		print("DB Connection Error")
		sys.exit(1)
	cursorObj = conn.cursor()
	try:
		cursorObj.execute("SELECT SMILES FROM PCP WHERE ZID=?",(zid,))
	except:
		print("Error %s"%zid)
		sys.exit(1)
	rows = cursorObj.fetchall()
	smi = ''
	if not rows:
		pass
	else:
		smi = unicode.encode(rows[0][0],"utf-8")
		id_smi[zid] = smi
	return id_smi
		
def Make_BB_Align_result2(file_name,id_set,ncutoff,aBB,DB_Path):
    # paths
    output_path = "./Data/ADC_Output/"
    sdf_out_path = os.path.join(output_path,"Out_SDF_%s"%file_name)
    input_path = "./Data/Input/" #"./Data/test_Input_yong/" #"./Data/Input/"
    infile = output_path + file_name + '.total.csv'
    # list and set
    df_list = []
    df_list2 = []
    df_list3 = []
    id_BB = Manager().dict()
    id_smi = Manager().dict()

    # constant values
    Ncpu = multiprocessing.cpu_count()
    # input file information

    ismi = input_path + file_name + '.smi'
    wsmi = Read_SMILES_FILE(ismi)
    wsmi = readstring("smi",wsmi).write("smi").strip()
    re_in = Extract_CP(wsmi)
    BB_in = Extract_BB(wsmi)
    wsmi_df = pd.DataFrame.from_dict(re_in,orient='index').T
    wsmi_df['ZID'] = '* ' + file_name
    wsmi_df["Backbone"] = BB_in

    df = pd.read_csv(infile)
    smi_list = df["SMILES"].drop_duplicates()

    if os.path.exists(sdf_out_path):
        os.system("rm %s/*"%sdf_out_path)
    else:
        os.mkdir(sdf_out_path)
    for smi in smi_list:
        ids = Manager().list()
        pcscore = np.float64(df[df["SMILES"] == smi]["PCScore"][:1])
        Fetch_BB_IDs(smi,ids,id_BB,DB_Path)
        dif = Final_Annot(ids,id_BB,pcscore,DB_Path)
        df_list.append(dif)
        if dif is None:
            pass
        else:
            for i in set(dif["ZID"].tolist()):
                id_set.add(unicode.encode(i,'utf-8'))
                print(len(id_set))
        if len(id_set) >= int(ncutoff):
            break
    if not df_list: return id_set,pd.DataFrame()
    else: pass
    try:
        fin_df = pd.concat(df_list).drop_duplicates()
    except:
        return set(),pd.DataFrame()
    for i in id_set:
        drop_df = fin_df[fin_df["ZID"] == i].sort_values(by="PCScore",ascending=False)[:1]
        df_list3.append(drop_df)
    fin_df = pd.concat(df_list3).sort_values(by="PCScore",ascending=False)
    pool = multiprocessing.Pool(Ncpu-2)
    func = partial(Fetch_SMILES_by_ID,id_smi=id_smi,DB_Path=DB_Path)
    pool.map(func,fin_df["ZID"])
    pool.close()
    pool.join()
    fin_score = AlignM3D(file_name,wsmi,id_smi.keys(),id_smi.values())
    fin_score.rename(columns={"Query":"ZID","PCScore":"Z_PCScore"},inplace=True)
    fin_df1 = pd.merge(fin_df,fin_score[["ZID","Z_PCScore"]],on="ZID").sort_values(by="Z_PCScore",ascending=False).rename(columns={"PCScore":"BB_PCScore"})
    tier_list = [] 
    for i in fin_df1["ZID"]:
        tier_list.append("T 1.5 Inner-Scaffold Search")
    fin_df1["Tier"] = tier_list
    fin_df1 = pd.concat([wsmi_df,fin_df1])
	
    fin_df1 = fin_df1[['ZID',"Z_PCScore",'BB_PCScore','MW','LogP','TPSA','RotatableB','HBD','HBA','Ring','Total_Charge','HeavyAtoms','CarBonAtoms','HeteroAtoms','Lipinski_Violation','VeBer_Violation','Egan_Violation','Toxicity','SMILES',"Backbone",'Purchasability',"Tier"]]
	
    fin_df1.to_csv(output_path + file_name + '_' + str(ncutoff) + '.csv',index=False)
	
    print("Number Of Extracting SDF Files : " + str(len(fin_df["ZID"].drop_duplicates())-1))

    os.chdir(sdf_out_path)
    """
    pool = multiprocessing.Pool(Ncpu-2)
    func = partial(SDF_load_MainS,DB_Path=DB_Path)
    pool.map(func,fin_df["ZID"])
    pool.close()
    pool.join()"""

    os.chdir('../../../')
    return id_set,fin_df1


def Extract_smiFromZID(zincid,DB_Path):
	SDF_load_MainS(zincid,DB_Path)
	smi = readfile("sdf",zincid + '.sdf').next().write("smi").strip().split()[0]
	os.remove(zincid + ".sdf")
	return smi
def load_BBID():
	with open("8M_BB_ID.dict.pkl","rb") as F:
		tmp_dic = pickle.load(F)
	return tmp_dic
def split_BB_list(input_path,ID_DIC):
	ff = glob.glob(input_path + "*.list")
	for i in ff :
		with open(i,"r") as F:
			lili = F.readlines()
	for line in lili:
		with open(input_path + ID_DIC[line.strip()]+'.smi',"w") as W:
			W.write(line)

'''
def BA_Class_show():
	print("""
	About Fetch\n
		Fetch_BB_Rings(ring,up_mw,low_mw,DB_Path)\n
		Fetch_Purch_Annot(ids,DB_Path)\n
		Fetch_CP_Annot(ids,DB_Path)\n
		Fetch_BB_IDs(smiles,idid,DB_Path)\n
		Fetch_BB_smis(smiles,DB_Path)\n
	About Operation for files\n
		SDF_load_MainS(zincid,DB_Path)\n
		multi_file_remove_func(a)\n
		Make_BB_Align_result(file_name,id_set,ncutoff,DB_Path)\n
	About Calculation for Operation\n
		divide_list(ls,n)\n
		query_parameters(smiles,mw_percent)\n
		Final_Annot(idid,pcscore,DB_Path)\n
		Extract_smiFromZID(zincid,DB_Path)\n
		make_BBdic_by_multiproc(mdic,DB_Path,i)""")'''
	
"""
if __name__ == "__main__":
	DB_Path = "/ssd/swshin/1D_Scan.v2/Data/DB_Table/"
	Fetch_Purch_Annot("CHEMBL3780743",DB_Path)"""
