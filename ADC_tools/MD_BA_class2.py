
import glob,sys,os,argparse,subprocess,multiprocessing
import json,copy,re,traceback

from time import time
from datetime import datetime
from multiprocessing import Process, Manager
from functools import partial
from pybel import *

from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import ChemicalFeatures

import pandas as pd
import numpy as np

from MD_Backbone import *
from MD_DB import *
from MD_Align import *
from MD_Output import *

class Connect_BB:
    def __init__(self,DB_Path):
        self.dbs = os.path.join(DB_Path,"ZINC_BB_idx.db")
        try:
            self.conn = sqlite3.connect(self.dbs)
        except:
            print("BB DB Connection Error")
            sys.exit(1)
        self.cursorObj = self.conn.cursor()
    def Fetch_BB_Rings(self,ring,up_mw,low_mw):
        try:
            self.cursorObj.execute("SELECT BB FROM BB_Table WHERE Ring=? AND MW>=? AND MW<=?",(ring,np.float64(low_mw),np.float64(up_mw),))
        except:
            print("Fetch BB Rings Query Error")
            sys.exit(1)
        rows = self.cursorObj.fetchall()
        for i,j in zip(rows,range(len(rows))):
            didi[str(j)] = str(i)
        return didi

    def Fetch_BB_smis(self,smiles,smi_set):
        didi = {}
        roro = set()
        smis = []
        InputCP = Extract_CP(smiles)
        entities = (InputCP["Ring"],InputCP["MW"],InputCP["LogP"],InputCP["TPSA"])
        try:
            self.cursorObj.execute("SELECT BB FROM BB_Table WHERE Ring=? AND MW=? AND LogP=? AND TPSA=?",entities)
        except:
            print("Fetch BB smis Query Error")
            sys.exit(1)

        rows = self.cursorObj.fetchall()

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
                out_smiles = df1[df1['PCScore'] >= 0.98]["SMILES"]
                return out_smiles

    def Fetch_BB_IDS(self,smiles,idid,id_BB):
        try:
            self.cursorObj.execute("SELECT IDs FROM BB_Table WHERE BB=?",(smiles,))
        except:
            print("Fetch BB IDS Query Error")
            sys.exit(1)
        rows = self.cursorObj.fetchall()
        for i in rows:
            for ids in set(unicode.encode(i[0],"utf-8").split(",")):
                iid = ids.strip("\'")
                idid.append(iid)
                id_BB[iid] = smiles

class Connect_PBB:
    def __init__(self,DB_Path):
        self.dbs = os.path.join(DB_Path,"ZINC_PBB_idx.db")
        try:
            self.conn = sqlite3.connect(self.dbs)
        except:
            print("BB DB Connection Error")
            sys.exit(1)
        self.cursorObj = self.conn.cursor()
    def Fetch_BB_Rings(self,ring,up_mw,low_mw):
        try:
            self.cursorObj.execute("SELECT BB FROM PBB_Table WHERE Ring=? AND MW>=? AND MW<=?",(ring,np.float64(low_mw),np.float64(up_mw),))
        except:
            print("Fetch PBB Rings Query Error")
            sys.exit(1)
        rows = self.cursorObj.fetchall()
        for i,j in zip(rows,range(len(rows))):
            didi[str(j)] = str(i)
        return didi

    def Fetch_BB_smis(self,smiles,smi_set):
        didi = {}
        roro = set()
        smis = []
        InputCP = Extract_CP(smiles)
        entities = (InputCP["Ring"],InputCP["MW"],InputCP["LogP"],InputCP["TPSA"])
        try:
            self.cursorObj.execute("SELECT BB FROM PBB_Table WHERE Ring=? AND MW=? AND LogP=? AND TPSA=?",entities)
        except:
            print("Fetch PBB smis Query Error")
            sys.exit(1)

        rows = self.cursorObj.fetchall()

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
                out_smiles = df1[df1['PCScore'] >= 0.98]["SMILES"]
                return out_smiles

    def Fetch_BB_IDS(self,smiles,idid,id_BB):
        try:
            self.cursorObj.execute("SELECT IDs FROM PBB_Table WHERE BB=?",(smiles,))
        except:
            print("Fetch PBB IDS Query Error")
            sys.exit(1)
        rows = self.cursorObj.fetchall()
        for i in rows:
            for ids in set(unicode.encode(i[0],"utf-8").split(",")):
                iid = ids.strip("\'")
                idid.append(iid)
                id_BB[iid] = smiles
class Connect_CP:
    def __init__(self,DB_Path):
        self.dbs = os.path.join(DB_Path,"ZINC_CP_idx.db")
        try:
            self.conn = sqlite3.connect(self.dbs)
        except:
            print("CP DB Connection Error")
            sys.exit(1)
        self.cursorObj = self.conn.cursor()
    def Fetch_CP_Annot(self,ids):
        ids = ids.strip("\'")
        try:
            self.cursorObj.execute("SELECT * FROM CP WHERE ZID=?",(ids,))
        except:
            print("Error ZID : %s"%ids)
            sys.exit(1)
        rows = self.cursorObj.fetchall()
        df = pd.DataFrame(rows)
        col_name = ['ZID','SMILES','MW','LogP','TPSA','RotatableB','HBD','HBA','Ring','Total_Charge','HeavyAtoms','CarBonAtoms','HeteroAtoms','Lipinski_Violation','VeBer_Violation','Egan_Violation','Toxicity']
        df.columns = col_name
        return df
class Connect_PCP:
    def __init__(self,DB_Path):
        self.dbs = os.path.join(DB_Path,"ZINC_PCP_idx.db")
        try:
            self.conn = sqlite3.connect(self.dbs)
        except:
            print("PCP DB Connection Error")
            sys.exit(1)
        self.cursorObj = self.conn.cursor()
    def Fetch_CP_Annot(self,ids):
        ids = ids.strip("\'")
        tmp = []
        try:
            self.cursorObj.execute("SELECT * FROM PCP WHERE ZID=?",(ids,))
        except:
            print("Error ZID : %s"%ids)
            sys.exit(1)
        rows = self.cursorObj.fetchall()
        df = pd.DataFrame(rows)
        if df.empty:
            pass
        else:
            for idx,line in df.iterrows():
                tmp.append("Purchasable")
            df["p"] = tmp
            col_name = ['ZID','SMILES','MW','LogP','TPSA','RotatableB','HBD','HBA','Ring','Total_Charge','HeavyAtoms','CarBonAtoms','HeteroAtoms','Lipinski_Violation','VeBer_Violation','Egan_Violation','Toxicity',"Purchasability"]
            df.columns = col_name
            return df

def Final_Annot(idid,id_BB,pcscore):
    id_df = []
    for ii in idid:
        iidf = cp_dbs.Fetch_CP_Annot(ii)
        iiidf = pcp_dbs.Fetch_CP_Annot(ii)
        
        if iiidf is None or iiidf.empty:
            pass
        else:
            merge_idf = pd.merge(iidf,iiidf[["ZID"]],on="ZID")
            merge_idf["Backbone"] = [id_BB[ii]]
            id_df.append(merge_idf)

    if not id_df:
        return pd.DataFrame()
    else:
        id_df = pd.concat(id_df)
        id_df["PCScore"] = pcscore
        return id_df
def Make_result(a_type,wsmi,file_name,id_set,ncutoff,aBB,InputCP,DB_Path):
    output_path = "./Data/ADC_Output/"
    infile = output_path + file_name + ".total.csv"

    df_list = []
    df_list3 = []
    tier_list = []

    id_BB = {}
    id_smi = {}
    global bb_dbs, cp_dbs, pcp_dbs
    if int(a_type) == 4:
        bb_dbs = Connect_BB(DB_Path)
        cp_dbs = Connect_CP(DB_Path)
    else:
        bb_dbs = Connect_PBB(DB_Path)
        cp_dbs = Connect_PCP(DB_Path)
    pcp_dbs = Connect_PCP(DB_Path)

    wsmi_df = pd.DataFrame.from_dict(InputCP,orient="index").T
    wsmi_df["ZID"] = "* " + file_name
    wsmi_df["Backbone"] = aBB

    df = pd.read_csv(infile)
    smi_list = df["SMILES"].drop_duplicates()

    for smi in smi_list:
        ids = []
        pcscore = np.float64(df[df["SMILES"]==smi]["PCScore"][:1])
        bb_dbs.Fetch_BB_IDS(smi,ids,id_BB)
        annot_df = Final_Annot(ids,id_BB,pcscore)

        if annot_df.empty: pass
        else:
            for i in set(annot_df["ZID"].tolist()):
                id_set.add(unicode.encode(i,"utf-8"))
        df_list.append(annot_df)
        if len(id_set) >= int(ncutoff):
            #df_list.append(annot_df)
            break
    if not df_list:
        print("No Result %s"%file_name)
        return id_set,pd.DataFrame()
    else:
        pass

    try: fin_df = pd.concat(df_list).drop_duplicates()
    except: return set(),pd.DataFrame()

    for i in id_set:
        drop_df = fin_df[fin_df["ZID"]==i].sort_values(by="PCScore",ascending=False)
        df_list3.append(drop_df)
    fin_df = pd.concat(df_list3).sort_values(by="PCScore",ascending=False)
    for idx,line in fin_df.iterrows():
        ids = line["ZID"]
        smi = line["SMILES"]
        id_smi[ids] = smi
        tier_list.append("T 1.5 Inner-Scaffold Search")
	print(len(tier_list))
    print(len(fin_df))
    fin_df["Tier"] = tier_list
    z_pcscore = AlignM3D(file_name,wsmi,id_smi.keys(),id_smi.values())
    z_pcscore.rename(columns={"Query":"ZID","PCScore":"Z_PCScore"},inplace=True)

    fin_df1 = pd.merge(fin_df,z_pcscore[["ZID","Z_PCScore"]],on="ZID").sort_values(by="PCScore",ascending=False).rename(columns={"PCScore":"BB_PCScore"})
    fin_df1 = pd.concat([wsmi_df,fin_df1])
    print(fin_df1)
    if a_type != 4:
        col_name = ["ZID","Z_PCScore","BB_PCScore","MW","LogP","TPSA","RotatableB","HBD","HBA","Ring","Total_Charge","HeavyAtoms","CarBonAtoms","HeteroAtoms","Lipinski_Violation","VeBer_Violation","Egan_Violation","Toxicity","SMILES","Backbone","Purchasability","Tier"]
    elif a_type == 4:
        col_name = ["ZID","Z_PCScore","BB_PCScore","MW","LogP","TPSA","RotatableB","HBD","HBA","Ring","Total_Charge","HeavyAtoms","CarBonAtoms","HeteroAtoms","Lipinski_Violation","VeBer_Violation","Egan_Violation","Toxicity","SMILES","Backbone","Tier"]
    else:
        pass
    fin_df1 =fin_df1[col_name]
    
    fin_df1.to_csv(output_path + file_name + "_" + str(ncutoff) + ".csv",index=False)
    return id_set,fin_df1

def divide_list(ls,n):
	if n == 0:
		num = 1
	else:
		num = n
	for i in range(0,len(ls),num):
		yield ls[i:i+num]

def multi_file_remove_func(a):
	os.remove(a)
