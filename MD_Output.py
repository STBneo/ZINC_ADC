import glob
import sys,os
from time import time
import argparse
import glob
import os
import subprocess
import pickle
from datetime import datetime
import sqlite3

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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from io import BytesIO
from zipfile import ZipFile
import zipfile

from MD_Align import *
from MD_Backbone import *
from MD_DB import *

def Write_Out_Summary(afile,aBB,zzdf):
    ZIDs = zzdf["ZID"][1:]
    Num_ZIDs = len(ZIDs)
    avg_sc = zzdf["BB_PCScore"].dropna().mean()
    t_list = [afile,aBB,Num_ZIDs,avg_sc]
    ddf = pd.DataFrame(t_list).T
    ddf.columns=["File_Name","Backbone","Num_ZIDs","Avg.BB_PCScore"]
    return ddf


def make_img_file(zid,png_path,fin_df):
	# input : DataFrame , ZID , Output Path
    filenumber = fin_df[fin_df["ZID"] == zid].index[0]
    smiles = ''.join(fin_df[fin_df["ZID"] == zid]["SMILES"].tolist())
    pcscore = fin_df[fin_df["ZID"] == zid]["Z_PCScore"]
    cps = Extract_CP(smiles)
    mol = Chem.MolFromSmiles(smiles)
    t_img = Draw.MolToImage(mol,size=(800,800),kekulize=True)


    plt.figure(figsize=(5,5))
    plt.title(smiles,fontsize=7.5,y=-10)
    plt.imshow(t_img,interpolation="nearest",aspect="auto")
    plt.text(10,600,"MW : %f \nTPSA : %f \nLogP : %f \nPCScore : %f"%(cps["MW"],cps["TPSA"],cps["LogP"],pcscore),fontsize=7.5)
    plt.axis('off')
    plt.savefig("%s/%d_%s_%f.png"%(png_path,filenumber,zid,pcscore),dpi=500)

def SDF_load_MainS(zincid,DB_Path):
    try:
        #conn = sqlite3.connect('/lwork02/yklee/1D_Scan.v3/Data/DB_Table/Main_S.db')
        conn = sqlite3.connect(DB_Path + 'Main_S.db')
    except:
        print('DB connect Error')
        sys.exit(1)
    conn.text_factory = str
    cursorObj = conn.cursor()
    try:
        cursorObj.execute('SELECT Sdf FROM Sdf_Files WHERE Zid=?',(zincid,))
    except:
        print('Error %s'%zincid)
        sys.exit(1)
    rows = cursorObj.fetchall()

    if not rows:
        print(zincid)
    else:
        ff = BytesIO(rows[0][0])
        zfp = zipfile.ZipFile(ff,'r')
        zfp_name = zfp.namelist()
        if 'In' in zfp_name[0]:
            zid = zfp_name[0].split('.')[0].split('_')[-1]
        else:
            zid=zincid

        ln_zfp_name = len(zfp_name)
        with open(zid + '.sdf','w') as W:
            W.write(zfp.read(zfp_name[0]))
            W.write('$$$$\n')

