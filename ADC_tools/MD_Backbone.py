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
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

import multiprocessing
from multiprocessing import Process, Manager
from functools import partial
import json
import copy
import re
import pandas as pd 
import traceback

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
import pprint

from MD_DB import *
from MD_Align import *

from networkx.algorithms import isomorphism
import numpy as np
import networkx as nx
import re
java_path = ""
#java_path = "/lwork01/tools/jdk1.8.0_101/bin/"
def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}
                    
def featurize_bonds(mol):
    feats = []
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        btype = bond_types.index(bond.GetBondType())
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([btype, btype])
    return {'type': torch.tensor(feats).reshape(-1, 1).float()}


def find_atom(one_hot):
    atoms=['C',' N',' O',' S',' F',' Si',' P',' Cl',' Br',' Mg',' Na',' Ca',' Fe',' As',' Al',' I',' B',' V',' K',' Tl',' Yb',' Sb',' Sn',' Ag',' Pd',' Co',' Se',' Ti',' Zn',' H',' Li',' Ge',' Cu',' Au',' Ni',' Cd',' In',' Mn',' Zr',' Cr',' Pt',' Hg',' Pb']
    #print(len(atoms))
    #print(len(one_hot))
    a_idx=one_hot.index(True)
    a_type=atoms[a_idx]
    #print (a_idx,a_type)
    return a_type


def find_bond_type(one_hot):
    atoms=['C',' N',' O',' S',' F',' Si',' P',' Cl',' Br',' Mg',' Na',' Ca',' Fe',' As',' Al',' I',' B',' V',' K',' Tl',' Yb',' Sb',' Sn',' Ag',' Pd',' Co',' Se',' Ti',' Zn',' H',' Li',' Ge',' Cu',' Au',' Ni',' Cd',' In',' Mn',' Zr',' Cr',' Pt',' Hg',' Pb']
    #print(len(atoms))
    #print(len(one_hot))
    a_idx=one_hot.index(True)
    a_type=atoms[a_idx]
    #print (a_idx,a_type)
    return a_type


def write_feature(fp_for_out,f_type,*f_list):
   
    #print (f_list,len(f_list))
    #sys.exit(1)

    if(len(f_list)==1 and f_list=='\n'):
        fp_for_out.wirte(f_list)
        return
    else:
        tmp_res=''
        for af_list in f_list:
            #print(af_list)
            #s=[str(i) for i in af_list]
            #res=",".join(s)
            res=','.join(map(str,af_list))
            tmp_res=tmp_res+res+','
        tmp_res=tmp_res+str(f_type)
        #print(tmp_res)
        #sys.exit(1)
        fp_for_out.write(tmp_res+'\n')
            
        return
            

def check_SMILES(asmiles):

    s_flag=True
    Max_Ring=3

    list_find=[]
    c_index=0


    ###################################################################################################
    # Corrct the atom code like 'cu' to 'Cu'
    ###################################################################################################
    asmiles_list=list(asmiles)
    c_index=0
    for achar in asmiles[:-3]:
        if asmiles[c_index]=='[' and asmiles[c_index+3]==']':
            if asmiles[c_index+1].islower():
                tmp_char=asmiles[c_index+1]
                tmp_char=tmp_char.upper()
                asmiles_list[c_index+1]=tmp_char
        c_index+=1

    asmiles="".join(asmiles_list)

    ridx_list=[]
    # Check the ring indexes
    for i in range(1,Max_Ring+1):
        tmp=[]
        tmp=[m.start() for m in re.finditer(str(i),asmiles)]
        if len(tmp)==2:
            ridx_list.append(tmp)

    # Check the ring indexes
    #print asmiles
    #print ridx_list

    list_idx=0
    sets = [set() for _ in xrange(3)]

    #print asmiles
    #print ridx_list

    for alist in ridx_list: 
        if len(alist)==0:
            continue
        #print alist
        s=alist[0]
        e=alist[1]
        idx=1
        while(asmiles[s-1].isdigit()):
            s-=1
        s-=1
        while(asmiles[e-1].isdigit()):
            e-=1
        e-=1
        alist[0]=s
        alist[1]=e
        for i in range(s,e+1):
            sets[list_idx].add(i)

        list_idx+=1

    tmp_set=set()
    tmp_set=sets[0]&sets[1]
    tmp_set.update(sets[1]&sets[2])
    tmp_set.update(sets[0]&sets[2])

    smi_type=0
    if len(tmp_set)==0:
        smi_type=1
    else:
        smi_type=0


    #print asmiles
    #print ridx_list
    #print sets
    #print 'set:',smi_type,tmp_set

    if smi_type==1:
        for alist in ridx_list:
            #print alist,
            if len(alist)>0:
                for i in range(alist[0],alist[1]+1):
                    if((asmiles[i].isalpha() and asmiles[i].islower() and asmiles[i]!='c') or asmiles[i]=='='):
                        #print ' -> PASS: Separated'
                        #print 
                        #name = raw_input('enter key')
                        return True
    if smi_type==0:
        list_idx=0
        for alist in ridx_list:
            go_ok=1
            #print alist,list_idx+1,
            if len(alist)>0:
                stack=[]
                for i in range(alist[0],alist[1]+1):
                    if((asmiles[i].isalpha() and asmiles[i].islower() and asmiles[i]!='c') or asmiles[i]=='='):
                        #print ' -> PASS: Overlapped'
                        #print 
                        #name = raw_input('enter key')
                        return True

                    if(asmiles[i].isdigit() and int(asmiles[i])!=list_idx):
                        c_ring_num=list_idx+1
                        n_ring_num=int(asmiles[i])
                        stack.append(int(asmiles[i]))
                        i+=1
                        while(True):
                            if not asmiles[i].isdigit():
                                # Skip the asmiles[i], Beacuse asmiles[i] is not R_num
                                pass
                            if asmiles[i].isdigit():
                                # In case of closing R_num
                                if int(asmiles[i])==c_ring_num:
                                    break
                                # In case of other R_num
                                if int(asmiles[i])!=c_ring_num:
                                    # In case of other closing R_num, remove other starting R_num
                                    if int(asmiles[i]) in stack:
                                        idx_ch=stack.index(int(asmiles[i]))
                                        stack.remove(int(asmiles[i]))
                                        if len(stack)==0:
                                            break
                                    # In case of other starting R_num, add the starting R_num 
                                    if stack[-1]!=int(asmiles[i]):
                                        stack.append(int(asmiles[i]))
                                    #name = raw_input('enter key')
                            i+=1
            list_idx+=1
       
    # Check fusion scaffold
    list_idx=1
    benz=[]
    for alist in ridx_list:
        if len(alist)>0:
            c_count=0
            for i in range(alist[0],alist[1]+1):
                if asmiles[i].isdigit():
                    continue
                if asmiles[i]=='c':
                    c_count+=1
                if asmiles[i]!='c':
                    break
            if c_count==6:
                benz.append(list_idx)
        list_idx+=1
                
    #print '\nbenz_list:',benz
    #print 'set size: ',len(sets),sets
    
    if '2' in asmiles or '3' in asmiles:
        for rn in benz:
            tmp_list=[1,2,3]
            tmp_list.remove(rn)
            for tmp_i in tmp_list:
                if (len(sets[rn-1])>0 and len(sets[tmp_i-1])>0) and ((sets[rn-1]<=sets[tmp_i-1] or sets[rn-1]>=sets[tmp_i-1])):
                    #print 'set1: ',sets[rn-1],'set2:',sets[tmp_i-1]
                    #print '-> PASS: Fusion'
                    #print 
                    return True

    #print 'set:',smi_type,tmp_set
    #print ' -> Fail'
    #print
    #name = raw_input('enter key')
    #sys.exit(1) 

    #return s_flag
    return False 


def mol_feature_extract2(m):


    ###################################################################################################
    # Corrct the atom code like 'cu' to 'Cu'
    ###################################################################################################
    m_list=list(m)
    c_index=0
    for achar in m[:-3]:
        if m[c_index]=='[' and m[c_index+3]==']':
            if m[c_index+1].islower():
                tmp_char=m[c_index+1]
                tmp_char=tmp_char.upper()
                m_list[c_index+1]=tmp_char
        c_index+=1

    m="".join(m_list)


    SMILES_flag=False
    SMILES_flag=check_SMILES(m)

    if(not SMILES_flag):
        return False

    # Check !!!
    # C1=CN[cu]N1 -> C1=CN[Cu]N1

    #print '1m',m
    #amol = readstring('smi',m)
    #print '2m',m

    try:
        #print m
        amol = readstring('smi',m)
    except:
        return False

    sub_trln_flag=True
    sub_rln_flag=True
    sub_not_benzene_flag=False

    list_ring=amol.sssr
    num_ring = len(list_ring)

    if num_ring>=7:
        sub_trln_flag=False

    for aring in list_ring:
        if aring.Size()>12:
            sub_rln_flag=False
            break
        if aring.GetType()!='benzene' and sub_not_benzene_flag==False:
            sub_not_benzene_flag=True
        
    ok=False 
    if(sub_trln_flag and sub_rln_flag and sub_not_benzene_flag):
        ok=True
    #print ok

    return ok


def mol_feature_extract(m):

    # m: SMILES

    ###################################################################################################
    # Corrct the atom code like 'cu' to 'Cu'
    ###################################################################################################
    m_list=list(m)
    c_index=0
    for achar in m[:-3]:
        if m[c_index]=='[' and m[c_index+3]==']':
            if m[c_index+1].islower():
                tmp_char=m[c_index+1]
                tmp_char=tmp_char.upper()
                m_list[c_index+1]=tmp_char
        c_index+=1

    m="".join(m_list)

    SMILES_flag=False
    SMILES_flag=check_SMILES(m)

    if(not SMILES_flag):
        return False

    #sys.exit(1)

    f_flag=False

    #print 
    print 'smiles:',m
    #mol=Chem.MolFromSmiles(m,sanitize=False)
    mol=Chem.MolFromSmiles(m)
    try:
       m_RingCount = Descriptors.RingCount(mol)
    except:
        print 'RingCount Error: ',m
        return
    #num_SSSR=Chem.GetSSSR(mol)
    num_sSSSR=Chem.GetSymmSSSR(mol)

    sub_rln_flag=True
    sub_idx_flag=True
    sub_het_flag=False 
    sub_bond_flag=False 

    for anum_sSSSR in num_sSSSR:

        ##### 1.Check the length of RING ####
        #print list(anum_sSSSR)
        if len(list(anum_sSSSR))>=7:
        #if len(list(anum_sSSSR))>=7 or len(list(anum_sSSSR))<=3:
            #print list(anum_sSSSR)
            sub_rln_flag=False
            break

        idxs = list(anum_sSSSR)
        #print idxs,len(idxs)

        i=0
        ln_idxs=len(idxs)

        ##### 2.Check the index for long RING ####
        Diff_Index=12 
        for idx in idxs:
            #print idxs[i],idxs[i+1],abs(idxs[i]-idxs[i+1])
            if i<ln_idxs-1:
                #print idxs[i],idxs[i+1],abs(idxs[i]-idxs[i+1])
                if abs(idxs[i]-idxs[i+1])>Diff_Index:
                    sub_idx_flag=False
                    break
                    
            if i==ln_idxs-1:
                #print idxs[i],idxs[0],abs(idxs[i]-idxs[0])
                if abs(idxs[i]-idxs[0])>Diff_Index:
                    sub_idx_flag=False
                    break
            i+=1
       

        ##### 3.Check the hetero ATOM  ####
        for idx in idxs:
            atom=mol.GetAtomWithIdx(idx).GetSymbol()
            #print 'index:',int(idx),'atom:',atom
            #### Check hetero atom!!!!!
            if atom=='N' or atom=='O' or atom=='S':
                sub_het_flag=True
                break


        ##### 4.Check the AROMATIC bond #### 
        i=0
        for idx in idxs:
            #### Check aromatic or bouble bond!!!!!
            #print sub_flag
            if i+1<ln_idxs:
                #print i,i+i,idxs[i],idxs[i+1]
                btype=mol.GetBondBetweenAtoms(idxs[i],idxs[i+1]).GetBondType()
                #print btype 
                #if btype=='AROMATIC' or btype=='DOUBLE':
                #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                if (btype==Chem.rdchem.BondType.AROMATIC):
                    #print 'AROMATIC'
                    sub_bond_flag=True
                    break
            if i==ln_idxs-1:
                #print i,'0',idxs[i],idxs[0]
                btype=mol.GetBondBetweenAtoms(idxs[i],idxs[0]).GetBondType()
                #print btype
                #Chem.rdchem.BondType.AROMATIC
                #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                if (btype==Chem.rdchem.BondType.AROMATIC):
                #if btype=='AROMATIC' or btype=='DOUBLE':
                    #print 'AROMATIC'
                    sub_bond_flag=True
                    break
            i+=1


        ##### 5.Check Donor or Acceptor #### 
        '''
        for idx in idxs:
            fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
            feats = factory.GetFeaturesForMol(mol)
            print feats
            print len(feats)
            print feats[0].GetFamily()
            print feats[1].GetFamily()
            #print feats[2].GetFamily()
            sys.exit(1)
            #### Check hetero atom!!!!!
        '''



    #print 'sub_flag:',sub_flag
    m_NumAromaticCarbocycels = Descriptors.NumAromaticCarbocycles(mol)
    m_NumSaturatedCarbocycles = Descriptors.NumSaturatedCarbocycles(mol)


    # ===================================
    sub_AC_flag=True 
    if(m_RingCount == m_NumAromaticCarbocycels or m_RingCount == m_NumSaturatedCarbocycles): 
        sub_AC_flag=False 
    # ===================================

    '''
    # Only C and H
    # Non-aromatic compound
    m_NumAromaticRings = Descriptors.NumAromaticRings(mol)
    # Aromatics
    m_NumAliphaticRings = Descriptors.NumAliphaticRings(mol)

    # Important Rule
    # Must include rule_1
    m_NumAromaticHeterocycels = Descriptors.NumAromaticHeterocycles(mol)

    # Must exclude rule_2
    m_NumAromaticCarbocycels = Descriptors.NumAromaticCarbocycles(mol)

    # Must exclude rule_3
    m_NumSaturatedHeterocycles = Descriptors.NumSaturatedHeterocycles(mol)

    # Must exclude rule_3
    m_NumSaturatedCarbocycles = Descriptors.NumSaturatedCarbocycles(mol)

    print m
    print 'm_SSSR: ', num_SSSR
    print 'm_RingCount: ', m_RingCount
    print 'm_NumAromaticRings: ',m_NumAromaticRings
    print 'm_NumAliphaticRings: ',m_NumAliphaticRings
    print 'm_NumAromaticHeterocycels: ',m_NumAromaticHeterocycels
    print 'm_NumAromaticCarbocycels: ',m_NumAromaticCarbocycels
    print 'm_NumSaturatedHeterocycles: ',m_NumSaturatedHeterocycles
    print 'm_NumSaturatedCarbocycles: ',m_NumSaturatedCarbocycles

    if (m_RingCount==m_NumAromaticHeterocycels):    
        f_flag=False
    else:
        f_flag=True
     


    raw_input("Press Enter to continue...")
    '''
    '''
    sub_rln_flag=True
    sub_idx_flag=True
    sub_het_flag=False 
    sub_bond_flag=False 
    '''
    '''
    print 'rln: ',sub_rln_flag
    print 'dix: ',sub_idx_flag
    print 'het: ',sub_het_flag 
    print 'bond:',sub_bond_flag
    print 
    '''

    ok=False
    #if(sub_rln_flag and sub_idx_flag and sub_het_flag and sub_bond_flag):
    if(sub_rln_flag and sub_idx_flag and sub_het_flag and sub_bond_flag and sub_AC_flag and SMILES_flag):
        ok=True

    return ok
    #return sub_flag



def mol_scaffold_extract(lid,amol):

    
    # making smiles
    t_dir='./Re_BackB/sng_tmp/'
    smiles=Chem.MolToSmiles(amol)
    # print smiles
    f_name=lid+'.smi'

    # write smiles
    t_path=os.path.join(t_dir,f_name)
    fp_for_out=open(t_path,'w')
    fp_for_out.write(smiles)
    fp_for_out.close()

    # excuate sng
    FNULL = open(os.devnull, 'w')
    subprocess.call(['%sjava'%java_path,'-jar','sng.jar','generate','-o', t_dir+lid+'.tmp',t_dir+f_name], stdout=FNULL, stderr=subprocess.STDOUT)

    # read the sng output
    re_name=lid+'.tmp'
    t_path=os.path.join(t_dir,re_name)
    fp_for_in=open(t_path,'r')
    lines=fp_for_in.readlines()
    fp_for_out.close()
    #print lines
    #print 'HIHIHIHI'

    scaffold_list=[]
    for aline in lines: 
        aline=aline.strip()
        if 'RING' not in aline:
            #print aline
            ascaffold=processing_line(aline)
            if ascaffold is not None: 
                #print ascaffold
                scaffold_list.append(ascaffold)

    #print scaffold_list
    #sys.exit(1)
    return scaffold_list


def mol_scaffold_backbone_extract(lid,amol):
    
    # making smiles
    t_dir='./Data/BB/sng_tmp/'
    #smiles=Chem.MolToSmiles(amol,sanitize=False)

    smiles=Chem.MolToSmiles(amol)

    #print smiles
    f_name=lid+'.smi'

    # write smiles
    t_path=os.path.join(t_dir,f_name)
    fp_for_out=open(t_path,'w')
    fp_for_out.write(smiles)
    fp_for_out.close()

    # excuate sng
    FNULL = open(os.devnull, 'w')
    subprocess.call(['%sjava'%java_path,'-jar','./Data/Sub_P/sng.jar','generate','-o', t_dir+lid+'.tmp',t_dir+f_name], stdout=FNULL, stderr=subprocess.STDOUT)
    #process.wait()

    # read the sng output
    re_name=lid+'.tmp'
    t_path=os.path.join(t_dir,re_name)
    fp_for_in=open(t_path,'r')
    lines=fp_for_in.readlines()
    fp_for_out.close()
    #print lines
    #print 'HIHI'
        
    #print smiles
    #########################################
    # Extract Backbone
    scaffold_backbone=''
    ln_lines=len(lines)
    #print ln_lines

    #####################################################################
    # If there is no scaffolds, The Original structure is used as backbone.
    if ln_lines==1:
        scaffold_backbone = smiles
    #####################################################################

    if ln_lines>1:
        last_line=lines[ln_lines-1]
        #print 'Test ',last_line
        #scaffold_list.append(last_line)

        if ln_lines>2:
            token=last_line.split(',')
            #print 'last: ',token[-2].strip()
            scaffold_backbone = token[-2].strip()
        if ln_lines==2:
            token=last_line.split()
            #print token
            #print token[-1]
            #print 'last: ',token[-1][:-1].strip()
            scaffold_backbone = token[-1][:-1].strip()
    #print 'Backbone: ',scaffold_backbone        
    ########################################
    #print lid,smiles
    #######################################
    # Extract scaffolds
    scaffold_list=[]
    for aline in lines: 
        aline=aline.strip()
        if 'RING' not in aline:
            #print aline
            #print smiles
            ascaffold=processing_line(aline)
            if ascaffold is not None: 
                #print ascaffold
                scaffold_list.append(ascaffold)

    #######################################
    #print scaffold_list
    #sys.exit(1)

    '''
    if len(scaffold_list)==0:
        return scaffold_list
    else:
        return scaffold_backbone_list
    '''

    return scaffold_list, scaffold_backbone



def processing_line(aline):

    Max_RING=3

    token=aline.split('\t')
    #print int(token[0])
    scaffold=None

    if int(token[0])<=Max_RING:
    #if int(token[0])==3: 
        #print token[0]
        f_flag=0
        # apply reult 1
        '''
        if 'N' in token[1]:
            f_flag=1
        if 'O' in token[1]:
            f_flag=1
        if 'S' in token[1]:
            f_flag=1
        '''
        # apply the rule 
        #if mol_feature_extract(token[1]):
        #print aline
        if mol_feature_extract2(token[1]):
            f_flag=1

        if f_flag==1:
            #print token[1]
            scaffold=token[1]


    return scaffold 


def print_date_time():

    now = datetime.now()
    w=datetime.now()
    dt_string=now.strftime("%d/%m/%Y %H:%M:%S")
    print 'Date and Time: ',dt_string




def check_one_ring(m):
    
    # m: smiles string
    mol=Chem.MolFromSmiles(m)
    try:
        m_RingCount = Descriptors.RingCount(mol)
        return m_RingCount
    except:
        print 'Something wrong in RingCount'
        return -1


def draw_sdf(lid,i_path,smi):


    t_dir='./Data/BB/IMG/'

    f_name=lid+'.sdf'
    t_path=os.path.join(i_path,f_name)

    smi_name=lid+'.smi'
    smi_path=os.path.join(i_path,smi_name)
    
    #print t_path,smi_path
    arg='obabel '+t_path+' -O '+smi_path
    os.system(arg)

    org_f_name=lid+'.1O.png'
    org_img_path = os.path.join(t_dir,org_f_name)
    arg='obabel '+smi_path+' -O '+org_img_path+' -xw 350 -xh 250 -d'
    #print arg
    os.system(arg)

    BB_f_name=lid+'.2B.png'
    BB_img_path = os.path.join(t_dir,BB_f_name)
    arg='obabel -:"'+smi+'" -O '+BB_img_path+' -xw 350 -xh 250 -d'
    #print arg
    os.system(arg)


    arg='rm '+smi_path
    os.system(arg)


    return



def draw_sdf2(i_path,backbone_dic,lid):


    smi=backbone_dic[lid]
    t_dir='./Data/BB/IMG/'
    f_name=lid+'.sdf'
    t_path=os.path.join(i_path,f_name)

    smi_name=lid+'.smi'
    smi_path=os.path.join(i_path,smi_name)
    
    #print t_path,smi_path
    arg='obabel '+t_path+' -O '+smi_path
    #os.system(arg)
    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)

    org_f_name=lid+'.1O.png'
    org_img_path = os.path.join(t_dir,org_f_name)
    arg='obabel '+smi_path+' -O '+org_img_path+' -xw 350 -xh 250 -d'
    #print arg
    #os.system(arg)
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)

    BB_f_name=lid+'.2B.png'
    BB_img_path = os.path.join(t_dir,BB_f_name)
    arg='obabel -:"'+smi+'" -O '+BB_img_path+' -xw 350 -xh 250 -d'
    #print arg
    #os.system(arg)
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)

    arg='rm '+smi_path
    #os.system(arg)
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)


    return


def make_mol2(t_dir,backbone_dic,lid):

    smi=backbone_dic[lid]
    t_id=lid 
    t_smiles=smi 
    f_name=t_id+'.BB.mol2'
    t_path=os.path.join(t_dir,f_name)
    arg='obabel -:"'+t_smiles+'" --gen2D --addtotitle '+t_id+'.BB'+' -O '+t_path
    #os.system(arg)
    FNULL = open(os.devnull, 'w')
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)

    return


def  Extract_Mol():

    time1=time()

    parser=argparse.ArgumentParser()
    parser.add_argument('-input',required=True, default=100, help='Input fold.')
    parser.add_argument('-top_p',required=True, default=0, help='Select the top X percent of scaffod frequency.')
    parser.add_argument('-mins',required=True, default=1, help='The minimum number of scaffold.')
    parser.add_argument('-img',required=True, choices=['y','n'], default='n',  help='Making the png files')
    args=parser.parse_args()
    i_path=args.input

    if not os.path.exists('./Data/Sub_P/sng.jar'):
        print 'This python program needs \'sng.jar\'.'
        print 'Move the \'sng.jar\' program at this fold and re excute this program.'
        os.exit(1)
    
    FNULL = open(os.devnull, 'w')

    t_dir='./Data/BB/sng_tmp/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        #os.system(arg)
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        process.wait()
        #shutil.rmtree(l_base)

    t_dir='./Data/BB/SCF/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        #os.system(arg)
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        process.wait()
        #shutil.rmtree(l_base)

    t_dir='./Data/BB/IMG/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        #os.system(arg)
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        process.wait()

    t_dir='./Data/BB/BackBone/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        #os.system(arg)
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        process.wait()


    #print i_path 
    f_list=glob.glob(i_path+'*.sdf')
    f_list.sort()
    #print(f_list)
    #print(f_list)
    #print(len(f_list))
    ln_f=len(f_list)


    # Input numnber check
    if ln_f==0:
        print 'No sdf input'
        print 'Check.............'

    # Extract the chemical property of input templates 
    print 'Step 1. Start Extracting input chmeical feature............'
    m_type='sdf'
    df_input_CF = Extract_Chemical_Feature(i_path,m_type)
    df_input_CF.to_csv('./Data/BB/Chmeical_Feature_Input.csv',sep=',',index=False)
    print ' --> End Extracting input chmeical feature............\n'

    #print df
    #return

    tmp_dic={}
    tmp_set=set()
    #return

    manager = Manager()
    Re_dic = manager.dict()

    id_list = manager.list()
    sid_list = manager.list()
    tmp_dic  = manager.dict()
    backbone_dic  = manager.dict()

    SMILES_dic = manager.dict()
    smi_list = manager.list()

    #print type(Re_dic)
    #print type(SMILES_dic)
    #print Re_dic2
    #sys.exit(1)

    #print ln_f,f_list
    #return

    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    #print Num_Of_CPU
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(core_make,sid_list,backbone_dic,ln_f,f_list,tmp_dic,id_list)
    pool.map(func,f_list)
    pool.close()
    pool.join()


    #print tmp_dic
    #return

    tmp_dic=dict(tmp_dic)
    #print tmp_dic
    #sys.exit(1)

    lst_key=tmp_dic.keys()
    #print 'befor:',len(lst_key)
    #return
   
    # Save the statics file
    lst_df=[]
    lst_key=tmp_dic.keys()
    for akey in lst_key:
        values=str(tmp_dic[akey])
        tmp=[]
        token=values.split(',')
        ln_token=len(token)
        tmp.append(akey)
        tmp.append(ln_token)
        lst_df.append(tmp)

    df=pd.DataFrame(lst_df,columns=['smiles','count'])
    df['Rank'] = df['count'].rank(ascending=False,method='min')
    df['Rank_pct'] = df['count'].rank(pct=True)
    #print df
    df=df.sort_values(by=['count'],ascending=False)

    st_dir='./Data/BB/Input_Scaffold_Statics/'
    if not os.path.exists(st_dir):
        os.makedirs(st_dir)
    else:
        pass
        #arg='rm '+st_dir+'*'
        #os.system(arg)
        #shutil.rmtree(l_base)
    df.to_csv('./Data/BB/Input_Scaffold_Statics/Scaffold_Statics.csv',sep=',',index=False)
    #print df

    #print args.top_p, args.mins
    df_selected=df.loc[(df['Rank_pct'] >= float(args.top_p)) & (df['count']>= int(args.mins))]
    # temparay blocked
    #print df_selected

    # Select 
    #df_tmp=df.sort_values(by='count', ascending=False).head(3)
    #print df_tmp

    #print SMILES_dic
    #sys.exit(1)


    # For the fold for scaffolds decomposition output which is input for Scaffold DB searching 
    st_dir='./Data/BB/SCF/'
    if not os.path.exists(st_dir):
        os.makedirs(st_dir)
    else:
        arg='rm '+st_dir+'*'
        #os.system(arg)
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        #shutil.rmtree(l_base)

    Scaffold_id=0
    Scaffold_id_f=0
    for index,row in df_selected.iterrows():
        #print(index,row['smiles'])
        t_smiles=row['smiles']
        #sid=str(Scaffold_id)
        #sid.zfill(3)
        #print sid
        sid=format(Scaffold_id,'03')
        #print sid
        Scaffold_id+=1
        f_name='Scaffold.'+str(sid)+'.smi'
        f_path=st_dir+f_name
        #print f_path
        fp_for_out=open(f_path,'w')
        fp_for_out.write(t_smiles)
        fp_for_out.close()
  

    # Draw the 2D PNG
    backbone_dic_p=dict(backbone_dic)
    #print backbone_dic
    t_dir='./Data/BB/IMG/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        #os.system(arg)
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
    img_flag = args.img
    k_list=backbone_dic_p.keys()
    if img_flag=='y':
        #k_list=backbone_dic_p.keys()
        pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
        func=partial(draw_sdf2,i_path,backbone_dic)
        pool.map(func,k_list)
        pool.close()
        pool.join()


    # wirte sid_list
    fp_for_out=open('./Data/BB/Processed_ID_list.txt','w')
    fp_for_out.write('ID\n')
    for aele in sid_list:
        fp_for_out.write(aele+'\n')
    fp_for_out.close()
    #df=pd.DataFrame(list(tmp_dic.items()),columns=['smiles', 'ids'])

    # write backbone list
    df=pd.DataFrame(list(backbone_dic_p.items()),columns=['IDs','Backbone_smiles'])
    df.to_csv('./Data/BB/BackBone_list.csv',sep=',',index=False)

    #print backbone_dic
    # write backbone sdf files 
    t_dir='./Data/BB/BackBone/'
    #k_list=backbone_dic_p.keys()
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(make_mol2,t_dir,backbone_dic)
    pool.map(func,k_list)
    pool.close()
    pool.join()


    # Clean the sng_tmp folder
    t_dir='./Data/BB/sng_tmp/'
    arg='rm '+t_dir+'*'
    #os.system(arg)
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)


    #print '\nThe Num. of total input: ', len(f_list)
    print '\n --> Processed the number of scaffold: ', len(sid_list)
    #print '  --> Not processed the number of scaffold: ',len(f_list)-len(sid_list)

    # remove Input_SCF contents before use!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!         
    arg='rm  ./Re_BackB/sng_tmp/*'
    #os.system(arg)
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)

    #print '  --> The length of library: '+str(len(tmp_dic))+'\n'
    time2=time()
    print ' Extracting input chmeical feature time: '+str('{:.2f}'.format(time2-time1))+' sec.'
    #print_date_time()
    print '\n'

    return



def core_make(sid_list,backbone_dic,ln_mols,f_list,tmp_dic,id_list,afile):
   
    #print afile
    aidx=f_list.index(afile)
    e_flag=0

    #print '\n\nProcessing....:'+str(aidx+1)+'/'+str(ln_mols)+' ,mol:'+afile[:-4] 

    print ' --> Processing....:'+str(aidx+1)+'/'+str(ln_mols)+' ,mol:'+afile[:-4]+'\r', 
    sys.stdout.flush()

    try:
        #print afile
        mols = Chem.SDMolSupplier(afile,sanitize=False)
        #mols.UpdatePropertyCache(strict=False)
        #Chem.SanitizeMol(mols,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)
        #print afile
    except:
        print(afile,'Error in RDKit')
        e_flag=1

    #amol=mols[0]
    #return

    #print afile

    aidx=f_list.index(afile)
    if(e_flag==0):
        mol_count=0
        for amol in mols:
            ln_2mols=len(mols)
            scaffold_list=[]
            afile_f=os.path.basename(afile)
            lid = os.path.splitext(afile_f)[0]
            try:
                #asf=Chem.MolToMolBlock(amol)
                #print afile,amol
                token=Chem.MolToMolBlock(amol).split('\n')
                #print token
                #sys.exit(1)
            except:
                print 'In except:',afile
                continue
            #sys.exit(1)
            #print token
            #lid=afile.split('/')[-1].split('.sdf')[0]
            #print(afile.split('/')[-1].split('.sdf')[0])
            #print '   Sub_processing....:'+str(mol_count+1)+'/'+str(ln_2mols)+' ,mol:'+lid
            mol_count+=1
            #print lid
            #sys.exit(1)

            scaffold_list, scaffold_backbone = mol_scaffold_backbone_extract(lid,amol)
            #scaffold_list = mol_scaffold_extract(lid,amol)

            #print lid

            #print 'scaffold_list: ',scaffold_list
            #print 'scaffold_backbone: ',scaffold_backbone
            #sys.exit(1)

            # Only processing the backbone: not check the backbone as scaffold
            scaffold_backbone = processing_backbone(scaffold_backbone)
            #print 'Backbone: ',scaffold_backbone
            if lid not in backbone_dic:
                backbone_dic[lid]=scaffold_backbone
            else:
                print 'ID duplication'

            ln_scaffold_list=len(scaffold_list)
            # making dic
            # If vaild ligand, then processing the scaffold information
            if ln_scaffold_list>0:

                '''
                scaffold_backbone = processing_backbone(scaffold_backbone)
                if lid not in backbone_dic:
                    backbone_dic[lid]=scaffold_backbone
                else:
                    print 'ID duplication'
                '''

                sid_list.append(lid)

                for a_sf in scaffold_list:
                    if a_sf in tmp_dic:
                        id_list=[]
                        id_list=tmp_dic[a_sf]
                        id_list.append(lid)
                        tmp_dic[a_sf]=id_list
                        #tmp_dic[a_sf].append(lid)
                    else:
                        id_list=[]
                        id_list.append(lid)
                        tmp_dic[a_sf]=id_list
            #print tmp_dic    
            #sys.exit(1)
    else:
        pass
    #print 
    #print len(tmp_dic),tmp_dic
    #print 'Processing....:'+str(aidx+1)+'/'+str(ln_mols)+' ,mol:'+afile[:-4]+'\n' 
    
    return



#func=partial(core,Re_dic,mols,ln2_mols)
def core(db_dic,Re_dic,tmp_list,SMILES_dic,id_list,f_list,ln_f,amfile):

    tmp_list=[]
    aidx=f_list.index(amfile)
    #print amfile
    #print aidx

    try:
        mols = Chem.SDMolSupplier(amfile)
    except:
        print 'Error in reading file RDKit'
        return
    #print amol
    amol=mols[0]

    scaffold_list=[]
    try:
        token=Chem.MolToMolBlock(amol).split('\n')
    except:
        print 'Erro in parsing the RD object'
        return
    lid=token[0]
    #print 'Searching....:'+str(aidx+1)+'/'+str(ln_f)+' ,mol:'+lid
    #print lid
    #sys.exit(1)

    scaffold_list = mol_scaffold_extract(lid,amol)
    #print scaffold_list
    #return
    #sys.exit(1)

    ln_scaffold_list=len(scaffold_list)
    tmp_list.append(ln_scaffold_list)
    # making dic
    Re_dic[lid]=[]
    #print Re_dic
    #sys.exit(1)
    f_count=0
    tmp_lists2=[]
    tmp_lists3=[]

    #print type(SMILES_dic)
    #print type(id_list)

    if ln_scaffold_list>0:
        for a_sf in scaffold_list:
            if a_sf in db_dic:
                #print 'F',a_sf
                tmp_lists2.append(a_sf)
                f_count+=1

                if a_sf in SMILES_dic:
                    id_list=[]
                    id_list=SMILES_dic[a_sf]
                    if lid not in id_list:
                        id_list.append(lid)
                        SMILES_dic[a_sf]=id_list
                    else:
                        pass
                    
                else:
                    id_list=[]
                    id_list.append(lid)
                    SMILES_dic[a_sf]=id_list
                    #print SMILES_dic[a_sf] 
               

    tmp_list.append(f_count)
    tmp_list.append(tmp_lists2)
    Re_dic[lid]=tmp_list

    print 'Searching....:'+str(aidx+1)+'/'+str(ln_f)+' ,mol:'+lid+' ,Num. of scaffold:'+str(len(scaffold_list))+',Found Scaffold: '+str(f_count)
    #print Re_dic
    #sys.exit(1)
    
    return



def search_db(i_path):

    time1=time()

    # load scaffold_db
    with open('scaffold_db.pickle', 'rb') as handle:
        db_dic = pickle.load(handle)

    print 'Loading the scaffold db file'
    print 'The size of db :',len(db_dic),'\n'

    f_list=glob.glob(i_path+'*.sdf')
    f_list.sort()

    if len(f_list)==0:
        print 'No sdf files, check the path!'
        sys.exit(1)

    #print(f_list)
    #print(len(f_list))
    #sys.exit(1)
    ln_f=len(f_list)

    manager = Manager()
    Re_dic = manager.dict()
    tmp_list = manager.list()

    SMILES_dic = manager.dict()
    smi_list = manager.list()

    #print type(Re_dic)
    #print type(SMILES_dic)
    #print Re_dic2
    #sys.exit(1)
    
    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    print Num_Of_CPU
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(core,db_dic,Re_dic,tmp_list,SMILES_dic,smi_list,f_list,ln_f)
    pool.map(func,f_list)
    pool.close()

    #print tmp_dic
    #print Re_dic
    #print SMILES_dic

    # Test Code
    '''
    smiles_d=dict(SMILES_dic)
    for key,values in smiles_d.iteritems():
        if len(values)>0:
            print key,values
    '''

    #sys.exit(1)

    #print Re_dic
    ###############################
    # For summary file output
    ###############################
    d=dict(Re_dic)
    #print 'The total inputed sdf is: ',len(Re_dic),'\n'
    print
    s_count=0
    f_count=0
    fp_for_out=open('Re_scaffold.txt','w')
    fp_for_out.write('Ligand ID\tNum. of Scaffold\tNum. of found scaffold\tSMILES'+'\n')
    for key,value  in d.iteritems():
        #print 'Ligand ID:',key,value,',The num of retrieval scaffold: ',str(len(value[1]))
        #print 'Ligand ID:',key,value,',The num of retrieval scaffold: ',str(len(value[1]))
        #fp_for_out.write(key+','+str(value[0])+','+str(value[1])+'\n')
        #print value[2]
        #res=",".join(s)
        if len(value[2])>0:
            res=",".join(value[2])
        else:
            res=''
        #print res
        #print 'Ligand ID: '+key+', Num. of Scaffold: '+str(value[0])+', Num. of found scaffold: '+str(value[1])+', Smiles: '+res
        fp_for_out.write(key+'\t'+str(value[0])+'\t'+str(value[1])+'\t'+res+'\n')
        if value[0]>0:
            f_count+=1
        if value[1]>0:
            s_count+=1
    fp_for_out.close()


    ##################################
    # For SMILES string file output
    #################################
    smiles_d=dict(SMILES_dic)
    fp_for_out=open('Re_scaffold_SMILE.txt','w')
    fp_for_out.write('SMILES\tNum. of ids\tLigand IDs'+'\n')
    for key,value  in smiles_d.iteritems():
        if len(value)>0:
            res=",".join(value)
        else:
            res=''
        fp_for_out.write(key+'\t'+str(len(value))+'\t'+res+'\n')
    fp_for_out.close()

    print '\nThe size of library: ',len(db_dic)
    print 'The total inputed sdf: ',len(Re_dic)
    print 'The num. for the input having scaffold: ',f_count 
    print 'The num. for the input matching scaffold of library: ',s_count,'('+str(float(s_count)/float(f_count)*100),'%)','\n'
    
    time2=time()
    print 'Processing time: '+str(time2-time1)
    print_date_time()
    print

    '''
    #save the dic
    with open('scaffold_db.pickle', 'wb') as handle:
        pickle.dump(tmp_dic,handle,protocol=pickle.HIGHEST_PROTOCOL)
    '''

    return

def core_CF(m_type,Re_dic,Ids_list,amol):

    
    #mol=Chem.MolFromSmiles(m)
    #mol=Chem.MolFromMol2File(amol)
    #smi=Chem.MolToSmiles(mol)

    amol_f=os.path.basename(amol)
    #print amol_f

    #return
   
    if m_type=='mol2':
        amol =readfile('mol2',amol).next()
    if m_type=='sdf':
        amol =readfile('sdf',amol).next()

    desc = amol.calcdesc(descnames=['MW', 'logP', 'HBA1', 'HBD','TPSA'])
    #print desc
    #t = amol.data
    #print t
    t= amol.sssr
    #print t
    Num_Ring = len(t)

    #print desc['MW']

    Re_dic={}
    Re_dic={'ID':amol_f}
    #Re_dic.update({'MW':desc['MW']}) 
    Re_dic.update({'MW':'{:.1f}'.format(desc['MW'])}) 
    #Re_dic.update({'LogP':desc['logP']}) 
    Re_dic.update({'LogP':'{:.1f}'.format(desc['logP'])}) 
    Re_dic.update({'HBA':desc['HBA1']}) 
    Re_dic.update({'HBD':desc['HBD']}) 
    #Re_dic.update({'TPSA':desc['TPSA']}) 
    Re_dic.update({'TPSA':'{:.1f}'.format(desc['TPSA'])}) 
    Re_dic.update({'Ring':Num_Ring}) 

    Ids_list.append(Re_dic)

    return


def Extract_Chemical_Feature(t_path,m_type):

    time1=time()
    print 'Step 2. Starting extracting chemical feature..............'


    if m_type=='mol2':
        f_list=glob.glob(t_path+'*.mol2')
    if m_type=='sdf':
        f_list=glob.glob(t_path+'*.sdf')

    '''
    # for serial P
    for amol in f_list:
        core_CF(amol)
    return
    '''
    #print f_list
    #return

    manager = Manager()
    Re_dic = manager.dict()
    Ids_list = manager.list()

    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    #print Num_Of_CPU
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(core_CF,m_type,Re_dic,Ids_list)
    pool.map(func,f_list)
    pool.close()
    pool.join()

    #print Ids_list
    #df = pd.DataFrame.from_dict(Ids_list)
    df = pd.DataFrame.from_records(Ids_list,columns=['ID','MW','LogP','HBA','HBD','TPSA','Ring'])
    #print df

    time2=time()
    #print 'Extracting chemical feature time: '+str(time2-time1)+'\n'
    print ' Extracting chemical feature time: '+str('{:.2f}'.format(time2-time1))+' sec.\n'

    return df
   

def Query_ScaffoldDB(in_file):


    df = pd.read_csv(in_file)
    slist=df['Backbone'].tolist()
    #print slist

    #Test_Test
    #slist=[]
    #slist.append('c1ccc2OCC(c2(c1))[N+]CCc3nccnc3')

    manager = Manager()
    Re_dic = manager.dict()
    Re_list = manager.list()

    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    #print Num_Of_CPU
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    #func=partial(Query_Fatch,ln_tmp_re_list,tmp_re_list,Re_dic,Re_list,OID)
    func=partial(Query_Fatch,Re_dic,Re_list)
    pool.map(func,slist)
    #pool.map(func,slist[0:20])
    pool.close()
    pool.join()

    df=pd.DataFrame(list(Re_dic.items()),columns=['Backbone', 'ID_Count'])
    #df = pd.DataFrame.from_records(Re_list,columns=['Backbone','ID_Count'])
    df=df.sort_values(by=['ID_Count'],ascending=False)
    out_f=in_file+'.Count.csv'
    #out_f_path=os.path.join(out_path,out_f)
    df.to_csv(out_f,index=False)


    return


def Query_Fatch(Re_dic,Re_list,asmi):


    con = sql_connection_existing_ZINC_SCF()
    if con != -1:
        cursorObj = con.cursor()
        cursorObj.execute('select SMILES,ids from SCF_Table where SMILES = ?',[asmi])
        rows = cursorObj.fetchall()

        #Re_dic={}
        if len(rows)>0:
            re_list = list(rows[0])
            ln_re_list=len(re_list)
            asmiles=re_list[0]
            IDs=re_list[1]
            token=IDs.split(',')
            #if len(IDs)>0:
            print asmi,len(token)
            #Re_dic={'Backbone':asmi}
            #Re_dic.update({'Count':len(token)}) 
            Re_dic[asmi]=len(token)
        else:
            print asmi,'0'
            #Re_dic={'Backbone':asmi}
            #Re_dic.update({'Count':0}) 
            Re_dic[asmi]=0 
        #Re_list.append(Re_dic)

    return


def Query_DB(df):
   

    FNULL = open(os.devnull, 'w')
    
    t_dir='./Data/3DAlign/Final_ADC_PDB/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        process.wait()


    time1=time()
    print 'Step 3. Starting querying and LSalign..............'

    out_path='./Data/3DAlign/Final_ADC_list/'
    con = sql_connection_existing_Main_FAF()
    #print df
    # df = pd.DataFrame.from_records(Ids_list,columns=['ID','MW','LogP','HBA','HBD','TPSA','Ring'])
    for index,row in df.iterrows():
        MW = row['MW']
        ring = row['Ring']
        #print MW,ring
        time_1=time()
        print ' --> Fetching ADC like.........: ', row['ID']
        re_list=sql_fetch_FAF(con,MW,ring)
        print ' --> Finishing searching ADC'
        print ' --> Retrieval ligand : '+str(len(re_list))
        time_2=time()
        print ' --> Fetching from FAF DB time: '+str('{:.2f}'.format(time_2-time_1))+' sec.\n'

        tmp_re_list=[]
        for aligand in re_list:
            tmp_re_list.append(aligand[0])
        ln_tmp_re_list=len(tmp_re_list)
        OID=row['ID']

        '''
        path_template = './Data/BB/BackBone/'
        t_fname=OID
        path_t=os.path.join(path_template,t_fname)
        fp_for_in=open(path_t,'r')  
        lines=fp_for_in.readlines()
        fp_for_in.close()
        new_str=''.join(lines)
        arg_t=To_ARG_str(new_str)
        '''

        #print arg_t
        #return

        print ' --> Fetching SDF and LSAlign......... '
        time_1=time()
        manager = Manager()
        Re_dic = manager.dict()
        Re_list = manager.list()

        Num_Of_CPU=multiprocessing.cpu_count()
        #Num_Of_CPU=2
        #print Num_Of_CPU
        pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
        func=partial(sql_fetch_Main_S_Check_PCscore,ln_tmp_re_list,tmp_re_list,Re_dic,Re_list,OID)
        # Test-Test
        #pool.map(func,tmp_re_list[0:5])
        pool.map(func,tmp_re_list)
        pool.close()
        pool.join()

        time_2=time()
        print '\n --> Fetching and alignment time: '+str('{:.2f}'.format(time_2-time_1))+' sec.'

        #print Re_list
        #print len(tmp_re_list)
        #tmp_Re_list=list(Re_list)
        #df=pd.DataFrame.from_dict(Re_list,columns=['ID', 'PCscore'])
        
        # New f
        #print Re_dic
        df=pd.DataFrame(list(Re_dic.items()),columns=['ID', 'PCscore'])
        df=df.sort_values(by=['PCscore'],ascending=False)
        #print df
        out_f=OID+'.retrieval.csv'
        out_f_path=os.path.join(out_path,out_f)
        df.to_csv(out_f_path,index=False)
    
        Num_retrieval=df.shape[0]
        print ' --> The Number of ADC from our DB: '+str(Num_retrieval)


    time2=time()
    print ' Querying and LSalign processing time: '+str('{:.2f}'.format(time2-time1))+' sec.\n'
        

    return    


def sql_connection_existing_Main_FAF():
    try:
        con = apsw.Connection('./Data/DB_Table/Main_FAF.db')
        return con
    except Error:
        print(Error)
    return



def sql_connection_existing_ZINC_SCF():
    try:
        con = apsw.Connection('../3D_Scan/Data/DB_Table/ZINC_SCF.db')
        return con
    except Error:
        print(Error)
        return -1   



def sql_fetch_FAF(con,MW,ring):

    # Calculate MW margin 20%
    MW1=float(MW)
    MW2=float(MW)+(float(MW)*0.2)
    Ring=int(ring)
    #print MW1, MW2, Ring

    # For Test
    #MW1=401.5
    #MW2=402
    #Ring=3

    cursorObj = con.cursor()
    time1=time()
    #cursorObj.execute('select ID from FAF where MW > ? and MW < ? and Rings=? and Lipinski_Violation=0 and LogP>0 and logP<5 and HBA < 10 and HBD<5 and PSA<140 and PSA>75',(MW1,MW2,Ring,))

    #TEST_TEST
    cursorObj.execute('select ID from FAF where MW > ? and MW < ? and Rings=? and Lipinski_Violation=0 and LogP>0 and logP<5 and HBA < 10 and HBD<5 and PSA<140 and PSA>75 LIMIT 5000',(MW1,MW2,Ring,))
    rows = cursorObj.fetchall()
    time2=time()
    #print 'Processing Query Time: '+str(time2-time1)
    re_list = list(rows)
    #print len(tmp_list)

    return re_list


def sql_fetch_Main_S_Check_PCscore(ln_tmp_re_list,tmp_re_list,re_dic,re_list,OID,zid):

    #print zid
    idx_zid=tmp_re_list.index(zid)
    #print '     Processing....'+str(idx_zid)+'/'+str(ln_tmp_re_list)+' '+zid+'\r',
    #sys.stdout.flush()
    #print '     Processing....'+str(idx_zid)+'/'+str(ln_tmp_re_list)+' '+zid+'\n'

    try:
        con = sqlite3.connect('./Data/DB_Table/Main_S.db')
    except Error:
        print('DB connection failed!')
        sys.exit(1)

    con.text_factory = str
    cursorObj = con.cursor()
    
    try:
        cursorObj.execute('SELECT Sdf FROM Sdf_Files WHERE Zid = ?', [zid]) 
    except:
        print 'query error'

    rows = cursorObj.fetchall()
    ln_zid = len(rows)
    tmp_sdf=''

    if ln_zid != 0:
        fp=BytesIO(rows[0][0])
        zfp=zipfile.ZipFile(fp,'r')
        zfp_name=zfp.namelist()
        ln_zfp_name=len(zfp_name)
        #print(ln_zfp_name)
        if ln_zfp_name!=1:
            print('Error: more than two id')
        else:
            pass
            #print(zfp_name[0])
            f_name=zfp_name[0].split('_')
            #print(f_name[-1])
            tmp_sdf=zfp.read(zfp_name[0])                    
            zfp.close()

            #print(sdf_path,zid)
            if 'STK' not in zid:
                tmp_sdf=tmp_sdf+'$$$$\n'
            #print(tmp_sdf)

            zid=zid.encode('utf-8')

            # Calculate the PCScore
            #Sum_ln = Align3D(arg_t,re_dic,OID,zid,tmp_sdf)
            Sum_ln = Align3D(re_dic,OID,zid,tmp_sdf)
            Max_ln=0
            if Sum_ln>Max_ln:
                Max_ln=Sum_ln
            #print zid, PCscore

    print '     Processing....'+str(idx_zid)+'/'+str(ln_tmp_re_list)+' '+zid+' Sum. of args length: '+str(Max_ln)+'\r',
    sys.stdout.flush()
    return 


#def Align3D(arg_t,re_dic,OID,zid,tmp_sdf):
def Align3D_IN(re_dic,OID,zid,tmp_sdf):
    
    path_template = './Data/BB/BackBone/'
    path_query    = './Data/3DAlign/Retrieval_MOL2_From_DB/'
    path_LSalign  = './Data/Sub_P/'

    #########################
    # Threshold for LSAlign
    # Test-Test
    T_PCScore=0.6
    #T_PCScore=0.9 
    #########################

    q_fname=zid+'.mol2'
    path_q=os.path.join(path_query,q_fname)
    path_t=os.path.join(path_template,OID)

    #path_template = './Data/BB/BackBone/'
    #t_fname=OID
    #path_t=os.path.join(path_template,t_fname)
    fp_for_in=open(path_t,'r')  
    lines=fp_for_in.readlines()
    fp_for_in.close()
    new_str=''.join(lines)
    arg_t=To_ARG_str(new_str)

    #print path_t
    #print path_q

    #print OID
    try:
        mymol = readstring('sdf',tmp_sdf)       
    except Exception as e:
        print traceback.format_exc()
        print e.message, e.args
        print e
        print 'reading ',zid,'\'s sdf string'
        return 0

    my_smi =mymol.write(format='smi')
    #mymol_mol2 = my_mol.write(format='mol2')
    # Test-Test
    #print my_smi
    #print mymol_mol2

    try:
        token = my_smi.split()
    except Exception as e:
        print traceback.format_exc()
        print e.message, e.args
        print my_smi
        return 0

    #mymol = readstring('smi',token[0])      
   
    #arg='obabel -:"'+token[0]+'" --gen2D --addtotitle '+zid+' -O '+path_q
    # Mol2 file written to STDOUT
    arg='obabel -:"'+token[0]+'" --gen2D --addtotitle '+zid+' -omol2'
    #os.system(arg)
    #FNULL = open(os.devnull, 'w')

    #process = subprocess.Popen(arg, shell=True,stderr=subprocess.STDOUT)
    #process.wait()
    #print arg
    #print 'ID : '+zid
    try:
        #proc = subprocess.Popen(arg,shell=True,stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        #out = proc.communicate()

        align = subprocess.Popen(arg, shell=True, stdout=subprocess.PIPE)
        out   = align.stdout.readlines()

        #out=subprocess.check_output(arg,stdin=None, stderr=None, shell=False, universal_newlines=True)
        #proc = subprocess.call(arg,shell=True,stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        #print type(proc)
        #print proc
        #return 0,-1
        #print 'out:',out

        #print out
        str_out=list(out)
        #print str_out
        new_str=''.join(str_out)
        arg_q=To_ARG_str(new_str)
        #print arg_q
        #return
    except Exception as e:
        print traceback.format_exc()
        print e.message, e.args
        print arg
        return 0,-1

    #print  process 

    #print zid,token[0]

    #print len(arg_q), arg_q
    #print len(arg_t), arg_t

    #print 'The total length of argument: '+str(len(arg_t)+len(arg_q))
    arg='./Data/Sub_P/LSalignM '+'\"'+arg_q+'\"'+' '+'\"'+arg_t+'\"'
    #print arg
    align = subprocess.Popen(arg, shell=True, stdout=subprocess.PIPE)
    lines = align.stdout.readlines()
    
    #print lines

    #######################################
    # In case of there is no data in SDF_DB
    if len(lines)<3:
        pcscore=0
        return -1
    #######################################

    try:
        str_test=lines[3]
    except Exception as e:
        print traceback.format_exc()
        print e.message, e.args
        print zid,lines[3]
        pcscore=0
        return -1

    str_error='has a problem'
    pcscore=0

    #print arg
    #print zid,lines[3],path.exists(path_q)
    #print lines
    try:
        if str_test.find(str_error) == -1:
            output = lines[3].split()
            pcscore = float(output[2])
        else:
            output = lines[-6].split()
            pcscore = float(output[2])
    except Exception as e:
        print traceback.format_exc()
        print e.message, e.args
        print zid,lines[3]
        pcscore=0
        return -1
        

    #print zid,pcscore
    #sum_len = len(arg_t)+len(arg_q)
    sum_len =  0 

    if pcscore >= T_PCScore:
        re_dic[zid]=pcscore
        path_F = './Data/3DAlign/Final_ADC_PDB/'
        q_fname=OID+'.'+zid+'.pdb'
        path_q=os.path.join(path_F,q_fname)

        mymol.write('pdb',filename=path_q,overwrite=True)


        #os.system(arg)
    else:
        #pass
        path_query = './Data/3DAlign/Retrieval_MOL2_From_DB/'
        q_fname=zid+'.mol2'
        path_q=os.path.join(path_query,q_fname)

    return int(sum_len)
    

def To_ARG_str(mol2):

    # Make Mol2 file type to string for LSalignM version input.
    mystring = mol2.replace('\n', '?').replace('\r', '')
    mystring = mystring.replace(' ', '~')

    return mystring



def Make_PDB_from_ADC():

    Max_line=200000


    time1=time()
    print 'Step 4. Making PDB file from ADC list..............'

    t_dir='./Data/3DAlign/Final_ADC_Files/'
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        #os.system(arg)
        FNULL = open(os.devnull, 'w')
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        process.wait()


    in_path='./Data/3DAlign/Final_ADC_list/'
    inf_list=glob.glob(in_path+'*.retrieval.csv')

    for alist in inf_list:
        #print alist
        df = pd.read_csv(alist)
        #print type(df)
        zlist=df['ID'].tolist()
        #print zlist

        afile_f=os.path.basename(alist)
        token=afile_f.split('.')
        OID = token[0]
        #print afile_f, OID

        ln_zlist=len(zlist)

        if ln_zlist<Max_line:
            Max_line=ln_zlist

        manager = Manager()
        #Re_dic = manager.dict()
        #Re_list = manager.list()
        
        Num_Of_CPU=multiprocessing.cpu_count()
        #Num_Of_CPU=2
        #print Num_Of_CPU
        pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
        func=partial(core_MPDB,ln_zlist,zlist,OID)
        # Test
        pool.map(func,zlist[0:Max_line])
        #pool.map(func,zlist)
        pool.close()
        pool.join()

    time2=time()
    print '\n Making PDB files from ADC list time: '+str('{:.2f}'.format(time2-time1))+' sec.\n'
    
    return


def core_MPDB(ln_zlist,zlist,OID,zid):

    t_path='./Data/3DAlign/Final_ADC_Files/'
    
    idx_zid=zlist.index(zid)
    print '     Processing....'+str(idx_zid+1)+'/'+str(ln_zlist)+' '+zid+'\r',
    sys.stdout.flush()

    try:
        con = sqlite3.connect('./Data/DB_Table/Main_S.db')
    except Error:
        print('DB connection failed!')
        sys.exit(1)

    con.text_factory = str
    cursorObj = con.cursor()
    
    try:
        cursorObj.execute('SELECT Sdf FROM Sdf_Files WHERE Zid = ?', [zid]) 
    except:
        print 'query error'

    rows = cursorObj.fetchall()
    ln_zid = len(rows)
    tmp_sdf=''

    if ln_zid != 0:
        fp=BytesIO(rows[0][0])
        zfp=zipfile.ZipFile(fp,'r')
        zfp_name=zfp.namelist()
        ln_zfp_name=len(zfp_name)
        if ln_zfp_name!=1:
            print('Error: more than two id')
        else:
            f_name=zfp_name[0].split('_')
            tmp_sdf=zfp.read(zfp_name[0])                    
            zfp.close()
            if 'STK' not in zid:
                tmp_sdf=tmp_sdf+'$$$$\n'

            try:
                mymol = readstring('sdf',tmp_sdf)       
            except Exception as e:
                print traceback.format_exc()
                print e.message, e.args
                print e
                print 'reading ',zid,'\'s sdf string'
                return 0

            f_name=OID+'_'+zid+'.pdb'
            f_path=os.path.join(t_path,f_name)
            my_pdb = mymol.write(format='pdb',filename=f_path,overwrite=True)

    return


def sql_connection():
    try:
        #con = sqlite3.connect(':memory:')
        con = apsw.Connection(':memory:')
        return con
    except Error:
        print(Error)
    return


def sql_create_BBR(con):
    cursorObj = con.cursor()
    try: 
        cursorObj.execute("CREATE TABLE BBR_Table(SMILES txt PRIMARY KEY, MW FLOAT, logP FLOAT, RingC SMALLINT, HBA TINYINT, HBD TINYINT, TPSA FLOAT, ids VARCHAR)")
        return 1
    except apsw.ConstraintError:
        return -1


def sql_insert_BBR(con, entities):
    global du_count
    cursorObj = con.cursor()
    try:
        cursorObj.execute('INSERT INTO BBR_Table(SMILES, MW, logP, RingC, HBA, HBD, TPSA, ids) VALUES(?,?,?,?,?,?,?,?)', entities)
        return 1
    except apsw.ConstraintError:
        du_count+=1
        return -1


def Query_Fatch_SCF(con,asmi):

    #con = sql_connection_existing_ZINC_SCF()
    if con != -1:
        cursorObj = con.cursor()
        cursorObj.execute('select SMILES,ids from SCF_Table where SMILES = ?',[asmi])
        rows = cursorObj.fetchall()

        if len(rows)>0:
            re_list = list(rows[0])
            ln_re_list=len(re_list)
            asmiles=re_list[0]
            IDs=re_list[1]
            return IDs
            #token=IDs.split(',')
            #if len(IDs)>0:
            #print asmi,len(token)
            #Re_dic={'Backbone':asmi}
            #Re_dic.update({'Count':len(token)}) 
            #Re_dic[asmi]=len(token)
        else:
            return ''
            #print asmi,'0'
            #Re_dic={'Backbone':asmi}
            #Re_dic.update({'Count':0}) 
            #Re_dic[asmi]=0 
        #Re_list.append(Re_dic)


def Make_BBR(in_file):
    
    
    con_SCF = sql_connection_existing_ZINC_SCF()
    con_BBR = sql_connection()
    re = sql_create_BBR(con_BBR)
    
    df = pd.read_csv(in_file)
    ln_df=len(df)

    idx = 1
    print 'Start making Backbone Ring DB............'
    for index,rows in df.iterrows():
        asmi=rows[0]
        ZIDs = rows[1].split(',')
        ZID_Count=len(ZIDs)
        #print str(idx)+'/'+str(ln_df)+' '+asmi+' '+str(ZID_Count)
        print '  ->'+str(idx)+'/'+str(ln_df)+' '+asmi+' '+str(ZID_Count)+'                                                \r',
        sys.stdout.flush()
        if ZID_Count>0:
            amol = readstring('smi',asmi)
            t= amol.sssr
            Num_Ring = len(t)
            if Num_Ring>1:
                ZIDs  = Query_Fatch_SCF(con_SCF,asmi)
                if ZIDs !='':
                    desc = amol.calcdesc(descnames=['MW', 'logP', 'HBA1', 'HBD','TPSA'])
                    entities = (asmi, desc['MW'],desc['logP'],Num_Ring,desc['HBA1'],desc['HBD'],desc['TPSA'],ZIDs)
                    re = sql_insert_BBR(con_BBR, entities)
                    if re==-1:
                        print 'Insert Error'
        idx+=1

    #########################################
    # Write to MemoryDB to File
    print '\nWriting BBR DB to disk...........'
    dest=apsw.Connection('./ZINC_BBR.db')
    with dest.backup('main',con_BBR,'main') as backup:
        backup.step()
    dest.close()
    con_BBR.close()
    print 'End writing..........OK'


def Extract_CP_SMILES(in_file):


    df = pd.read_csv(in_file)
    ln_df=len(df)

    Main_list=[]

    print 'Start processing ............'
    idx=1

    for index,rows in df.iterrows():
        tmp_list=[]
        asmi=rows[0]
        ZIDs = rows[1].split(',')
        #ZID_Count=len(ZIDs)

        #print '  ->'+str(idx)+'/'+str(ln_df)+' '+asmi+'                                                                  '
        print '  ->'+str(idx)+'/'+str(ln_df)+' '+asmi+'                                                                  \r',
        sys.stdout.flush()
        idx+=1
        try:
            amol = readstring('smi',asmi)
        except:
            print '\nReading smiles Error :'+asmi
            continue

        desc = amol.calcdesc(descnames=['MW', 'logP', 'HBA1', 'HBD','TPSA'])
        #print desc
        t= amol.sssr
        Num_Ring = len(t)
        #print tmp_list

        '''
        Re_dic={}
        Re_dic={'Backbone':asmi}
        Re_dic.update({'MW':desc['MW']}) 
        #Re_dic.update({'MW':'{:.1f}'.format(desc['MW'])}) 
        Re_dic.update({'LogP':desc['logP']}) 
        #Re_dic.update({'LogP':'{:.1f}'.format(desc['logP'])}) 
        Re_dic.update({'HBA':desc['HBA1']}) 
        Re_dic.update({'HBD':desc['HBD']}) 
        Re_dic.update({'TPSA':desc['TPSA']}) 
        #Re_dic.update({'TPSA':'{:.1f}'.format(desc['TPSA'])}) 
        Re_dic.update({'Ring':Num_Ring}) 
        Re_dic.update({'IDs':ZIDs}) 
        Main_list.append(Re_dic)
        '''

        tmp_list=[asmi,desc['MW'],desc['logP'],desc['HBA1'],desc['HBD'],desc['TPSA'],Num_Ring,ZIDs]
        Main_list.append(tmp_list)
        #print idx

    #df=pd.DataFrame(Main_list)
    df = pd.DataFrame.from_records(Main_list,columns=['Backbone','MW','LogP','HBA','HBD','TPSA','Ring','IDs'])
    print df.head()
    infile=in_file+'.CP.csv'
    df.to_csv(infile,sep=',',index=False)


def Processing_input(i_type, i_content):

    f_list=[]

    if i_type =='l': 
        df = pd.read_csv(i_content)
        #df = df.iloc[:,0]
        #ln_df=len(df)
        return df,f_list

    Main_list=[]
    if i_type =='f':
        f_list=glob.glob(i_content+'*.smi')
        for afile in f_list:
            fp_for_in = open(afile,'r')
            line= fp_for_in.readline()
            token=line.split()
            tmp_list=[]
            tmp_list = [token[0]]
            Main_list.append(tmp_list)
        df = pd.DataFrame.from_records(Main_list,columns=['Backbone'])
        return df, f_list
            
        
def Make_BB(df,f_list):
    
    # Making tmp dir
    t_dir='./sng_tmp/' 
    FNULL = open(os.devnull, 'w')
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        arg='rm '+t_dir+'*'
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
        process.wait()


    c_name=df.columns[0]
    slist=df[c_name].values.tolist()
    ln_slist=len(slist)

    if len(f_list)==0:
        f_list=['']*len(slist)

    nslist=zip(slist,f_list)
    nslist = list(nslist)
    ln_nslist=len(nslist)


    print 'Starting making backbone........'
    # Mulitprocessing
    manager = Manager()
    Re_dic = manager.dict()

    sid_list = manager.list()
    tmp_dic  = manager.dict()
    backbone_dic  = manager.dict()

    SMILES_dic = manager.dict()
    smi_list = manager.list()

    Main_list = manager.list()
    tmp_list = manager.list()

    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    #print Num_Of_CPU
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(Make_BB_Core,ln_nslist,nslist,Main_list,tmp_list)
    pool.map(func,nslist)
    pool.close()
    pool.join()

    #sys.exit(1)


    Main_list=list(Main_list)
    df = pd.DataFrame.from_records(Main_list,columns=['Backbone','Original','File'])
    #print df.head()
    print '\n -> Ending making backbone........'


    arg = 'rm ./sng_tmp/*'
    process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)
    process.wait()

    #print df.head()
    #sys.exit(1)


    return df


def Make_BB(asmi):

    # File input and output version!
    # Check the sng.jar exists or not!
    if not os.path.exists('./sng.jar'):
        print 'This python program needs \'sng.jar\'.'
        print 'Move the \'sng.jar\' program at this fold and re excute this program.'
        os.exit(1)

    # File input and output version!
    proc = os.getpid()
    f_name=str(proc)+'.smi'

    fp_for_out=open(f_name,'w')
    fp_for_out.write(asmi)
    fp_for_out.close()

    # excuate sng
    FNULL = open(os.devnull, 'w')
    subprocess.call(['%sjava'%java_path,'-jar','sng.jar','generate','-o', str(proc)+'.tmp',f_name], stdout=FNULL, stderr=subprocess.STDOUT)

    re_name=str(proc)+'.tmp'
    fp_for_in=open(re_name,'r')
    lines=fp_for_in.readlines()
    fp_for_out.close()

    #########################################
    # Extract Backbone
    scaffold_backbone=''
    ln_lines=len(lines)

    #####################################################################
    # If there is no scaffolds, The Original structure is used as backbone.
    if ln_lines==1:
        scaffold_backbone = asmi
    #####################################################################

    if ln_lines>1:
        last_line=lines[ln_lines-1]
        if ln_lines>2:
            token=last_line.split(',')
            scaffold_backbone = token[-2].strip()
        if ln_lines==2:
            token=last_line.split()
            scaffold_backbone = token[-1][:-1].strip()

    scaffold_backbone = processing_backbone(scaffold_backbone)
    os.unlink(f_name)
    os.unlink(re_name)
    
    return scaffold_backbone


def Extract_CP_SMILES2(i_type,i_content,df):

    ln_df=len(df)
    print '\nStart Extract CP............'
    idx=1

    c_name=df.columns[0]
    slist=df[c_name].values.tolist()
    ln_slist=len(slist)
    if i_type=='f':
        c_name=df.columns[1]
        oslist=df[c_name].values.tolist()
        ln_oslist=len(oslist)

        c_name=df.columns[2]
        fplist=df[c_name].values.tolist()
        ln_fplist=len(fplist)
    if i_type=='l':
        oslist=['']*len(slist)
        fplist=['']*len(slist)
        
    nslist=zip(slist,oslist,fplist)

    tmp_slist=df[c_name]
    #print type(tmp_slist)

    ### Check input file has 'nan' value
    print ' -> Check input file has \'nan\' value.....'
    NAN_list=[]
    for index, rows in df.iterrows():
        if pd.isna(rows[0]):
            print 'Index: ', index
            NAN_list.append(index)
    if len(NAN_list)!=0:
        print ' ->Error indexes: ',NAN_list
        return
    else:
        pass
    print ' -> End checking nan value......'

    #return

    #c_name=df.columns[1]
    #zlist=df[c_name].values.tolist()
    #print slist[0:50]
    #print len(slist[0:50])
    #return

    manager = Manager()
    Re_dic = manager.dict()
    tmp_list = manager.list()
    Main_list = manager.list()
        
    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    #print Num_Of_CPU
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(Core_ExCP,ln_slist,slist,tmp_list,Main_list,Re_dic)
    pool.map(func,nslist)
    pool.close()
    pool.join()

    Main_list = list(Main_list)
    try:
        df = pd.DataFrame.from_records(Main_list,columns=['Backbone','Original','File_Loc','MW','LogP','HBA','HBD','TPSA','Ring'])
    except Exception as e:
        print traceback.format_exc()
        print e.message, e.args
        print e
    #return
    #print df.head()
    #return
    if i_type=='f':
        token=i_content.split('/')
        infile='Re_'+token[-2]+'.files.csv'
        #print token
    else:
        infile=i_content+'.CP.csv'
    df.to_csv('./Uniq_Proc/' + infile,sep=',',index=False)

    print '\n -> End extracting CP......'
    return


def Core_ExCP(ln_slist,slist,tmp_list,Main_list,Re_dic,ele):

    asmi=ele[0]
    oasmi=ele[1]
    fpasmi=ele[2]

    try:
        idx=slist.index(asmi)
    except Exception as e:
        print 'asmi:',asmi
        print traceback.format_exc()
        print e.message, e.args
        print e
    #aline=df.iloc[idx]

    #print '  ->'+str(idx)+'/'+str(ln_slist)+' '+asmi+'                                                                  '
    print '  ->'+str(idx)+'/'+str(ln_slist)+' '+asmi+'                                                                  \r',
    sys.stdout.flush()
    #return

    try:
        amol = readstring('smi',asmi)
    except:
        print '\nReading smiles Error :'+asmi
        return

    tmp_list=[]
    desc = amol.calcdesc(descnames=['MW', 'logP', 'HBA1', 'HBD','TPSA'])
    #print desc
    t= amol.sssr
    Num_Ring = len(t)
    #print tmp_list

    tmp_list=[asmi,oasmi,fpasmi,desc['MW'],desc['logP'],desc['HBA1'],desc['HBD'],desc['TPSA'],Num_Ring]
    Main_list.append(tmp_list)
    tmp_list=[]

    return



######################################################################################################

def Extract_CP(asmi):

    # Extract chemical perporty
    tmp_list=[]
    Re_dic={}
    try:
        amol = readstring('smi',asmi)
    except:
        print '\nReading smiles Error :'+asmi
        return tmp_list

    desc = amol.calcdesc(descnames=['MW', 'logP', 'HBA1', 'HBD','TPSA'])
    t= amol.sssr
    Num_Ring = len(t)

    tmp_list=[asmi,desc['MW'],desc['logP'],desc['HBA1'],desc['HBD'],desc['TPSA'],Num_Ring]
    Re_dic={'SMILES':asmi}
    Re_dic.update({'MW':desc['MW']}) 
    Re_dic.update({'LogP':desc['logP']}) 
    Re_dic.update({'HBA':desc['HBA1']}) 
    Re_dic.update({'HBD':desc['HBD']}) 
    Re_dic.update({'TPSA':desc['TPSA']}) 
    Re_dic.update({'Ring':Num_Ring}) 
    
    #return tmp_list 
    return Re_dic 


def Extract_BB(asmi):

    # File input and output version!
    # Check the sng.jar exists or not!
    if Check_sng() ==-1:  
        return 

    # File input and output version!
    proc = os.getpid()
    f_name=str(proc)+'.smi'

    fp_for_out=open(f_name,'w')
    fp_for_out.write(asmi)
    fp_for_out.close()

    # excuate sng
    FNULL = open(os.devnull, 'w')
    subprocess.call(['%sjava'%java_path,'-jar','sng.jar','generate','-o', str(proc)+'.tmp',f_name], stdout=FNULL, stderr=subprocess.STDOUT)

    re_name=str(proc)+'.tmp'
    fp_for_in=open(re_name,'r')
    lines=fp_for_in.readlines()
    fp_for_out.close()

    #########################################
    # Extract Backbone
    scaffold_backbone=''
    ln_lines=len(lines)

    #####################################################################
    # If there is no scaffolds, The Original structure is used as backbone.
    if ln_lines==1:
        scaffold_backbone = asmi
    #####################################################################

    if ln_lines>1:
        last_line=lines[ln_lines-1]
        if ln_lines>2:
            token=last_line.split(',')
            scaffold_backbone = token[-2].strip()
        if ln_lines==2:
            token=last_line.split()
            scaffold_backbone = token[-1][:-1].strip()

    scaffold_backbone = processing_backbone(scaffold_backbone)
    os.unlink(f_name)
    os.unlink(re_name)
    
    return scaffold_backbone


def processing_backbone(scaffold_backbone):

    type1 = '[*]'
    type2 = '([*])'

    old_backbone=scaffold_backbone
    scaffold_backbone = scaffold_backbone.replace(type2,'')
    scaffold_backbone = scaffold_backbone.replace(type1,'')
    #print 'Before: '+old_backbone+' ---> After: '+scaffold_backbone

    return scaffold_backbone


def Check_sng():
    if not os.path.exists('./sng.jar'):
        print 'This python program needs \'sng.jar\'.'
        print 'Move the \'sng.jar\' program at this fold and re excute this program.'
        return -1
    else:
        return 1


#def ExtractM_BB(flist:list,slist:list):
def ExtractM_BB(iPath):
    
    time1=time()

    if Check_sng() ==-1:  
        return 

    flist=glob.glob(iPath+'*.smi')
    slist=[]
    for afile in flist:
        fp_for_in = open(afile,'r')
        line=fp_for_in.readline().strip()
        token=line.split()
        #print token[0]
        slist.append(token[0])

    nslist=zip(slist,flist)
    nslist = list(nslist)
    ln_nslist=len(nslist)

    print 'Starting making backbone........'
    # Mulitprocessing
    manager = Manager()
    Re_dic = manager.dict()
    Main_list = manager.list()
    tmp_list = manager.list()

    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(Make_BB_Core,ln_nslist,nslist,Main_list,tmp_list)
    pool.map(func,nslist)
    pool.close()
    pool.join()

    Main_list=list(Main_list)
    df = pd.DataFrame.from_records(Main_list,columns=['Backbone','Original','File'])
    #print df.head()
    print '\n --> Ending making backbone........'
    #print df.head()

    time2=time()
    print 'Extracting input chmeical feature time: '+str('{:.2f}'.format(time2-time1))+' sec.'

    return df



def Make_BB_Core(ln_slist,slist,Main_list,tmp_list,casmi):

    aidx=slist.index(casmi)
    #return
    asmi=casmi[0]
    asmi_path=casmi[1]
    
    print ' --> Processing....:'+str(aidx+1)+'/'+str(ln_slist)+' ,mol:'+asmi+'\r', 
    sys.stdout.flush()

    proc = os.getpid()
    f_name=str(proc)+'.smi'

    # write smiles
    fp_for_out=open(f_name,'w')
    fp_for_out.write(asmi)
    fp_for_out.close()

    # excuate sng
    FNULL = open(os.devnull, 'w')
    subprocess.call(['%sjava'%java_path,'-jar','sng.jar','generate','-o', str(proc)+'.tmp',f_name], stdout=FNULL, stderr=subprocess.STDOUT)

    re_name=str(proc)+'.tmp'
    fp_for_in=open(re_name,'r')
    lines=fp_for_in.readlines()
    fp_for_out.close()

    #########################################
    # Extract Backbone
    scaffold_backbone=''
    ln_lines=len(lines)

    #####################################################################
    # If there is no scaffolds, The Original structure is used as backbone.
    if ln_lines==1:
        scaffold_backbone = asmi
    #####################################################################

    if ln_lines>1:
        last_line=lines[ln_lines-1]
        if ln_lines>2:
            token=last_line.split(',')
            scaffold_backbone = token[-2].strip()
        if ln_lines==2:
            token=last_line.split()
            scaffold_backbone = token[-1][:-1].strip()

    scaffold_backbone = processing_backbone(scaffold_backbone)
    tmp_list=[scaffold_backbone,asmi,asmi_path]
    Main_list.append(tmp_list)

    os.unlink(f_name)
    os.unlink(re_name)
    return


def Draw_BB_smi(o_path,f_name,smi):

    BB_f_name=f_name+'.png'
    BB_img_path = os.path.join(o_path,BB_f_name)
    #print BB_img_path
    arg='obabel -:"'+smi+'" -O '+BB_img_path+' -xw 350 -xh 250 -d'
    #print arg
    #FNULL = open(os.devnull, 'w')
    #subprocess.call(arg,stdout=FNULL, stderr=subprocess.STDOUT)
    os.system(arg)

    return


def Extract_SubScaffold(asmi):

    if Check_SNG()==-1:
        sys.exit(1)

    # File input and output version!
    proc = os.getpid()
    f_name=str(proc)+'.smi'
    fp_for_out=open(f_name,'w')
    fp_for_out.write(asmi)
    fp_for_out.close()

    # excuate sng
    FNULL = open(os.devnull, 'w')
    subprocess.call(['%sjava'%java_path,'-jar','sng.jar','generate','-o', str(proc)+'.tmp',f_name], stdout=FNULL, stderr=subprocess.STDOUT)

    re_name=str(proc)+'.tmp'
    fp_for_in=open(re_name,'r')
    head=fp_for_in.readline()
    lines=fp_for_in.readlines()
    fp_for_out.close()

    #########################################
    # Extract Backbone
    scaffold_backbone=''
    ln_lines=len(lines)

    #####################################################################
    # If there is no scaffolds, The Original structure is used as backbone.
    if ln_lines==0:
        scaffold_backbone = asmi
    #####################################################################

    Main_list=[]
    for aline in lines:
        tmp_list=[]
        token=aline.split('\t')
        #print token[0],token[1]

        #### - replace [nH] with n
        #token[1] = token[1].replace('[nH]','n')
        #print token[1]

        tmp_list=[token[0],token[1]]
        Main_list.append(tmp_list)

    df = pd.DataFrame.from_records(Main_list,columns=['Ring_Num','smiles'])
    df = df.sort_values(by=['Ring_Num'],ascending=True)

    os.unlink(f_name)
    os.unlink(re_name)
    #print df

    return df 


def Extract_FusionR(asmi):

    df = Extract_SubScaffold(asmi)
    FR=[]
    for index,row in df.iterrows():
        Num_Ring=int(row[0])
        if Num_Ring>1:
            smi=row[1]
            mol = Chem.MolFromSmiles(smi)
            m_RB = Descriptors.NumRotatableBonds(mol)
            if m_RB==0:
                FR.append(smi)

        '''
        mol = Chem.MolFromSmiles(smi)
        m_RingCount = Descriptors.RingCount(mol)
        m_RB = Descriptors.NumRotatableBonds(mol)
        m_SA = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        m_BHA= rdMolDescriptors.CalcNumBridgeheadAtoms(mol) 
        print 'smiles:',smi,',RC:',m_RingCount,',RB:',m_RB,',SA:',m_SA,',BHA:',m_BHA
        '''
    #print FR, df
    return FR, df


def Swap_Ring(asmi):

    # Handling both side rings
    list_FR, scaffold_df = Extract_FusionR(asmi) 
    BB = Extract_BB(asmi)
    list_ring=[]
    r1=''
    r2=''
    r_flag=0
    idx_r1=0
    for achar in BB:
        r1=r1+achar
        if achar.isdigit() and r_flag==0:
            r_flag=achar
            continue
        if achar.isdigit() and r_flag!=0 and r_flag==achar:
            break
        idx_r1+=1
    print r1,idx_r1

    r_BB=BB[::-1]
    print len(BB),BB
    print len(r_BB),r_BB
    
    r_flag=0
    idx_r2=0
    for achar in r_BB:
        r2=r2+achar
        if achar.isdigit() and r_flag==0:
            r_flag=achar
            continue
        if achar.isdigit() and r_flag!=0 and r_flag==achar:
            r2=r2+r_BB[idx_r2+2]
            r2=r2[::-1]
            break
        idx_r2+=1    
            
    print r2,idx_r2
        
    
    New_BB=r2+BB[idx_r1+2:len(BB)-(idx_r2+3)]+r1
    print New_BB

    '''
    One_Ring=[]
    # Extract One-Ring
    for index,row in scaffold_df.iterrows():
        Num_Ring=int(row[0])
        if Num_Ring==1:
            One_Ring.append(row[1])

    print One_Ring,list_FR

    tmp_list=[]
    #Matching substructure
    for aFR in list_FR:
        print aFR
        maFR = Chem.MolFromSmiles(aFR)
        for aOR in One_Ring:
            maOR = Chem.MolFromSmiles(aOR)
            if maFR.HasSubstructMatch(maOR):
                #print aOR,'Yes'
                tmp_list.append(aOR)

    print tmp_list
    Side_One_Ring = [x for x in One_Ring if x not in tmp_list]
    print One_Ring
    print Side_One_Ring
    print BB
    '''
    return New_BB


def Check_SNG():
    if not os.path.exists('./sng.jar'):
        print ' This python program needs \'sng.jar\'.'
        print ' Move the \'sng.jar\' program at this fold and re excute this program.'
        return -1
    else:
        return 1


def Read_SMILES_FILE(f_path):
    
    fp_for_in=open(f_path,'r')
    line=fp_for_in.readline()
    fp_for_in.close()

    token=line.split()
    asmi=token[0]

    return asmi
    


def Search_ASMILES(asmi,m_type):

    T1_pcscore_cutoff=0.98
    T_ZIDs=set()

    if m_type==1 or m_type==3: 
        #########################################
        # 1. Extract ZIDs by Backbone using BB DB
        print ' -> Starting search: BB exact matching'

        T_ZIDs_BB=set()

        re=Fetch_BB(asmi,DB_Path)
        if len(re)>0:
            candi =re[-1]
            candi = set(candi)
            T_ZIDs_BB=T_ZIDs_BB.union(candi)
        else:
            print 'SMILES: '+asmi+', There is no result for extact BB matching.'
               
        cp = Extract_CP(asmi)

        entry=(cp['MW'],cp['LogP'],cp['HBA'],cp['HBD'],cp['TPSA'],cp['Ring'])
        Same_CPs = Fetch_BB_CP(entry,DB_Path)
        print  ' --> Same_CPs num: ',len(Same_CPs)

        # Extract smiles for same CP
        list_smi=[x[0] for x in Same_CPs]

        # Make id list for AlignM3D
        list_smi_id=[range(0,len(list_smi),1)]
        list_smi_id = list_smi_id[0]
        list_smi_id = [str(x) for x in list_smi_id]
        df = AlignM3D('template',asmi,list_smi_id,list_smi)
        df = df.drop(df[df.PCScore<T1_pcscore_cutoff].index)
        
        for index,row in df.iterrows():
            candi = Same_CPs[int(row[1])][-1]  
            candi = set(candi) 
            T_ZIDs_BB=T_ZIDs_BB.union(candi)


    if m_type==2 or m_type==3: 

        ############################################
        # 2. Extract ZIDs by Scaffold using Scaffold DB
        print '\n -> Starting search: Scaffold exact matching'
        T_ZIDs_SF=set()
        list_zids = Fetch_Scaffold(asmi,DB_Path)

        if len(list_zids)==0:
            print ' --> There is no result for scaffold matching'
        else:
            re = FetchM_SDF_to_SMI(list_zids,DB_Path)
            list_smi = [x[0] for x in re]
            list_smi_id = [x[1] for x in re]

            df = AlignM3D('template',asmi,list_smi_id,list_smi)
            df = df.drop(df[df.PCScore<T1_pcscore_cutoff].index)

            tmp_list=[]
            for index,row in df.iterrows():
                tmp_list.append(row[1])
            T_ZIDs_SF = T_ZIDs_SF.union(set(tmp_list))

    if m_type==1:
        return T_ZIDs_BB 
    if m_type==2:
        return T_ZIDs_SF 
    if m_type==3:
        T_ZIDs = T_ZIDs.union(T_ZIDs_SF)
        T_ZIDs = T_ZIDs.union(T_ZIDs_BB)
        return T_ZIDs 

    return


def Extract_SSSR_Idx(asmi):

    amol = readstring('smi',asmi)
    list_ring=amol.sssr
    #num_ring = len(list_ring)
    tmp_list=[]

    #list_ring=amol.OBMol.FindLSSR()
    print list_ring

    #list_ring_info=amol.OBMol.FindRingAtomsAndBonds()  
    #print list_ring_info
    

    for aring in list_ring:
        tmp_list.append(aring._path)
        
    return tmp_list



def Leave_Ring_NoInterconnection(df):

    # Remove like this example: (Ring) --- O --- (Ring)
    # Only leave the ring itself: Ring, RingRing
    for index,row in df.iterrows():
        #print row[0],row[1]
        #amol = readstring('smi',row[1])
        #mol = Chem.MolFromSmiles(row[1])
        #m_RB = Descriptors.NumRotatableBonds(mol)

        re = Have_leaf_Ring(row[1])
        #print 're:',re
        #print 
        if re == -1:
            df = df.drop(df[df.smiles==row[1]].index)
        
        continue

    return df
    '''
    #print row[1]
        #print m_RB

        # Structure: Ring - Ring
        if m_RB>0:
            df = df.drop(df[df.smiles==row[1]].index)
            print 'm_RB'
            print 'OK'
            continue

        # By num. of ring
        ssr = Chem.GetSymmSSSR(mol)

        # In case of a ring
        if len(ssr) == 1:
            continue

        if m_RB==0:
            continue

        # In case of more than two rings
        set_idx=set()
        for aring in ssr:
            print set(aring)
            set_idx = set_idx | set(aring)

        for aring in ssr:
            print set(aring)
            set_idx = set_idx & set(aring)

        #print set_idx
        #return
        if len(set_idx)==0:
            df = df.drop(df[df.smiles==row[1]].index)
            print 'set_idx'
            print 'OK'
            continue


        for atom in amol:
            if not atom.OBAtom.IsInRing():
                # pass atom having doublebond to ring 
                if atom.OBAtom.HasDoubleBond():
                    continue
                if atom.type !='H':
                    df = df.drop(df[df.smiles==row[1]].index)
                    print 'Not in ring OK'
                    break

    return df
    '''


def Have_leaf_Ring(asmi):

    #mol = readstring('smi',asmi)
    mol = Chem.MolFromSmiles(asmi)
    #print 'In Have_leaf_Ring (mol): ',mol,asmi,type(mol)
    dic_atom_neig_idx, dic_atom_neig_sym = Extract_Atom_Neighbors(mol)
    ssr = Chem.GetSymmSSSR(mol)
    #print dic_atom_neig_idx

    # a ring in ssr is leaf ring and remove from the candidate

    diff_set = set()
    if len(ssr) == 1:
        return 1
    for aring in ssr:
        set_idx=set(aring)
        #print list(aring)
        for num_atom in list(aring):
            nei_idx = dic_atom_neig_idx[num_atom]
            #print num_atom, set(nei_idx)
            tmp_set = set()
            if set(nei_idx) <= set_idx:
                pass
            else:
                tmp_set=set(nei_idx) - set_idx   
                #print 'tmp_set:',tmp_set
                tmp_list = list(tmp_set)
                while(0<len(tmp_list)):
                    X = tmp_list.pop()
                    #print num_atom,x
                
                    # Remove speical case
                    xbond = mol.GetBondBetweenAtoms(num_atom,X).GetBondType()
                    #print xbond,type(xbond)
                    if xbond == Chem.rdchem.BondType.DOUBLE:
                        #print 'doble'
                        continue

                    # X located at other ring clustrer?
                    #print 'X:',X
                    xnode = mol.GetAtomWithIdx(X).IsInRing()
                    if xnode == True:
                        cluster_set = set()
                        cluster_set = cluster_set | set_idx
                        for xaring in ssr:
                            tmp_idx = set(xaring)
                            i_idx =set()
                            i_idx = cluster_set & tmp_idx 
                            if len(i_idx) != 0:
                                cluster_set = cluster_set | tmp_idx
                        #print 'cluster set:',cluster_set
                        if X not in cluster_set:
                            return -1
                    else:
                        i_idx = set()
                        for xaring in ssr:
                            tmp_dix = set(xaring)
                            i_idx = set(xaring) & set_idx
                            if len(i_idx) ==0:
                                return -1
                        
                diff_set = diff_set | tmp_set
        #print 'len of diff_set:',len(diff_set)
        #if len(diff_set) == 1 or len(diff_set) == 3 or len(diff_set) == 5:
        if len(diff_set) == 1: 
            return -1
        diff_set.clear() 

    #print 
    return 1



def Check_SubScaffold(df):

    for index,row in df.iterrows():
        #print row[1]
        mol = Chem.MolFromSmiles(row[1])
        if mol == None:
            return -1
    return 1



def Extract_Inner_Scaffold(asmi):


    #m1 = Chem.MolFromSmiles(asmi,kekuleSmiles=True)

    '''
    re = asmi.find('i')
    if re != -1:
        print 'Having \'i\''
        return -1
    '''

    #amol = readstring('smi',asmi)
    #my_smi = amol.write(format='smi')

    # For substructure matching using RDKit
    my_smi = Make_Canonical_SMI(asmi)
    #print 'Canonical SMI:',my_smi
    m = Chem.MolFromSmiles(my_smi)
    # Test_Code
    #smiles_1b  = Chem.MolToSmiles(m).replace('-','~')
    #pattern_1b = Chem.MolFromSmarts(smiles_1b)

    if m==None:
        return -1

    #dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(my_smi)
    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    #dic_atom_sym =  Extract_Atom_sym(my_smi)
    dic_atom_sym =  Extract_Atom_sym(m)


    #print dic_atom_neig_idx
    #print
    #print dic_atom_neig_sym
    #print 
    #print dic_atom_sym
    #print

    #df = Extract_SubScaffold(my_smi)
    df = Extract_SubScaffold(asmi)

    # Check SubScaffold is nomal
    re = Check_SubScaffold(df)
    
    if re == -1:
        return -1

    #df = df.sort_values(by=['Ring_Num'],ascending=False)
    #print df
    #return

    df = Leave_Ring_NoInterconnection(df)
    df = df.sort_values(by=['Ring_Num'],ascending=False)
   
    #print 'Orginal DF'
    #print df
    #print

    tmp_df = copy.deepcopy(df)
    col_smiles_list = tmp_df['smiles'].tolist()
    #print col_smiles_list

    tmp_dic={}
    Sidx_dic={}
    Ridx_dic={}

    idx = 0
    ln_tmp_df = len(df)
    for index,row in tmp_df.iterrows():
        if int(row[0]) > 1:

            #print 'orig:',row[0],row[1]
            c_smi = Make_Canonical_SMI(row[1])
            #print 'befor:',row[1],'after:',c_smi

            tmp_mol = Chem.MolFromSmiles(c_smi)

            #print  idx,ln_tmp_df

            for l_idx in range(idx+1,ln_tmp_df):
                #print 'col_smiles_list[l_idx]:',col_smiles_list[l_idx]
                p_smi = Make_Canonical_SMI(col_smiles_list[l_idx])
                #print p_smi
                patt = Chem.MolFromSmiles(p_smi)
                try:
                    match_list = tmp_mol.GetSubstructMatches(patt)
                except:
                    return -1
                #print match_list
                if len(match_list) != 0:
                    df = df.drop(df[df.smiles==col_smiles_list[l_idx]].index)

                else:
                    # using LSAlign 
                    #pc_score = Align3D('m',row[1],'patt',p_smi)
                    #print pc_score
                    m_set = set()
                    tmol = readstring('smi',row[1])
                    m_set = Extract_Atoms_Set(tmol)
                    m_list = Extract_Atoms_List(tmol)
                    q_set = set()
                    qmol = readstring('smi',p_smi)
                    q_set =  Extract_Atoms_Set(qmol)
                    q_list = Extract_Atoms_List(qmol)
                    #print m_set, q_set
                    #print m_list, q_list


            #print 
        idx+=1

    #print 'Remove dependency'
    #print df 
  
    tmp_df = copy.deepcopy(df)

    #print 'Is terminal ring'
    for index,row in tmp_df.iterrows():
        #print 'orig:',row[0],row[1]
        re = Is_terminal_Ring(m,row[1],dic_atom_neig_idx)
        #print 're:',re
        if re == -1 or re == 0:
            df = df.drop(df[df.smiles==row[1]].index)

    #print df 
    #return  

    tmp_df = copy.deepcopy(df)

    if len(tmp_df) == 0:
        print 'There is no inter-connected ring'
        return -1
        #return 0

    list_re =[]
    for index,row in tmp_df.iterrows():
        p_smi = Make_Canonical_SMI(row[1])
        #print p_smi
        patt = Chem.MolFromSmiles(p_smi)
        sma = Chem.MolToSmarts(patt)
        patt_1 = Chem.MolFromSmarts(sma)
        match_list = m.GetSubstructMatches(patt_1)
        #print match_list

        re = Matching_Using_MakeScaffoldGeneric(m,patt)
        match_list2 = []
        if re == -1:
            pass 
        else:
            match_list2 = re

        #print match_list2
        if len(match_list)==0:
            match_list = match_list2

        # For editiing
        mw = Chem.RWMol(patt)
        list_re_tmp =[]
        if len(match_list)>0:
            X = list(match_list)
            atom_list = list(X[0])
            atom_set = set(X[0])
            #print atom_list,atom_set
            #print atom_list

            i = 0
            brench = 0
            for atom_idx in atom_list:
                #print atom_idx
                nei_atom = dic_atom_neig_sym[atom_idx]
                nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
                #print nei_atom,nei_atom_idx

                if len(nei_atom_idx - atom_set) != 0:
                    brench +=1 
                    #print 'Hi',nei_atom_idx
                    is_idx=nei_atom_idx - atom_set
                    #print 'Hi',is_idx
                    is_idx = list(is_idx).pop()
                    #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                    btype = m.GetBondBetweenAtoms(atom_idx,is_idx).GetBondType()
                    atype = dic_atom_sym[is_idx]
             
                    atom_num = m.GetAtomWithIdx(is_idx).GetAtomicNum()
                    #print 'atomic num:',m.GetAtomWithIdx(is_idx).GetAtomicNum()

                    #print 'Hi',i,atom_idx,is_idx,btype,atype
                    ire = mw.AddAtom(Chem.Atom(atom_num))
                    #print 'ire:',ire
                    #print i,ire
                    bre = mw.AddBond(i,ire,btype)
                    #print 'bre:',bre
                    #print 

                i+=1 
        #print 'brench:',brench 

        try:
            Chem.SanitizeMol(mw)
            asmi = Chem.MolToSmiles(mw)
            #print asmi
            list_re_tmp.append(asmi)
            list_re_tmp.append(row[1])
            list_re_tmp.append(brench)
            list_re.append(list_re_tmp)
        except:
            pass

    #print list_re

    return list_re



def Extract_Inner_Scaffold_2(asmi):


    #m1 = Chem.MolFromSmiles(asmi,kekuleSmiles=True)

    '''
    re = asmi.find('i')
    if re != -1:
        print 'Having \'i\''
        return -1
    '''

    #amol = readstring('smi',asmi)
    #my_smi = amol.write(format='smi')

    # For substructure matching using RDKit
    my_smi = Make_Canonical_SMI(asmi)
    #print 'Canonical SMI:',my_smi
    m = Chem.MolFromSmiles(my_smi)
    # Test_Code
    #smiles_1b  = Chem.MolToSmiles(m).replace('-','~')
    #pattern_1b = Chem.MolFromSmarts(smiles_1b)

    if m==None:
        return -1

    #dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(my_smi)
    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    #dic_atom_sym =  Extract_Atom_sym(my_smi)
    dic_atom_sym =  Extract_Atom_sym(m)


    #print dic_atom_neig_idx
    #print
    #print dic_atom_neig_sym
    #print 
    #print dic_atom_sym
    #print

    #df = Extract_SubScaffold(my_smi)
    df = Extract_SubScaffold(asmi)
    #print(df)
    #return

    # Check SubScaffold is nomal
    re = Check_SubScaffold(df)
    
    if re == -1:
        return -1

    #df = df.sort_values(by=['Ring_Num'],ascending=False)
    #print df
    #return

    df = Leave_Ring_NoInterconnection(df)
    df = df.sort_values(by=['Ring_Num'],ascending=False)
   
    #print 'Orginal DF'
    #print df
    #print

    tmp_df = copy.deepcopy(df)
    col_smiles_list = tmp_df['smiles'].tolist()
    #print col_smiles_list

    tmp_dic={}
    Sidx_dic={}
    Ridx_dic={}

    idx = 0
    ln_tmp_df = len(df)
    for index,row in tmp_df.iterrows():
        if int(row[0]) > 1:

            #print 'orig:',row[0],row[1]
            c_smi = Make_Canonical_SMI(row[1])
            #print 'befor:',row[1],'after:',c_smi

            tmp_mol = Chem.MolFromSmiles(c_smi)

            #print  idx,ln_tmp_df

            for l_idx in range(idx+1,ln_tmp_df):
                #print 'col_smiles_list[l_idx]:',col_smiles_list[l_idx]
                p_smi = Make_Canonical_SMI(col_smiles_list[l_idx])
                #print p_smi
                patt = Chem.MolFromSmiles(p_smi)
                try:
                    match_list = tmp_mol.GetSubstructMatches(patt)
                except:
                    return -1
                #print match_list
                if len(match_list) != 0:
                    df = df.drop(df[df.smiles==col_smiles_list[l_idx]].index)

                else:
                    # using LSAlign 
                    #pc_score = Align3D('m',row[1],'patt',p_smi)
                    #print pc_score
                    m_set = set()
                    tmol = readstring('smi',row[1])
                    m_set = Extract_Atoms_Set(tmol)
                    m_list = Extract_Atoms_List(tmol)
                    q_set = set()
                    qmol = readstring('smi',p_smi)
                    q_set =  Extract_Atoms_Set(qmol)
                    q_list = Extract_Atoms_List(qmol)
                    #print m_set, q_set
                    #print m_list, q_list


            #print 
        idx+=1

    #print 'Remove dependency'
    #print df 
  
    tmp_df = copy.deepcopy(df)

    #print 'Is terminal ring'
    for index,row in tmp_df.iterrows():
        #print 'orig:',row[0],row[1]
        re = Is_terminal_Ring(m,row[1],dic_atom_neig_idx)
        #print 're:',re
        if re == -1 or re == 0:
            df = df.drop(df[df.smiles==row[1]].index)

    #print df 
    #return  

    tmp_df = copy.deepcopy(df)

    if len(tmp_df) == 0:
        print 'There is no inter-connected ring'
        #return -1
        return 0

    list_re =[]
    for index,row in tmp_df.iterrows():
        p_smi = Make_Canonical_SMI(row[1])
        #print p_smi
        patt = Chem.MolFromSmiles(p_smi)
        sma = Chem.MolToSmarts(patt)
        patt_1 = Chem.MolFromSmarts(sma)
        match_list = m.GetSubstructMatches(patt_1)
        #print match_list

        re = Matching_Using_MakeScaffoldGeneric(m,patt)
        match_list2 = []
        if re == -1:
            pass 
        else:
            match_list2 = re

        #print match_list2
        if len(match_list)==0:
            match_list = match_list2

        # For editiing
        mw = Chem.RWMol(patt)
        list_re_tmp =[]
        if len(match_list)>0:
            X = list(match_list)
            atom_list = list(X[0])
            atom_set = set(X[0])
            #print atom_list,atom_set
            #print atom_list

            i = 0
            brench = 0
            for atom_idx in atom_list:
                #print atom_idx
                nei_atom = dic_atom_neig_sym[atom_idx]
                nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
                #print nei_atom,nei_atom_idx

                if len(nei_atom_idx - atom_set) != 0:
                    brench +=1 
                    #print 'Hi',nei_atom_idx
                    is_idx=nei_atom_idx - atom_set
                    #print 'Hi',is_idx
                    is_idx = list(is_idx).pop()
                    #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                    btype = m.GetBondBetweenAtoms(atom_idx,is_idx).GetBondType()
                    atype = dic_atom_sym[is_idx]
             
                    atom_num = m.GetAtomWithIdx(is_idx).GetAtomicNum()
                    #print 'atomic num:',m.GetAtomWithIdx(is_idx).GetAtomicNum()

                    #print 'Hi',i,atom_idx,is_idx,btype,atype
                    ire = mw.AddAtom(Chem.Atom(atom_num))
                    #print 'ire:',ire
                    #print i,ire
                    bre = mw.AddBond(i,ire,btype)
                    #print 'bre:',bre
                    #print 

                i+=1 
        #print 'brench:',brench 

        try:
            Chem.SanitizeMol(mw)
            asmi = Chem.MolToSmiles(mw)
            #print asmi
            list_re_tmp.append(asmi)
            list_re_tmp.append(row[1])
            list_re_tmp.append(brench)
            list_re.append(list_re_tmp)
        except:
            pass

    #print list_re

    if len(list_re) == 0:
        return 0

    return list_re




def Extract_Inner_Scaffold_3(asmi):

    #m1 = Chem.MolFromSmiles(asmi,kekuleSmiles=True)
    #amol = readstring('smi',asmi)
    #my_smi = amol.write(format='smi')

    # For substructure matching using RDKit
    my_smi = Make_Canonical_SMI(asmi)
    #print 'Canonical SMI:',my_smi
    m = Chem.MolFromSmiles(my_smi)

    if m==None:
        return -1

    #dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(my_smi)
    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    #dic_atom_sym =  Extract_Atom_sym(my_smi)
    dic_atom_sym =  Extract_Atom_sym(m)


    #print dic_atom_neig_idx
    #print
    #print dic_atom_neig_sym
    #print 
    #print dic_atom_sym
    #print

    #df = Extract_SubScaffold(my_smi)
    df = Extract_SubScaffold(asmi)
    t_df = copy.deepcopy(df)
    #print(df)
    #return

    # Check SubScaffold is nomal
    re = Check_SubScaffold(df)
    
    if re == -1:
        return -1

    #df = df.sort_values(by=['Ring_Num'],ascending=False)
    #print df
    #return

    df = Leave_Ring_NoInterconnection(df)
    df = df.sort_values(by=['Ring_Num'],ascending=False)
   
    #print 'Orginal DF'
    #print df
    #print

    tmp_df = copy.deepcopy(df)
    col_smiles_list = tmp_df['smiles'].tolist()
    #print col_smiles_list

    tmp_dic={}
    Sidx_dic={}
    Ridx_dic={}

    idx = 0
    ln_tmp_df = len(df)
    for index,row in tmp_df.iterrows():
        if int(row[0]) > 1:

            #print 'orig:',row[0],row[1]
            c_smi = Make_Canonical_SMI(row[1])
            #print 'befor:',row[1],'after:',c_smi

            tmp_mol = Chem.MolFromSmiles(c_smi)

            #print  idx,ln_tmp_df

            for l_idx in range(idx+1,ln_tmp_df):
                #print 'col_smiles_list[l_idx]:',col_smiles_list[l_idx]
                p_smi = Make_Canonical_SMI(col_smiles_list[l_idx])
                #print p_smi
                patt = Chem.MolFromSmiles(p_smi)
                try:
                    match_list = tmp_mol.GetSubstructMatches(patt)
                except:
                    return -1
                #print match_list
                if len(match_list) != 0:
                    df = df.drop(df[df.smiles==col_smiles_list[l_idx]].index)

                else:
                    # using LSAlign 
                    #pc_score = Align3D('m',row[1],'patt',p_smi)
                    #print pc_score
                    m_set = set()
                    tmol = readstring('smi',row[1])
                    m_set = Extract_Atoms_Set(tmol)
                    m_list = Extract_Atoms_List(tmol)
                    q_set = set()
                    qmol = readstring('smi',p_smi)
                    q_set =  Extract_Atoms_Set(qmol)
                    q_list = Extract_Atoms_List(qmol)
                    #print m_set, q_set
                    #print m_list, q_list


            #print 
        idx+=1

    #print 'Remove dependency'
    #print df 
  
    tmp_df = copy.deepcopy(df)

    #print 'Is terminal ring'
    for index,row in tmp_df.iterrows():
        #print 'orig:',row[0],row[1]
        re = Is_terminal_Ring(m,row[1],dic_atom_neig_idx)
        #print 're:',re
        if re == -1 or re == 0:
            df = df.drop(df[df.smiles==row[1]].index)

    #print df 
    #return  

    tmp_df = copy.deepcopy(df)

    if len(tmp_df) == 0:
        print 'There is no inter-connected ring'
        #return -1
        return 0

    list_re =[]
    for index,row in tmp_df.iterrows():
        p_smi = Make_Canonical_SMI(row[1])
        #print p_smi
        patt = Chem.MolFromSmiles(p_smi)
        sma = Chem.MolToSmarts(patt)
        patt_1 = Chem.MolFromSmarts(sma)
        match_list = m.GetSubstructMatches(patt_1)
        #print match_list

        re = Matching_Using_MakeScaffoldGeneric(m,patt)
        match_list2 = []
        if re == -1:
            pass 
        else:
            match_list2 = re

        #print match_list2
        if len(match_list)==0:
            match_list = match_list2

        # For editiing
        mw = Chem.RWMol(patt)
        list_re_tmp =[]
        if len(match_list)>0:
            X = list(match_list)
            atom_list = list(X[0])
            atom_set = set(X[0])
            #print atom_list,atom_set
            #print atom_list

            i = 0
            brench = 0
            for atom_idx in atom_list:
                #print atom_idx
                nei_atom = dic_atom_neig_sym[atom_idx]
                nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
                #print nei_atom,nei_atom_idx

                if len(nei_atom_idx - atom_set) != 0:
                    brench +=1 
                    #print 'Hi',nei_atom_idx
                    is_idx=nei_atom_idx - atom_set
                    #print 'Hi',is_idx
                    is_idx = list(is_idx).pop()
                    #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                    btype = m.GetBondBetweenAtoms(atom_idx,is_idx).GetBondType()
                    atype = dic_atom_sym[is_idx]
             
                    atom_num = m.GetAtomWithIdx(is_idx).GetAtomicNum()
                    #print 'atomic num:',m.GetAtomWithIdx(is_idx).GetAtomicNum()

                    #print 'Hi',i,atom_idx,is_idx,btype,atype
                    ire = mw.AddAtom(Chem.Atom(atom_num))
                    #print 'ire:',ire
                    #print i,ire
                    bre = mw.AddBond(i,ire,btype)
                    #print 'bre:',bre
                    #print 

                i+=1 
        #print 'brench:',brench 

        try:
            Chem.SanitizeMol(mw)
            asmi = Chem.MolToSmiles(mw)
            #print asmi
            list_re_tmp.append(asmi)
            list_re_tmp.append(row[1])
            list_re_tmp.append(brench)
            list_re.append(list_re_tmp)
        except:
            pass

    #print list_re

    if len(list_re) == 0:
        return 0

    # For merage 
    #print 'In3',list_re,len(list_re)
    if len(list_re)>1:
        merged_smi = Merge_Scaffold(t_df,list_re)
        #Attach_hand(m,merged_smi)
        #print merged_smi
        list_re = Attach_hand(m,merged_smi)
        #return list_re
    else:
        pass

    return list_re



def Extract_Inner_Scaffold_4(asmi):

    #m1 = Chem.MolFromSmiles(asmi,kekuleSmiles=True)
    #amol = readstring('smi',asmi)
    #my_smi = amol.write(format='smi')

    # For substructure matching using RDKit
    my_smi = Make_Canonical_SMI(asmi)
    #print 'Canonical SMI:',my_smi
    m = Chem.MolFromSmiles(my_smi)

    if m==None:
        return -1

    #dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(my_smi)
    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    #dic_atom_sym =  Extract_Atom_sym(my_smi)
    dic_atom_sym =  Extract_Atom_sym(m)

    '''
    print dic_atom_neig_idx
    print
    print dic_atom_neig_sym
    print 
    print dic_atom_sym
    print
    '''

    #return

    #df = Extract_SubScaffold(my_smi)
    df = Extract_SubScaffold(asmi)
    t_df = copy.deepcopy(df)
    #print(df)
    #return

    '''
    #amol = readstring('smi',alist[1])
    ssr = Chem.GetSymmSSSR(m)
    print len(ssr)
    for aring in range(0,len(ssr)):
        print list(ssr[aring])
    return
    '''

    # Check SubScaffold is nomal
    re = Check_SubScaffold(df)
    
    if re == -1:
        return -1

    #df = df.sort_values(by=['Ring_Num'],ascending=False)
    #print df
    #return

    df = Leave_Ring_NoInterconnection(df)
    df = df.sort_values(by=['Ring_Num'],ascending=False)
   
    #print 'Orginal DF'
    #print df
    #print

    tmp_df = copy.deepcopy(df)
    col_smiles_list = tmp_df['smiles'].tolist()
    #print col_smiles_list

    tmp_dic={}
    Sidx_dic={}
    Ridx_dic={}

    idx = 0
    ln_tmp_df = len(df)
    for index,row in tmp_df.iterrows():
        if int(row[0]) > 1:

            #print 'orig:',row[0],row[1]
            c_smi = Make_Canonical_SMI(row[1])
            #print 'befor:',row[1],'after:',c_smi

            tmp_mol = Chem.MolFromSmiles(c_smi)

            #print  idx,ln_tmp_df

            for l_idx in range(idx+1,ln_tmp_df):
                #print 'col_smiles_list[l_idx]:',col_smiles_list[l_idx]
                p_smi = Make_Canonical_SMI(col_smiles_list[l_idx])
                #print p_smi
                patt = Chem.MolFromSmiles(p_smi)
                try:
                    match_list = tmp_mol.GetSubstructMatches(patt)
                except:
                    return -1
                #print match_list
                if len(match_list) != 0:
                    df = df.drop(df[df.smiles==col_smiles_list[l_idx]].index)

                else:
                    # using LSAlign 
                    #pc_score = Align3D('m',row[1],'patt',p_smi)
                    #print pc_score
                    m_set = set()
                    tmol = readstring('smi',row[1])
                    m_set = Extract_Atoms_Set(tmol)
                    m_list = Extract_Atoms_List(tmol)
                    q_set = set()
                    qmol = readstring('smi',p_smi)
                    q_set =  Extract_Atoms_Set(qmol)
                    q_list = Extract_Atoms_List(qmol)
                    #print m_set, q_set
                    #print m_list, q_list


            #print 
        idx+=1

    #print 'Remove dependency'
    #print df 
  
    tmp_df = copy.deepcopy(df)

    #print 'Is terminal ring'


    Terminal_Scaffold_idx =[]
    for index,row in tmp_df.iterrows():
        #print 'orig:',row[0],row[1]
        re = Is_terminal_Ring(m,row[1],dic_atom_neig_idx)
        #print 're:',re
        if re == -1 or re == 0:
            df = df.drop(df[df.smiles==row[1]].index)
            patt1 = Chem.MolFromSmiles(row[1])
            match_list = m.GetSubstructMatches(patt1)
            if len(match_list)>0:
                #print 'Termianl scaffold:',match_list[0]
                #print 'match_list:',match_list[0]
                tmp = list(match_list[0])
                tmp.sort()
                #print tmp
                Terminal_Scaffold_idx.append(tmp) 

    #print 'Scaffold:',df 
    #print 'Terminal scaffold idx list:',Terminal_Scaffold_idx
    #return  

    tmp_df = copy.deepcopy(df)

    if len(tmp_df) == 0:
        #print 'There is no inter-connected ring'
        #return -1
        return 0

    list_re =[]
    for index,row in tmp_df.iterrows():
        p_smi = Make_Canonical_SMI(row[1])
        #print p_smi
        patt = Chem.MolFromSmiles(p_smi)
        sma = Chem.MolToSmarts(patt)
        patt_1 = Chem.MolFromSmarts(sma)
        match_list = m.GetSubstructMatches(patt_1)
        #print match_list

        re = Matching_Using_MakeScaffoldGeneric(m,patt)


        #####################################################################
        ## Case 1. There is more than two same scaffolds of which is terminal scaffold in ligand
        ## Romve the scaffold which is terminal scaffold 
        # Convert the tuple 're' as list 're' to remove terminal scaffold
        lst_re = []
        for are in re:
            tmp_re = list(are)
            #print(tmp_re)
            tmp_re.sort()
            lst_re.append(tmp_re)
        #print 'lst_re:',lst_re

        lst_re2 = copy.deepcopy(lst_re)
        # Check whether matching index is in terminal scaffold or not
        if len(re)>=2:
            tmp_re = list(re)
            #print lst_re,re,type(re),'HIHIHIHI'
            for aidx in lst_re:
                if aidx in Terminal_Scaffold_idx:
                    #print 'Terminal scaffold index:',aidx
                    lst_re2.remove(aidx)
        #print 'list_re2:',lst_re2
        
        # Convert the list_re2 as tuple re
        #print re
        re = tuple()
        for are in lst_re2:
            atp = tuple(are)
            #print atp
            re =re + (atp,)
        #print 'Re:',re
        #return
        ## End Case 1. ########################################################


        match_list2 = []
        if re == -1:
            pass 
        else:
            match_list2 = re

        #print match_list2
        if len(match_list)==0:
            match_list = match_list2

        #print 'At Ring match index: ', match_list
        #return

        # For editiing
        mw = Chem.RWMol(patt)
        list_re_tmp =[]
        if len(match_list)>0:
            X = list(match_list)
            atom_list = list(X[0])
            atom_set = set(X[0])
            #print atom_list,atom_set
            #print atom_list
            #return

            i = 0
            brench = 0
            for atom_idx in atom_list:
                #print atom_idx
                nei_atom = dic_atom_neig_sym[atom_idx]
                nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
                #print nei_atom,nei_atom_idx

                if len(nei_atom_idx - atom_set) != 0:
                    brench_diff = len(nei_atom_idx - atom_set)
                    #print 'brench dff:',brench_diff
                    '''
                    for idx in range(0,brench_diff):
                        print idx
                    '''

                    brench = brench + brench_diff
                    #brench +=1 
                    #print 'Hi',nei_atom_idx
                    is_idx=list(nei_atom_idx - atom_set)
                    #print 'nei_idx:',is_idx
                    for tidx in range(0,brench_diff):
                        ais_idx = is_idx.pop()
                        #print tidx,ais_idx
                        #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                        btype = m.GetBondBetweenAtoms(atom_idx,ais_idx).GetBondType()
                        atype = dic_atom_sym[ais_idx]
             
                        atom_num = m.GetAtomWithIdx(ais_idx).GetAtomicNum()
                        #print 'atomic num:',m.GetAtomWithIdx(is_idx).GetAtomicNum()

                        #print 'Hi',i,atom_idx,is_idx,btype,atype
                        ire = mw.AddAtom(Chem.Atom(atom_num))
                        #print 'ire:',ire
                        #print i,ire
                        bre = mw.AddBond(i,ire,btype)
                        #print 'bre:',bre
                        #print 

                i+=1 
        #print 'brench:',brench 

        try:
            Chem.SanitizeMol(mw)
            asmi = Chem.MolToSmiles(mw)
            #print asmi
            list_re_tmp.append(asmi)
            list_re_tmp.append(row[1])
            list_re_tmp.append(brench)
            list_re.append(list_re_tmp)
        except:
            pass

    #print list_re

    if len(list_re) == 0:
        return 0

    # For merage 
    #print 'In3',list_re,len(list_re)
    if len(list_re)>1:
        merged_smi = Merge_Scaffold(t_df,list_re)
        #Attach_hand(m,merged_smi)
        #print merged_smi
        list_re = Attach_hand(m,merged_smi)
        #return list_re
    else:
        pass

    return list_re



def Extract_Inner_Scaffold_5(asmi):

    #m1 = Chem.MolFromSmiles(asmi,kekuleSmiles=True)
    #amol = readstring('smi',asmi)
    #my_smi = amol.write(format='smi')

    # For substructure matching using RDKit
    my_smi = Make_Canonical_SMI(asmi)
    #print 'Canonical SMI:',my_smi
    m = Chem.MolFromSmiles(my_smi)

    if m==None:
        return -1

    #dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(my_smi)
    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    #dic_atom_sym =  Extract_Atom_sym(my_smi)
    dic_atom_sym =  Extract_Atom_sym(m)

    '''
    print dic_atom_neig_idx
    print
    print dic_atom_neig_sym
    print 
    print dic_atom_sym
    print
    '''

    #return

    #df = Extract_SubScaffold(my_smi)
    df = Extract_SubScaffold(asmi)
    ##### -  Result of sng
    t_df = copy.deepcopy(df)
    #print(df)
    #return

    '''
    #amol = readstring('smi',alist[1])
    ssr = Chem.GetSymmSSSR(m)
    print len(ssr)
    for aring in range(0,len(ssr)):
        print list(ssr[aring])
    return
    '''

    # Check SubScaffold is nomal
    re = Check_SubScaffold(df)
    #print 're',re


    if re == -1:
        return -1

    #df = df.sort_values(by=['Ring_Num'],ascending=False)
    #print df
    #return

    #### - Remove scaffold which connected with bond like Ring-Ring, Ring-Ring-Ring
    df = Leave_Ring_NoInterconnection(df)
    df = df.sort_values(by=['Ring_Num'],ascending=False)
   
    #print 'Orginal DF'
    #print df
    #print
    #return

    tmp_df = copy.deepcopy(df)
    col_smiles_list = tmp_df['smiles'].tolist()
    #print col_smiles_list

    tmp_dic={}
    Sidx_dic={}
    Ridx_dic={}

    idx = 0
    ln_tmp_df = len(df)
    for index,row in tmp_df.iterrows():
        if int(row[0]) > 1:

            #print 'orig:',row[0],row[1]
            c_smi = Make_Canonical_SMI(row[1])
            #print 'befor:',row[1],'after:',c_smi

            tmp_mol = Chem.MolFromSmiles(c_smi)

            #print  idx,ln_tmp_df

            #### - Remove the single rings which make the fusion ring
            for l_idx in range(idx+1,ln_tmp_df):
                #print 'col_smiles_list[l_idx]:',col_smiles_list[l_idx]
                p_smi = Make_Canonical_SMI(col_smiles_list[l_idx])
                #print p_smi
                patt = Chem.MolFromSmiles(p_smi)
                '''
                try:
                    match_list = tmp_mol.GetSubstructMatches(patt)
                except:
                    return -1
                print match_list
                '''
                #if len(match_list) == 0:           
                try:
                    match_list , mapping = is_substructure_mol(tmp_mol, patt)
                except:
                    return -1

                #print match_list, mapping
                try:                
                    Tmatch_list , Tmapping = is_substructure_mol(m, patt)
                except:
                    return -1 

                #print Tmatch_list, Tmapping
                # the ring exist at more than one loaction
                if len(Tmatch_list)>1:
                    continue

                if len(match_list) != 0:
                    df = df.drop(df[df.smiles==col_smiles_list[l_idx]].index)
                else:
                    pass 

            #print 
        idx+=1

    #print 'Resut of removing dependency'
    #print df 
  
    tmp_df = copy.deepcopy(df)
    #print 'Is terminal ring'
    Terminal_Scaffold_idx =[]
    #print tmp_df
    #return

    #### - Making the terminal ring index and remove terminal ring
    for index,row in tmp_df.iterrows():
        #print 'orig:',row[0],row[1]
        re = Is_terminal_Ring_2(m,row[1],dic_atom_neig_idx)
        #print 're:',re
        if re == -999:
            return -1

        # If row[1] is not termianl ring
        if re == 1 or re == 0:       # '1' is terminal node'
            df = df.drop(df[df.smiles==row[1]].index)
            # Decide if the ring is substructure or not!!
            patt1 = Chem.MolFromSmiles(row[1])
            match_list = m.GetSubstructMatches(patt1)
            #print 'HIHI',row[1],match_list
    
            # Additional code test substructure matching one more
            if len(match_list)==0:
                try:
                    match_list , mapping = is_substructure_mol(m, patt1)
                except:
                    return -1

                #print match_list, mapping

            if len(match_list)>0:
                #print 'Termianl scaffold:',match_list[0]
                #print 'match_list:',match_list[0]
                tmp = list(match_list[0])
                tmp.sort()
                #print tmp
                Terminal_Scaffold_idx.append(tmp) 
            #print 
    #print 'Scaffold:',df 
    #print 'Terminal scaffold idx list:',Terminal_Scaffold_idx
    #return  

    tmp_df = copy.deepcopy(df)

    if len(tmp_df) == 0:
        #print 'There is no inter-connected ring'
        #return -1
        return 0

    list_re =[]

    #print 'after making terminal ring idnex:'
    #print tmp_df
    for index,row in tmp_df.iterrows():
        p_smi = Make_Canonical_SMI(row[1])
        #print p_smi
        patt = Chem.MolFromSmiles(p_smi)
        sma = Chem.MolToSmarts(patt)
        patt_1 = Chem.MolFromSmarts(sma)
        match_list = m.GetSubstructMatches(patt_1)
        #print match_list

        #re = Matching_Using_MakeScaffoldGeneric(m,patt)
        #print '1re:',re
        try:
            re , mapping = is_substructure_mol(m, patt)
        except:
            return -999
        #print re,mapping
        #map_dic = mapping[0]
        #print '2re:',re,mapping[0]
       
        re, mapping = select_mapping(re,mapping,Terminal_Scaffold_idx)
        map_dic = mapping[0]
        #print re,map_dic

        #####################################################################
        ## Case 1. There is more than two same rings of which is terminal ring in ligand
        ## Romve the scaffold which is terminal scaffold 

        # Convert the tuple 're' as list 're' to remove terminal ring
        lst_re = []
        for are in re:
            tmp_re = list(are)
            #print(tmp_re)
            tmp_re.sort()
            lst_re.append(tmp_re)
        #print 'lst_re:',lst_re

        lst_re2 = copy.deepcopy(lst_re)

        # Check whether matching index is in terminal ring or not
        if len(re)>=2:
            tmp_re = list(re)
            #print lst_re,re,type(re),'HIHIHIHI'
            for aidx in lst_re:
                if aidx in Terminal_Scaffold_idx:
                    #print 'Terminal scaffold index:',aidx
                    lst_re2.remove(aidx)
        #print 'list_re2:',lst_re2

        # Convert the list_re2 as tuple re
        #print re
        re = tuple()
        for are in lst_re2:
            atp = tuple(are)
            #print atp
            re =re + (atp,)
        #print 'Re:',re
        #return
        ## End Case 1. ########################################################


        match_list2 = []
        if re == -1:
            pass 
        else:
            match_list2 = re

        #print match_list2
        if len(match_list)==0:
            match_list = match_list2

        #print 'At Ring match index: ', match_list
        #return

        #### - For editiing attach hand
        #print patt
        mw = Chem.RWMol(patt)
        list_re_tmp =[]
        if len(match_list)>0:
            X = list(match_list)
            atom_list = list(X[0])
            atom_set = set(X[0])
            #print atom_list,atom_set
            #return

            i = 0
            brench = 0
            for atom_idx in atom_list:
                #print 'atom index:',atom_idx
                nei_atom = dic_atom_neig_sym[atom_idx]
                nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
                #print nei_atom,nei_atom_idx

                if len(nei_atom_idx - atom_set) != 0:
                    brench_diff = len(nei_atom_idx - atom_set)
                    #print 'brench dff:',brench_diff
                    '''
                    for idx in range(0,brench_diff):
                        print idx
                    '''

                    brench = brench + brench_diff
                    #brench +=1 
                    #print 'Hi',nei_atom_idx
                    is_idx=list(nei_atom_idx - atom_set)
                    #print 'nei_idx:',is_idx
                    for tidx in range(0,brench_diff):
                        ais_idx = is_idx.pop()
                        #print tidx,'ais_idx:',ais_idx
                        #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                        btype = m.GetBondBetweenAtoms(atom_idx,ais_idx).GetBondType()
                        atype = dic_atom_sym[ais_idx]
             
                        atom_num = m.GetAtomWithIdx(ais_idx).GetAtomicNum()
                        #print 'atomic num:',m.GetAtomWithIdx(is_idx).GetAtomicNum()

                        #print 'Hi',i,atom_idx,is_idx,btype,atype
                        ire = mw.AddAtom(Chem.Atom(atom_num))
                        #print 'i:',i,'ire',ire
                        try:
                            bre = mw.AddBond(map_dic[atom_idx],ire,btype)
                        except:
                            return -1
                        #bre = mw.AddBond(i,ire,btype)
                        #print 'bre:',bre
                        #print 

                i+=1 
        #print 'brench:',brench 

        try:
            Chem.SanitizeMol(mw)
            asmi = Chem.MolToSmiles(mw)
            #print asmi
            list_re_tmp.append(asmi)
            list_re_tmp.append(row[1])
            list_re_tmp.append(brench)
            list_re.append(list_re_tmp)
        except:
            pass

    #print list_re

    if len(list_re) == 0:
        return 0

    # For merage 
    #print 'In3',list_re,len(list_re)
    if len(list_re)>1:
        merged_smi = Merge_Scaffold(t_df,list_re)
        #Attach_hand(m,merged_smi)
        #print 'merger_smi:',merged_smi
        list_re = Attach_hand_2(m,merged_smi)
        #return list_re
    else:
        pass

    return list_re




def Extract_Inner_Scaffold_6(asmi):

    #m1 = Chem.MolFromSmiles(asmi,kekuleSmiles=True)
    #amol = readstring('smi',asmi)
    #my_smi = amol.write(format='smi')

    # For substructure matching using RDKit
    try:
        my_smi = Make_Canonical_SMI(asmi)
        #print 'Canonical SMI:',my_smi
    except:
        return -1

    m = Chem.MolFromSmiles(my_smi)

    if m==None:
        return -1

    #dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(my_smi)
    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    #dic_atom_sym =  Extract_Atom_sym(my_smi)
    dic_atom_sym =  Extract_Atom_sym(m)
    SSSR_idxs = Extract_SSSR_Idx_RDkit(m)
    #print SSSR_idxs
    #return

    '''
    print dic_atom_neig_idx
    print
    print dic_atom_neig_sym
    print 
    print dic_atom_sym
    print
    '''

    #return

    #df = Extract_SubScaffold(my_smi)
    df = Extract_SubScaffold(asmi)

    #### - Make the index list and mapping dic for a smiles: 2021.10.01
    df = Make_Scaffold_Index(m,df)

    ##### Check Point ####
    #print df[['smiles','MIdx']]
    #return

    ##### -  Result of sng for use later
    t_df = copy.deepcopy(df)

    #print df[['smiles','MIdx']]
    #return

    # Error in RDKir and return -1
    if type(df) == int:
        return -1

    #print df
    #return

    #### - Remove scaffolds which connected with bond like Ring-Ring, Ring-Ring-Ring
    #### - Leave the scaffold: a ring or a scaffold
    df = Leave_Ring_NoInterconnection(df)
    df = df.sort_values(by=['Ring_Num'],ascending=False)
   
    #print 'Orginal DF'
    #print df

    ##### Check Point ####
    #print df[['smiles','MIdx']]
    #return

    tmp_df = copy.deepcopy(df)

    #### - Remove a index of ring which is part of fusion ring
    for index,row in tmp_df.iterrows():
        #print 'orig:',row[0],row[1],row[2]
        #p_mol = Chem.MolFromSmiles(row[1])
        #patt = Chem.MolFromSmiles(row[1])
        re_fusion = Is_Part_Of_Fusion_Ring(SSSR_idxs,dic_atom_neig_idx,row[2])

        #print re_fusion
        # '1' means a part of fusion ring
        if re_fusion == 1:
            #print row[2]
            df = df.drop(df[df.MIdx == row[2]].index)
        # not a part of fusion ring    
        else:
            pass

    #print 'Resut of removing dependency'
    #print df 

    ##### Check Point ####
    #print df[['smiles','MIdx']]
    #return


    tmp_df = copy.deepcopy(df)
    #print tmp_df
    #return
    #### - Making the terminal ring index and remove terminal ring
    for index,row in tmp_df.iterrows():
        #print 'orig:',row[0],row[1],row[2]
        re = Is_terminal_Ring_3(m,row[2],dic_atom_neig_idx)
        #print 're:',re

        # '1' means that the ring is terminal node 
        if re == 1:       
            #df = df.drop(df[df.smiles==row[1]].index)
            df = df.drop(df[df.MIdx == row[2]].index)

    #print 'Scaffold:' 

    ##### Check Point ####
    #print df[['smiles','MIdx']]
    #return  


    if len(df) == 0:
        #print 'There is no inter-connected ring'
        return 0


    tmp_df = copy.deepcopy(df)
    list_re =[]
    #print tmp_df

    #### - Add the hand to the scaffold

    #### rings in the tmp_df are passed. so need the index set
    #### for more than one scaffold!!! use later
    #print SSSR_idxs
    total_index =set()
    for index,row in tmp_df.iterrows():
        #print 'In add hand:orig:',row[0],row[1],row[2],row[3]
        p_smi = row[1]
        match_list = row[2]
        map_dic = row[3]
        # Test
        '''
        if p_smi == 'c1ccnc2[nH]ncc12':
            p_smi = 'c1ccnc2Nncc12'
        '''
    
        patt = Chem.MolFromSmiles(p_smi)
        #patt = Chem.RemoveHs(patt)
       

        total_index.update(match_list)

        #print 'match list:',match_list
        #continue
        mw = Chem.RWMol(patt)
        atom_list = list(match_list)
        atom_set = set(match_list)
        #print atom_list,atom_set
        #Show_Index_Information(p_smi)

        list_re_tmp =[]
        brench = 0
        for atom_idx in atom_list:
            #print 'atom index:',atom_idx
            nei_atom = dic_atom_neig_sym[atom_idx]
            nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
            #print nei_atom,nei_atom_idx

            if len(nei_atom_idx - atom_set) != 0:
                brench_diff = len(nei_atom_idx - atom_set)
                #print 'brench dff:',brench_diff
                brench = brench + brench_diff
                #print 'neighbor atom list index:',nei_atom_idx
                is_idx=list(nei_atom_idx - atom_set)
                #print 'nei_idx:',is_idx
                for tidx in range(0,brench_diff):
                    ais_idx = is_idx.pop()
                    #print tidx,'ais_idx:',ais_idx
                    #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                    btype = m.GetBondBetweenAtoms(atom_idx,ais_idx).GetBondType()
                    atype = dic_atom_sym[ais_idx]
         
                    atom_num = m.GetAtomWithIdx(ais_idx).GetAtomicNum()
                    #print 'atomic num:',m.GetAtomWithIdx(ais_idx).GetAtomicNum()

                    #print 'Hi',atom_idx,is_idx,btype,atype
                    ire = mw.AddAtom(Chem.Atom(atom_num))
                    #print '   atom_idx:',atom_idx,'corresp:,',map_dic[atom_idx],'ire:',ire
                    atom_num = m.GetAtomWithIdx(ais_idx).GetAtomicNum()
                    try:
                        bre = mw.AddBond(map_dic[atom_idx],ire,btype)
                    except: 
                        print 'Eorror in attaching hand'
                        return -1
        #print 'brench:',brench 
        try:
            #Chem.SanitizeMol(mw)
            asmi = Chem.MolToSmiles(mw)
            #print asmi
            list_re_tmp.append(asmi)
            list_re_tmp.append(row[1])
            list_re_tmp.append(brench)
            list_re.append(list_re_tmp)
        except:
            pass

    #print 'list_re:',list_re

    if len(list_re) == 0:
        return 0

    # For merage 
    #print 'In3',list_re,len(list_re)
    #t_df = copy.deepcopy(df)
    #print total_index
    if len(list_re)>1:
        merged_smi = Merge_Scaffold(t_df,list_re,total_index)
        #Attach_hand(m,merged_smi)
        #print 'merger_smi:',merged_smi
        list_re = Attach_hand_2(m,merged_smi)
        #return list_re
    else:
        pass

    return list_re



def Extract_SSSR_Idx_RDkit(rdmol):

    tmp_list = []

    sssr = Chem.GetSymmSSSR(rdmol)
    for aridx in sssr:
            tlist = list(aridx)
            tlist.sort()
            tmp_list.append(tlist)
        
    return tmp_list



def Is_Part_Of_Fusion_Ring(SSSR_idxs,dic_atom_neig_idx,mlist):
   
    #print type(mlist),mlist
    #print 'SSSR_idxs:',SSSR_idxs
    #print dic_atom_neig_idx

    tmp_set = set()
    for amidx in mlist:
        tmp_idx = dic_atom_neig_idx[amidx]
        for aidx in tmp_idx:
            if aidx not in mlist:
                tmp_set.add(aidx)

    #print tmp_set

    #Check tmp_set for SSSR
    ln_re_intersection = 0
    for aSSSR in SSSR_idxs:
        aSSSR_Set = set(aSSSR)
        re_intersecton = aSSSR_Set.intersection(tmp_set)
        #print 'intersection:',re_intersecton 
        ln_re_intersection = len(re_intersecton)
        # a part of fusion ring
        if ln_re_intersection >=2:
            return 1
        
    #print
    # Not part of fusion ring
    if ln_re_intersection <2:
        return -1



def Make_Scaffold_Index(rdmol,df):
    #print(df)
   
    T_re = []
    for row in df.itertuples():
        #print 'row:',list(row)
        try:
            p_smi = Make_Canonical_SMI(row[2])
        except:
            print 'Error in making canonical smi'
            return -1
        try:
            patt = Chem.MolFromSmiles(p_smi)
            '''
            print p_smi
            idx =0
            for atom in patt.GetAtoms():
                print idx, atom.GetSymbol()   
                idx+=1
           '''
        except:
            print 'Error in Mol from smiles'
            return -1
        try:
            re , mapping = is_substructure_mol(rdmol, patt)
        except:
            print 'Error in is_substructure_mol'
            return -1 
        #print p_smi, re, mapping

        idx =0 
        for are in re:
            #print are
            tmp_re = []
            tmp_re= [row[1]] #list(row[1])
            tmp_re.append(p_smi)
            tmp_re.append(are)
            tmp_re.append(mapping[idx])
            idx+=1
            T_re.append(tmp_re)

    df = pd.DataFrame.from_records(T_re,columns=['Ring_Num','smiles','MIdx','Map_Dic'])
    df = df.sort_values(by=['Ring_Num'],ascending=True)
    #print df[['smiles','MIdx']]
    df = df.sort_values('Ring_Num', ascending=True).drop_duplicates('MIdx')
    #print df[['smiles','MIdx']]
    return df 


def select_mapping(re,mapping,Terminal_Scaffold_idx):

    #print 'select_mapping re:',re 
    #print 'select_mapping mapping:', mapping
    #print 'select_mapping terminal idx:',Terminal_Scaffold_idx

    tmp_re = list(copy.deepcopy(re))
    re_list = list(re)

    i=0
    for are in tmp_re:
        are_list = list(are)
        #print are_list 
        if are_list in Terminal_Scaffold_idx:
            del re_list[i]
            del mapping[i] 
        i+=1

    #print re_list
    #print mapping
    #print 'HIHI'
    
    return tuple(re_list),mapping


def Attach_hand(m,smi):
    
    p_smi = Make_Canonical_SMI(smi)
    print p_smi
    patt = Chem.MolFromSmiles(p_smi)
    sma = Chem.MolToSmarts(patt)
    patt_1 = Chem.MolFromSmarts(sma)
    match_list = m.GetSubstructMatches(patt_1)
    
    print 'attach_hand:',match_list

    if len(match_list) == 0:
        core = MurckoScaffold.GetScaffoldForMol(m)
        t_smi = Chem.MolToSmiles(core)
        #print t_smi
        core1 = MurckoScaffold.GetScaffoldForMol(patt)
        match_list = core.GetSubstructMatches(core1)
        #print match_list 
        m = core
        smi = p_smi

    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    dic_atom_sym =  Extract_Atom_sym(m)

    mw = Chem.RWMol(patt)
    list_re_tmp =[]
    if len(match_list)>0:
            X = list(match_list)
            atom_list = list(X[0])
            atom_set = set(X[0])
            #print atom_list,atom_set

            i = 0
            brench = 0

            for atom_idx in atom_list:
                #print atom_idx
                nei_atom = dic_atom_neig_sym[atom_idx]
                nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
                #print nei_atom,nei_atom_idx

                if len(nei_atom_idx - atom_set) != 0:
                    brench_diff = len(nei_atom_idx - atom_set)
                    #brench +=1 
                    brench = brench + brench_diff
                    #print 'Hi',nei_atom_idx
                    is_idx=list(nei_atom_idx - atom_set)
                    #print 'Hi',is_idx
                    for tidx in range(0,brench_diff):
                        ais_idx = list(is_idx).pop()
                        #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                        btype = m.GetBondBetweenAtoms(atom_idx,ais_idx).GetBondType()
                        atype = dic_atom_sym[ais_idx]
             
                        atom_num = m.GetAtomWithIdx(ais_idx).GetAtomicNum()
                        #print 'atomic num:',m.GetAtomWithIdx(is_idx).GetAtomicNum()

                        #print 'Hi',i,atom_idx,is_idx,btype,atype
                        ire = mw.AddAtom(Chem.Atom(atom_num))
                        #print 'ire:',ire
                        #print i,ire
                        bre = mw.AddBond(i,ire,btype)
                        #print 'bre:',bre
                        #print 
                i+=1 
    list_re = []
    try:
        Chem.SanitizeMol(mw)
        asmi = Chem.MolToSmiles(mw)
        #print asmi
        list_re_tmp.append(asmi)
        list_re_tmp.append(smi)
        list_re_tmp.append(brench)
        list_re.append(list_re_tmp)
        return list_re
    except:
        list_re_tmp.append(0)
        list_re.append(list_re_tmp)
        return list_re


def Attach_hand_2(m,smi):
    try:
        p_smi = Make_Canonical_SMI(smi)
    except:
        return 2
    #print p_smi
    patt = Chem.MolFromSmiles(p_smi)
    sma = Chem.MolToSmarts(patt)
    patt_1 = Chem.MolFromSmarts(sma)
    match_list = m.GetSubstructMatches(patt_1)
    #print match_list
   

    #Show_Index_Information(p_smi)
    #return

    map_dic = {}

    re,mapping = is_substructure_mol(m, patt_1)
    map_dic = mapping[0]
    match_list = re
    #print re,map_dic

    if len(match_list)==0:
        re,mapping = is_substructure_mol(m, patt_1)
        #print re,mapping
        if len(re)==1:
            map_dic = mapping[0]
            #print 'ok'

        #print 'In attach_hand:',re
        if len(re)!=0:
            match_list = re

    #print 'attach_hand:',match_list

    if len(match_list) == 0:
        core = MurckoScaffold.GetScaffoldForMol(m)
        t_smi = Chem.MolToSmiles(core)
        #print t_smi
        core1 = MurckoScaffold.GetScaffoldForMol(patt)
        match_list = core.GetSubstructMatches(core1)
        #print match_list 
        m = core
        smi = p_smi

    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    dic_atom_sym =  Extract_Atom_sym(m)

    mw = Chem.RWMol(patt)
    list_re_tmp =[]
    if len(match_list)>0:
            X = list(match_list)
            atom_list = list(X[0])
            atom_set = set(X[0])
            #print atom_list,atom_set

            i = 0
            brench = 0

            for atom_idx in atom_list:
                #print atom_idx
                if m.GetAtomWithIdx(atom_idx).IsInRing():
                    pass
                    #print 'In Ring',atom_idx
                else:
                    # In case of out ring atom, pass the brench check
                    #print 'Out Ring',atom_idx
                    #i+=1
                    continue
                    
                nei_atom = dic_atom_neig_sym[atom_idx]
                nei_atom_idx = set(dic_atom_neig_idx[atom_idx])
                #print nei_atom,nei_atom_idx
                
                # Check a atome has out node
                if len(nei_atom_idx - atom_set) != 0:
                    #print atom_idx,map_dic[atom_idx]
                    brench_diff = len(nei_atom_idx - atom_set)
                    #brench +=1 
                    brench = brench + brench_diff
                    #print 'Hi',nei_atom_idx
                    is_idx=list(nei_atom_idx - atom_set)
                    #print 'Hi',is_idx
                    for tidx in range(0,brench_diff):
                        ais_idx = list(is_idx).pop()
                        #if (btype==Chem.rdchem.BondType.AROMATIC or btype==Chem.rdchem.BondType.DOUBLE or btype==Chem.rdchem.BondType.TRIPLE):
                        btype = m.GetBondBetweenAtoms(atom_idx,ais_idx).GetBondType()
                        atype = dic_atom_sym[ais_idx]
             
                        atom_num = m.GetAtomWithIdx(ais_idx).GetAtomicNum()
                        #print 'atomic num:',m.GetAtomWithIdx(is_idx).GetAtomicNum()

                        #print 'Hi',i,atom_idx,is_idx,btype,atype
                        ire = mw.AddAtom(Chem.Atom(atom_num))
                        #print 'ire:',ire
                        #print i,ire
                        bre = mw.AddBond(map_dic[atom_idx],ire,btype)
                        #print 'bre:',bre
                        #print 
                i+=1 
    list_re = []
    try:
        Chem.SanitizeMol(mw)
        asmi = Chem.MolToSmiles(mw)
        #print asmi
        list_re_tmp.append(asmi)
        list_re_tmp.append(smi)
        list_re_tmp.append(brench)
        list_re.append(list_re_tmp)
        return list_re
    except:
        list_re_tmp.append(0)
        list_re.append(list_re_tmp)
        return list_re



def Merge_Scaffold(t_df,list_re,total_index):

    #print t_df[['smiles','MIdx']]
    #print total_index

    tmp_list = []
    t_num_ring=0
    for alist in list_re:
        tmp_list.append(alist[1])
        try:
            amol = readstring('smi',alist[1])
        except:
            print 'error in readstring'
            return 0
        list_ring = amol.sssr
        num_ring = len(list_ring)
        t_num_ring = t_num_ring+num_ring


    tmp_df = t_df[t_df.Ring_Num == str(t_num_ring)]
    #print tmp_df[['smiles','MIdx']]

    for row in tmp_df.itertuples():
        midx = row[3]
        #print 'midx:',midx
        if len(total_index-set(midx))==0:
            return row[2]



def powerset(s):
    tmp_list=[]
    x = len(s)
    for i in range(1 << x):
        #print [s[j] for j in range(x) if (i & (1 << j))]
        alist = [s[j] for j in range(x) if (i & (1 << j))]
        if len(alist) >=2:
            tmp_list.append(alist)
    return tmp_list


def Extract_Atoms_Set(pybelmol):

    Atom_Set = set()
    for atom in pybelmol:
        Atom_Set.add(atom.type)

    return Atom_Set
        


def Extract_Atoms_List(pybelmol):

    Atom_List = []
    for atom in pybelmol:
        Atom_List.append(atom.type)

    return Atom_List


def Matching_Using_MakeScaffoldGeneric(rdmol,p_mol):

    # Code: Matching using MakeScaffoldGeneric
    try:
        core = MurckoScaffold.GetScaffoldForMol(rdmol)
        fw = MurckoScaffold.MakeScaffoldGeneric(core)
    except:
        return -1

    try:
        patt_g = MurckoScaffold.MakeScaffoldGeneric(p_mol)
    except:
        return -1

    match_list = fw.GetSubstructMatches(patt_g)
    return match_list



def Is_terminal_Ring(rdmol,asmi,dic_atom_neig_idx):

    #p_smi = Make_Canonical_SMI(asmi)
    #print 'p_smi:',p_smi
    patt1 = Chem.MolFromSmiles(asmi)
    #patt1 = Chem.MolFromSmiles(p_smi)
    '''
    sma = Chem.MolToSmarts(patt1)
    patt = Chem.MolFromSmarts(sma)
    '''
    match_list = rdmol.GetSubstructMatches(patt1)
    #print match_list

    ssr = Chem.GetSymmSSSR(patt1)

    if len(match_list)==0 and len(ssr)>=2:
    #if len(match_list)==0 and len(ssr)>=1:

        re = Matching_Using_MakeScaffoldGeneric(rdmol,patt1)
        match_list2 = []
        if re == -1:
            pass 
        else:
            match_lsit2 = re

        '''
        # Test_Code: Matching using MakeScaffoldGeneric
        core = MurckoScaffold.GetScaffoldForMol(rdmol)
        fw = MurckoScaffold.MakeScaffoldGeneric(core)
        #gf = Chem.MolToSmiles(fw)
        #print gf

        #patt_2 = MurckoScaffold.GetScaffoldForMol(patt1)
        #patt_g = MurckoScaffold.MakeScaffoldGeneric(patt_2)
        patt_g = MurckoScaffold.MakeScaffoldGeneric(patt1)
        match_list2 = fw.GetSubstructMatches(patt_g)
        #gf2 = Chem.MolToSmiles(patt_g)
        #print 'gf2:',gf2
        #print match_list2
        '''
        if len(match_list2)!=0:
            match_list = match_list2

    #print match_list

    if len(match_list)!= 0:
        for alist in match_list:
            X = list(match_list)
            atom_list = list(X[0])
            #print atom_list
            brench = 0
            for atom_idx in atom_list:
                #print atom_idx,len(dic_atom_neig_idx[atom_idx]), dic_atom_neig_idx[atom_idx]
                nei_atoms = dic_atom_neig_idx[atom_idx]
                for anei_atom in nei_atoms:
                    if anei_atom not in atom_list:
                        brench+=1
        #print 'the number of brench:',brench
        if brench == 1:
            return -1
        else:
            #print '1'
            return 1
    else:
        #print '0'
        return 0

    return



def Is_terminal_Ring_3(rdmol,Midxs,dic_atom_neig_idx):


    #print 'Is terminal:',type(Midxs),Midxs

    atom_list = list(Midxs)
    brench = 0
    for atom_idx in atom_list:
        nei_atoms = dic_atom_neig_idx[atom_idx]
        for anei_atom in nei_atoms:
            b_type = rdmol.GetBondBetweenAtoms(atom_idx,anei_atom).GetBondType()
            if b_type != Chem.rdchem.BondType.DOUBLE:
                if anei_atom not in atom_list:
                    brench+=1
    #print 'brench:',brench

    if brench <=1:
        return 1
    else:
        return -1



def Is_terminal_Ring_2(rdmol,Midxs,dic_atom_neig_idx):

    ''' 
    ssr = Chem.GetSymmSSSR(rdmol)
    for aring in range(0,len(ssr)):
        print list(ssr[aring])
    '''
    

    '''
    #p_smi = Make_Canonical_SMI(asmi)
    #print 'p_smi:',p_smi
    patt1 = Chem.MolFromSmiles(asmi)
    #patt1 = Chem.MolFromSmiles(p_smi)
    
    patt1 = Chem.MolFromSmiles(asmi)
    match_list = rdmol.GetSubstructMatches(patt1)
    #print 'match_list :',match_list
    

    if len(match_list) == 0:
        re, mapping = is_substructure_mol(rdmol, patt1)
        if re == -1 and mapping ==-1:
            return -999

        match_list = re
        #print 'Is_terminal_Ring_2: ',type(match_list),match_list


    ssr = Chem.GetSymmSSSR(patt1)
    #ln_ring_asmi = len(ssr)
    #for aring in ssr:
        

    if len(match_list)==0 and len(ssr)>=2:
        re = Matching_Using_MakeScaffoldGeneric(rdmol,patt1)
        match_list2 = []
        if re == -1:
            pass 
        else:
            match_lsit2 = re

        if len(match_list2)!=0:
            match_list = match_list2


    #print 'HIHI',asmi,match_list
    '''

    #print 'Is terminal:',type(Midxs),Midxs

    atom_list = list(Midxs)
    brench = 0
    for atom_idx in atom_list:
        nei_atoms = dic_atom_neig_idx[atom_idx]
        for anei_atom in nei_atoms:
            b_type = rdmol.GetBondBetweenAtoms(atom_idx,anei_atom).GetBondType()
            if b_type != Chem.rdchem.BondType.DOUBLE:
                if anei_atom not in atom_list:
                    brench+=1
    #print 'brench:',brench

    if brench <=1:
        return 1
    else:
        return -1


    '''
    re_brench=[]
    if len(match_list)!= 0:
        brench = 0
        for alist in match_list:
            X = list(alist)
            atom_list = list(X)
            #print atom_list
            #print alist
            reFusion = Is_Ring_In_Fusion(rdmol,atom_list,dic_atom_neig_idx)
            #print 'Is_Ring_In_Fusion:', reFusion
            if reFusion == 1:
                #print 'In Fusion'
                continue
            #sys.exit(1)
            #brench = 0
            for atom_idx in atom_list:
                #print atom_idx,len(dic_atom_neig_idx[atom_idx]), dic_atom_neig_idx[atom_idx]
                nei_atoms = dic_atom_neig_idx[atom_idx]
                for anei_atom in nei_atoms:
                    #print atom_idx,anei_atom
                    b_type = rdmol.GetBondBetweenAtoms(atom_idx,anei_atom).GetBondType()
                    #print type(b_type),b_type
                    if b_type != Chem.rdchem.BondType.DOUBLE:
                        if anei_atom not in atom_list:
                            brench+=1
            #print

            #print 'brench:',brench
            #print 'the number of brench:',brench
            tmp_re=[]

            # 1 means terminal node
            if brench == 1:
                tmp_re.append(1)
                tmp_re.append(alist)
                re_brench.append(tmp_re)
            else:
                #print '1'
                tmp_re.append(-1)
                tmp_re.append(alist)
                re_brench.append(tmp_re)
            brench=0
            #print re_brench

        #print 're brench:',re_brench,'\n' 
        if len(re_brench)==0:
            return 1

        if len(re_brench)==1:
            return re_brench[0][0]
        else:
            for are in re_brench:
                re_b=are[0]
                if re_b == 1:
                    re_b ==1
            # if all ring are terminal ring
            if re_b == 1:
                return 1
    else:
        #print '0'
        return 0

    return re_brench
    '''



def Is_Ring_In_Fusion(m,idx_list,dic_atom_neig_idx):

    R_count=0
    #print 'idx_list:',idx_list
    idx_set = set(idx_list)

    total_diff_list =[]
    for aidx in idx_list:
        nei_idx_list = dic_atom_neig_idx[aidx]
        #print nei_idx_list,idx_set 
        diff_list = list(set(nei_idx_list)-idx_set)
        #print 'diff_list:',diff_list
        if len(diff_list)>0:
            #print diff_list[0]
            total_diff_list.append(diff_list[0])
            
    #print 'total_diff_list:',total_diff_list
    re = Check_Common_Index_In_ARing(m,total_diff_list)
    #print 'Is fusion?:',re,'\n\n\n'
    if re == 1:
        return 1
    else:
        return -1


def Check_Common_Index_In_ARing(m,idx_list):

    ssr = Chem.GetSymmSSSR(m)
    for aring in ssr:
        #print 'ring list:',list(aring)
        #a= set(idx_list)-set(list(aring))
        inter_s= set(idx_list).intersection(set(list(aring)))
        #print 'In Check_Common_Index_In_ARing:a',inter_s
        # 1 is fusion ring
        if len(inter_s)>=2:
            return 1
    # -1 is not fusion ring    
    return -1


def Extract_Atom_Neighbors(m):

    #m = Chem.MolFromSmiles(asmi)

    dic_atom_neig_idx = {}
    dic_atom_neig_sym = {}

    for atom in m.GetAtoms():
        br1 = [x.GetIdx() for x in atom.GetNeighbors()]
        br2 = [x.GetSymbol() for x in atom.GetNeighbors()]
        dic_atom_neig_idx[atom.GetIdx()] = br1
        dic_atom_neig_sym[atom.GetIdx()] = br2

    #print dic_atom_neig_idx
    #print dic_atom_neig_sym 

    return dic_atom_neig_idx, dic_atom_neig_sym



#def Extract_Atom_sym(asmi):
def Extract_Atom_sym(m):

    #m = Chem.MolFromSmiles(asmi)
    dic_atom_sym = {}

    for atom in m.GetAtoms():
        dic_atom_sym[atom.GetIdx()] = atom.GetSymbol()

    return dic_atom_sym



def Make_Canonical_SMI(asmi):

    amol = readstring('smi',asmi)
    my_smi =amol.write(format='smi')
    try:
        ca_smi = Chem.MolToSmiles(Chem.MolFromSmiles(my_smi))
    except:
        print 'rdkit error: Cani\'t kekulize mol'
        print 'converting using pybel'
        ca_smi =amol.write(format='smi')
        #sys.exit(1)
        return ca_smi

    return ca_smi


def Is_Fusion_Ring(asmi):

    #df = Extract_SubScaffold(asmi)

    my_smi = Make_Canonical_SMI(asmi)
    m = Chem.MolFromSmiles(my_smi)
    #dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    dic_atom_sym =  Extract_Atom_sym(m)


    amol = readstring('smi',asmi)
    list_ring=amol.sssr
    num_ring = len(list_ring)
    #print asmi,num_ring
    #print dic_atom_sym
    
    tmp_set = set()

    # Extracrt Ring_index !!!!!!!
    r1 = set(list_ring[0]._path)
    r2 = set(list_ring[1]._path)
    
    tmp_set = r1 & r2
    
    #print r1,r2
    #print tmp_set

    df = Extract_SubScaffold(asmi)
    print df
    t_smi = ''
    re_list =2
    if len(tmp_set)>0:
        #tmp_df = df.loc[df['Ring_Num']==str(2)]
        tmp_df = df[df['Ring_Num']==str(2)]
        #print tmp_df
        for row in tmp_df.itertuples():
            # selecting SMILES
            t_smi = row[2]
            #print t_smi

            my_smi = Make_Canonical_SMI(asmi)
            mol = Chem.MolFromSmiles(my_smi)

            re_list = Attach_hand(mol,t_smi)
            #print re_list

        return re_list

    else:
        tmp_df = df[df['Ring_Num']==str(1)]

        r1_smi =''
        r2_smi =''
        idx = 1
        for row in tmp_df.itertuples():
            if idx == 1:
                r1_smi = row[2]
            if idx == 2:
                r2_smi = row[2]
            idx+=1

        #print r1_smi, r2_smi

        R1_NC = r1_smi.count('N') + r1_smi.count('n')
        R1_OC = r1_smi.count('O') + r1_smi.count('o')

        R2_NC = r2_smi.count('N') + r2_smi.count('n')
        R2_OC = r2_smi.count('O') + r2_smi.count('o')

        R1_AF=R1_NC+R1_OC
        R2_AF=R2_NC+R2_OC
        #print R1_AF,R2_AF

        my_smi = Make_Canonical_SMI(asmi)
        mol = Chem.MolFromSmiles(my_smi)

        if R1_AF > R2_AF:
            re_list = Attach_hand(mol,r1_smi)

        if R1_AF < R2_AF:
            re_list = Attach_hand(mol,r2_smi)

        if R1_AF == R2_AF:
            tmp_df = df[df['Ring_Num']==str(2)]
            for row in tmp_df.itertuples():
                # selecting SMILES
                t_smi = row[2]
            re_list = Attach_hand(mol,t_smi)
        
        #print re_list

        return re_list



def Check_Ring_Total_Ring_Num(smi):

    amol = readstring('smi',smi)
    list_ring=amol.sssr
    num_ring = len(list_ring)

    return num_ring



def Less_Than_Two_Ring(smi):

    num_ring = Check_Ring_Total_Ring_Num(smi)
    #print num_ring

    if num_ring == 1:
        df = Extract_SubScaffold(smi)
        tmp_df = df[df['Ring_Num']==str(1)]
        for row in tmp_df.itertuples():
            # selecting SMILES
            t_smi = row[2]
            my_smi = Make_Canonical_SMI(smi)
            mol = Chem.MolFromSmiles(my_smi)
            re_list = Attach_hand(mol,t_smi)
            #print re_list
            return re_list

    if num_ring == 2:
        re_list = Is_Fusion_Ring(smi)
        return re_list

    else:
        return 0
        

def Show_Index_Information(asmi):

    asmi = Make_Canonical_SMI(asmi)
    amol = Chem.MolFromSmiles(asmi)

    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(amol)
    dic_atom_sym = Extract_Atom_sym(amol)
    
    print
    print 'dic_atom_neig_idx:',dic_atom_neig_idx
    print 'dic_atom_neig_sym:',dic_atom_neig_sym
    print 'dic_atom_sym:',dic_atom_sym
    print




def Check_Same_Scaffold(ssmi,tsmi):

    # ssmi: scaffold smiles
    # tsmi: target smiles

    ssmi = Make_Canonical_SMI(ssmi)
    tasmi = Make_Canonical_SMI(tsmi)

    tm_mol = Chem.MolFromSmiles(tasmi)
    patt = Chem.MolFromSmiles(ssmi)

    ###############################
    # For test driver
    #Show_Index_Information(tsmi)
    #Show_Index_Information(ssmi) 
    ##############################


    match_list = tm_mol.GetSubstructMatches(patt)
    match_set = set(match_list[0])
    #print 'match set:',match_set

    p_ssr = Chem.GetSymmSSSR(patt)
    s_ring_number = len(p_ssr)
    #print s_ring_number

    ssr = Chem.GetSymmSSSR(tm_mol) 
    '''
    print len(ssr)
    for aring in ssr:
        print set(list(aring))
    #return
    print '\n\n'
    #return
    '''

    ring_count=0
    for aring in ssr:
        #print 'ring idex:',set(list(aring))
        aring_path = set(list(aring)) 
        tmp_set = (aring_path & match_set)
        #print tmp_set
        if len(aring_path & match_set) > 1:
            #print 'found'
            ring_count +=1
        #print

    #print ring_count

    if s_ring_number == ring_count:
        # Good
        return 1
    else:
        # Bad
        return -1



    '''
    ############
    # Bad code because of wrong index: rdkit index is different from pybel
    ############

    # for num. of ring of scaffold 
    smol = readstring('smi',ssmi)
    list_ring=smol.sssr
    s_ring_number = len(list_ring)
    print s_ring_number


    amol = readstring('smi',tsmi)
    list_ring=amol.sssr
    num_ring = len(list_ring)
    print num_ring
    #print list_ring
    ring_count = 0 
    
    for aring in list_ring:
        print 'ring idex:',aring._path
        aring_path = set(aring._path)
        tmp_set = (aring_path & match_set)
        print tmp_set
        if len(aring_path & match_set) > 1:
            #print 'found'
            ring_count +=1
        print

    print ring_count

    if s_ring_number == ring_count:
        return 1
    else:
        return -1
    '''


def topology_from_rdkit(rdkit_molecule):

    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx())

        # Add the bonds as edges
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())

    return topology


def is_isomorphic(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2)


def Make_Adjacency_Matrix(m):
    
    #smi = Make_Canonical_SMI(smi)
    #m = Chem.MolFromSmiles(smi)
    try:
        ln_smi = m.GetNumAtoms()
    except:
        return -1
    npa = np.zeros((ln_smi,ln_smi))

    '''
    dic_atom_neig_idx, dic_atom_neig_sym =  Extract_Atom_Neighbors(m)
    dic_atom_sym =  Extract_Atom_sym(m)

    print dic_atom_neig_idx
    print
    print dic_atom_neig_sym
    print 
    print dic_atom_sym
    print
    '''

    for atom in m.GetAtoms():
        for bond in atom.GetNeighbors():
            #print atom.GetIdx(),bond.GetIdx()
            npa[atom.GetIdx()][bond.GetIdx()]=1
            npa[bond.GetIdx()][atom.GetIdx()]=1

    #print npa
    #print npa.shape
    #print type(npa)
    return npa



def Add_Node_Attribute(G,mol):
    
    for atom in mol.GetAtoms():
        #G.nodes[atom.GetIdx()]['atom']=atom.GetAtomicNum()
        G.nodes[atom.GetIdx()]['atom']=atom.GetSymbol()   

    return G



def Match_Subgraph(tnp,tmol,pnp,pmol):

    #G1 = nx.from_numpy_matrix(np.array(a), create_using=nx.MultiGraph())
    #G2 = nx.from_numpy_matrix(np.array(b), create_using=nx.MultiGraph())

    #print a,b
    GT = nx.from_numpy_matrix(tnp)
    GT = Add_Node_Attribute(GT,tmol)

    GP = nx.from_numpy_matrix(pnp)
    GP = Add_Node_Attribute(GP,pmol)

    #print 'G1 Node:',G1.nodes()
    #print nx.get_node_attributes(G1,'atom')

    #print 'G2 node:',G2.nodes()
    #print nx.get_node_attributes(G2,'atom')


    GM = isomorphism.MultiGraphMatcher(GT,GP)
    #GM = isomorphism.MultiGraphMatcher(G1,G2,edge_match=lambda x, y: x[0]['weight'] == y[0]['weight'] )
    print(GM.subgraph_is_isomorphic())
    #print(GM.mapping)

    
    key_list = []
    tmp_list = []
    map_dic_list = []

    for mapping in GM.subgraph_isomorphisms_iter():
        print mapping
        #print type(mapping)
        keys = mapping.keys()
        keys.sort()
        #print keys
        if keys not in key_list:
            key_list.append(keys)
            tkeys = tuple(keys)
            #print tkeys
            tmp_list.append(tkeys)
            map_dic_list.append(mapping)
        #print
    #print key_list
    #print tmp_list
    re_list = tuple(tmp_list)
    #print re_list 

    #GM = isomorphism.MultiGraphMatcher(G2,G1)
    #GM = isomorphism.MultiGraphMatcher(G1,G2,edge_match=lambda x, y: x[0]['weight'] == y[0]['weight'] )
    #print(GM.subgraph_is_isomorphic())
    #print(GM.mapping)

    return re_list,map_dic_list




def Match_Subgraph_2(tnp,tmol,pnp,pmol):

    GT = nx.from_numpy_matrix(tnp)
    GT = Add_Node_Attribute(GT,tmol)

    GP = nx.from_numpy_matrix(pnp)
    GP = Add_Node_Attribute(GP,pmol)

    #print nx.get_node_attributes(G1,'atom')
    #print nx.get_node_attributes(G2,'atom')

    GM = isomorphism.MultiGraphMatcher(GT,GP)
    #GM = isomorphism.MultiGraphMatcher(G1,G2,edge_match=lambda x, y: x[0]['weight'] == y[0]['weight'] )
    #print(GM.subgraph_is_isomorphic())
    #print(GM.mapping)

    if GM.subgraph_is_isomorphic() == False:
        return [],[]    


    key_list = []
    tmp_list = []
    map_dic_list = []

    for mapping in GM.subgraph_isomorphisms_iter():
        #print mapping
        GT_atom,GP_atom = '',''
        for key in mapping.keys():
            #print key,GT.nodes[key],GT.nodes[key]['atom']
            #print mapping[key],GP.nodes[mapping[key]]['atom']

            # Compare the atom using node's attribute
            if GT.nodes[key]['atom']!=GP.nodes[mapping[key]]['atom']:
                break
            else:
                GT_atom=GT_atom+GT.nodes[key]['atom']
                GP_atom=GP_atom+GP.nodes[mapping[key]]['atom']

            #print GP_atom
            if len(mapping.keys()) == len(GP_atom): 
                #print GP_atom 
                keys = mapping.keys()
                keys.sort()
                #print keys
                if keys not in key_list:
                    key_list.append(keys)
                    tkeys = tuple(mapping.keys())
                    tmp_list.append(tkeys)
                    map_dic_list.append(mapping)
    re_list = tuple(tmp_list)
    return re_list,map_dic_list



def is_substructure_smi(target_smi, pattern_smi):

    try:
        target_smi = Make_Canonical_SMI(target_smi)
        target_mol = Chem.MolFromSmiles(target_smi)

        #G1 = topology_from_rdkit(target_mol)
        #print 'Test:',G1.nodes()

        pattern_smi = Make_Canonical_SMI(pattern_smi)
        pattern_mol = Chem.MolFromSmiles(pattern_smi)

        #G2 = topology_from_rdkit(pattern_mol)
        #print 'Test:',G2.nodes()
        print target_mol, pattern_mol
    except:
        return -9

    tnp = Make_Adjacency_Matrix(target_mol)
    pnp = Make_Adjacency_Matrix(pattern_mol)


    re,mapping = Match_Subgraph_2(tnp,target_mol,pnp,pattern_mol)
    #print 'In is_substructure_smi:',re,mapping
    #re,mapping = check_substructure_mol(re,mapping,target_mol, pattern_mol)
    #print 'In is_substructure_smi:',re,mapping

    return re,mapping



def is_substructure_mol(target_mol, pattern_mol):

    tnp = Make_Adjacency_Matrix(target_mol)
    pnp = Make_Adjacency_Matrix(pattern_mol)

    re,mapping = Match_Subgraph_2(tnp,target_mol,pnp,pattern_mol)
    #print 'In is_substructure_smi:',re,mapping
    #re,mapping = check_substructure_mol(re,mapping,target_mol, pattern_mol)
    #print 'In is_substructure_smi:',re,mapping

    return re,mapping



def check_substructure_mol(re_list,mapping,target_mol, pattern_mol):


    ###### !!!!! Not use !!!!!! #########

    # Compare the target atom and pattern atom using circular string matching
    tmp_re = list(copy.deepcopy(re_list))
    tmp_mapping = []
    #print 'tmp_re: type:',type(tmp_re)

    # For Pattern Atom
    if len(re_list)<=1:
        return re_list,mapping
    ln_pattern = pattern_mol.GetNumAtoms()
    patom=''

    # if pattern is single ring, check for the many single ring
    if ln_pattern <=6:
        for aidx in range(0,ln_pattern):
            patom=patom+pattern_mol.GetAtomWithIdx(aidx).GetSymbol()
        #print '  check_substructure_mol_pattern:', patom   

        # For Target Atom
        mapping_idx=0
        for are in re_list:
            tatom=''
            for aidx in are:
                tatom=tatom+target_mol.GetAtomWithIdx(aidx).GetSymbol()
            #print '  check_substructure_mol_target:', tatom   
            m_re = [m.group(0) for m in re.finditer(patom, tatom+tatom) if m.start() < len(tatom)]
            #print m_re
            if len(m_re)==0:
                tmp_re.remove(are)
            else:
                tmp_mapping.append(mapping[mapping_idx])
            mapping_idx+=1

        re_list = tuple(tmp_re)
        mapping = tmp_mapping

        return re_list, mapping 
    # pattern is more than 2 ring, pass the re_list without checking
    else:
        return re_list,mapping




def MD_Backbone_show():
    print '\'asmi\' means a smiles string'
    print 'Extract_BB(asmi),str'
    print 'ExtractM_BB(iPath),list'
    print 'Extract_CP(asmi),str'
    print 'Draw_BB_smi(o_path,f_name,smi),'
    print 'Extract_SubScaffold(asmi),'
    print 'Swap_Ring(asmi),'
    print 'Extract_FusionR(asmi),'
    print 'Read_SMILES_FILE(f_path),'
    print 'Search_ASMILES(asmi,m_type),'
    print 'Extract_SSSR_Idx(asmi),'
    print 'Extract_Inner_Scaffold(asmi),'
    print 'Extract_Inner_Scaffold_2(asmi)' # if there is no interConnected Ring return '0'
    print 'Extract_Inner_Scaffold_3(asmi)' # if there is no interConnected Ring return '0' and merge two scafolds into one scafold
    print 'Extract_Inner_Scaffold_4(asmi)' # bug fix: if a ligand has more than two hands at a index -> correct the num. of hands
    print 'Make_Canonical_SMI(asmi),'
    print 'topology_from_rdkit(rdkit_molecule),'
    print 'Show_Index_Information(asmi),'

    return



def main():

    # 2021.04.08.16:37

    # m = Chem.MolFromSmiles(myMolecule)

    parser=argparse.ArgumentParser()
    #parser.add_argument('-t',required=True, choices=['l','f'], default='n',  help='Input type: list(csv) or files(smi).')
    #parser.add_argument('-BB',required=True, choices=['y','n'], default='n',  help='Input type: Backbone or not backboen.')
    parser.add_argument('-i',required=False, help='Input list or path.')
    args=parser.parse_args()

    #iPath = args.i
    #ExtractM_BB(iPath)

    smi='COc1ccc(c2cc(C(=O)[O-])nc3c2c(C)nn3c2ccccc2)cc1'
    smi='S(=O)(=O)(NC(=O)c1cc(c(Cc2c3c(n(c2)C)ccc(NC(=O)OC2CCCC2)c3)cc1)OC)c1c(C)cccc1'
    smi='c1ccccc1S(=O)(=O)NC(=O)c2ccc(cc2)Cc4cnc3ccc(cc34)NC(=O)OC5CCCC5'
    smi='c1nccc(c1)NC(=O)C(=O)c3cn(c2ncccc23)Cc4ncccc4'
    #Extract_SubScaffold(smi)
    #Extract_FusionR(smi)
    #Swap_Ring(smi)


    smi = 'O=C(c1ccco1)N1CCN(c2nc(c3ccccc3)nc3ccccc23)CC1'
    #smi = 'C1=CC=C(C=C1)C3=C([N]2C=CN=CC2=N3)NC4=CC5=C(C=C4)OCCO5'

    # seperated ring 
    smi = 'O=C(c1occc1)N2CCNCC2'

    # fusion ring
    smi = 'c1ncc2ccccc2(n1)'


    smi = 'CC(C)N(O)C1=CC(O)=CC2=NC(C)=NC=C12'
    smi = 'COc1cc(C)oc1C(=O)N1CCNC(C1)[O](C)C'
    smi = 'O=C(c1occc1)N2CCNCC2'
    smi = 'O=C(C1CCCCC1)c1cccc1'
    smi = 'CC(=O)N1CCNCC1'
    smi = 'COC1CN(CCN1)C(C)=O'
    smi = 'Cc1c(C(=O)N2CCN(c3ccccc3O)CC2)cnn1c1ccccc1Cl'
    smi = 'CC1=NC2=C(N1)C=NC(C)=C2O'

    # 5NDZ_8UN
    #smi = 'c1cc2c(cc1C(=O)Nc1ccc(cc1)C#N)nc(n2[C@H](C1CCCCC1)C)c1cc2c(cc1Br)OCO2'

    # 5NDZ_8TZ
    smi = 'O[C@H](c1ncc[nH]1)c1ccc(F)cc1Cc1cncnc1'

    # Test
    #smi = 'CCc1ccc(c2cc(C(=O)Nc3cn(CC)nc3C(=O)NC3CCCC3)c3ccccc3n2)cc1'
    smi = 'c1ccc(cc1)c3nc2ccccc2c(c3)C(=O)Nc5cnnc5(C(=O)NC4CCCC4)'
    #smi = 'O=C1NC(=O)[C@@](c2ccccc2)(C2CCCCC2)N1'
    #smi = 'COCCSCCCSc1nnc(Cc2cccs2)n1C1CC1'
    smi = 'S(=O)(=O)(NC(=O)c1cc(c(Cc2c3c(n(c2)C)ccc(NC(=O)OC2CCCC2)c3)cc1)OC)c1c(C)cccc1'
    smi = 'n1(n(c(c(c1=O)CC[S@](=O)c1ccccc1)O)c1ccccc1)c1ccccc1'
    smi ='COc1ccc(c2cc(C(=O)[O-])nc3c2c(C)nn3c2ccccc2)cc1'
    smi ='CC(C)n1c(=O)[nH]c2c(C(=O)N3CCC(Cc4ccccc4)CC3)snc2c1=O'
    smi = 'COCC(=O)Nc1ccc2nc(N3CC[NH+](Cc4ccccc4F)CC3)cc(C(=O)[O-])c2c1'
    smi = 'O=C1C=C(CNc2cccc(C(F)(F)F)c2)N[C@H]2N=C(c3ccccc3)NN12'
    smi ='Oc1ccccc1c1nc2=c3cn[nH]c3=NCn2[nH]1'
    smi ='COc1ccc(c2c3CNN=c3[nH]c3c2oc2cc(O)ccc32)c(OC)c1'
    smi = 'CCn1cc(C[C@@H]2C(=O)Nc3ccccc23)c2ccccc12'
    smi ='Cc1oc2c(c(C)cc3OC(=O)[C@H](CC(=O)Nn4cnnc4)[C@@H](C)c23)c1C'
    smi = 'COc1ccc(c2cc(=O)c3c(OC)c(OC)c(OC)cc3o2)cc1O'
    smi ='Cc1nnc(CSC2=N[C@H]3SC[C@H](c4ccccc4)[C@@H]3C(=O)N2Cc2ccccc2)o1'
    smi ='CCn1cc(C(=O)NCc2cccnc2)c(=O)c2cc(S(=O)(=O)N3CCC(C)CC3)ccc12'
    smi ='CCc1ccccc1NC(=O)[C@@H]1C=C2[C@H](S1)c1ccccc1OC2'
    smi ='CC(C)(C)c1ccc(CN2[C@H](N)N(C[C@@H](O)COc3ccccc3)c3ccccc23)cc1'
    smi ='Cc1cccc(C[N]23CN2[C@H](NC(=O)c2cc(c4ccncc4)nc4c(Cl)cccc24)N3)c1'
    smi ='CCn1cc2c(n1)C(=O)N(c1ccccc1)[C@@H]1c3ccc(OC)c(OC)c3C(=O)N21'
    smi ='C1C(=NN=C1NC(=O)c1ccc(cc1)N1CCN2[C@H](C1)C2)c1cccc(NS(=O)(=O)c2ccccc2)c1 '
    smi = 'CCOC(=O)c1c(N)n(C2=N[C@@H](n3ncc(C(=O)OCC)c3N)c3c(N2)sc2c3CC(C)(C)OC2)nc1'
    smi = 'CCc1ccc(c2cc(C(=O)Nc3cn(CC)nc3C(=O)NC3CCCC3)c3ccccc3n2)cc1'
    smi ='O=C(CCC1CCN(c2cnc3ccccc3n2)CC1)NC[C@@H]1CCCO1'
    smi ='CCc1ncc(C(=O)N2CCC([C@H]3NNC[C@@H]3c3cccc(F)c3)CC2)cn1'
    smi = 'Cc1ccsc1[C@H]1Nc2ccc3ncccc3c2C2=C1C(=O)CCC2'
    smi ='O=S(=O)(NNc1ccc(Br)cc1)[C@H]1CONC1'
    smi = 'COc1ccc(c2cc(C(=O)[O-])nc3c2c(C)nn3c2ccccc2)cc1' ###############
    #smi =  'COC(=O)c1cc(c2ccncc2)nc2c1c(=O)n(C)c(=O)n2CC(C)C'
    #smi = 'CCN1CC(=O)Nc2cc(C(=O)Nc3cc(Cl)ccc3O)ccc12' ############
    #smi ='CC(C)n1nc(O)c2c(c3coc4ccc(Cl)cc4c3=O)c3oc4c(O)c(O)ccc4c3nc12' ###################
    smi = 'O=S(=O)(NNc1ccc(Br)cc1)[C@H]1CONC1'




    smi1 = 'Cc1nn(C)cc1N' 
    smi2 = 'C1CCCC1'
    smi3 = 'O=C(NC1CNNC1)c1ccnc2ccccc12'

    smi4 = 'C1CNNC1'
    smi5 = 'C1CCCC1'
    smi6 = 'C1CNCN1'
    smi7 = 'C1CNNC1'
    smi8 = 'c1ccccc1'
    smi9 = 'C1CCCCC1'

    #a,b =is_substructure_smi(smi, smi8)
    #print a,b
    #return


    '''
    npa = Make_Adjacency_Matrix(smi)
    npb = Make_Adjacency_Matrix(smi3)
    Match_Subgraph(npa,npb)
    return
    '''


    # Test
    '''
    f = Chem.CanonSmiles(smi)
    f_mol = Chem.MolFromSmiles(f)
    fg = topology_from_rdkit(f_mol)


    #s = Chem.CanonSmiles(smi1)
    s_mol = Chem.MolFromSmiles(smi1)
    sg = topology_from_rdkit(s_mol)


    print is_isomorphic(fg, sg)

    #ismags = nx.isomorphism.ISMAGS(fg,sg)
    #print ismags

    GM = isomorphism.GraphMatcher(sg, fg)
    GM = isomorphism.MultiGraphMatcher(sg, fg,edge_match=lambda x, y: x[0]['weight'] == y[0]['weight'] )
    print GM.is_isomorphic()
    '''

    #return
    #smi = 'O=C(CCc1ccccc1)NC(C(=O)NC(CCC2CC2)CC3CCCCC3)Cc4cnc[nH]4'
    #smi = 'S(=O)(=O)(c1ccc(cc1)CCNC(=O)N1CC(=C(C1=O)CC)C)NC(=O)N[C@H]1CC[C@@H](CC1)C'
    #smi = 'N1(CCN(CC1)c1c(cccc1)C)CCc1n2CCCCc2nn1' #OR <- rdkit error
    smi = 'O=C(NCCc2c[nH]c1ccccc12)C4(CCCCc3ccccc3)(CCCC4)' #OK
    smi = 'CCc1ccc(c2cc(C(=O)Nc3cn(CC)nc3C(=O)NC3CCCC3)c3ccccc3n2)cc1'
    smi = 'S(=O)(=O)(NC(=O)c1cc(c(Cc2c3c(n(c2)C)ccc(NC(=O)OC2CCCC2)c3)cc1)OC)c1c(C)cccc1'
    smi = 'COc1ccc(c2cc(C(=O)[O-])nc3c2c(C)nn3c2ccccc2)cc1'

    lines =[]
    ifile = args.i
    if ifile == None:
        lines.append(smi)
    else:
        fp_for_in = open(ifile,'r')
        lines = fp_for_in.readlines()
        fp_for_in.close()

    ln_line=len(lines)

    count=1
    for aline in lines:
        if not aline.startswith('#'):
            aline = aline.strip()
            token = aline.split()
            print str(count)+'/'+str(ln_line)+': Input SMI:',token[0]
            asmi = Make_Canonical_SMI(token[0])
            pmol = readstring('smi',asmi)
            asmi = pmol.write(format='smi')
            pBB = Extract_BB(asmi)
            print '------> BB:',pBB
            #return

            if Check_Ring_Total_Ring_Num(pBB) <=2:
                #print 'less than 2'
                #print pBB
                tmp_list =[]
                tmp_list.append(pBB)
                tmp_list.append(pBB)
                tmp_list.append(0)
                re_list =[]
                #re = Less_Than_Two_Ring(pBB)
                re_list.append(tmp_list)
                print re_list
            else:
                #print 'more than 2'
                #re = Extract_Inner_Scaffold_2(pBB)
                #print 'R2',re
                re = Extract_Inner_Scaffold_6(pBB)
                print '*'*100
                print '------>R3',re
                print '*'*100
                print '\n\n'
        count+=1

    return
    #print re

    #re_list = Extract_SSSR_Idx(smi)
    #print re_list

    # 8UN
    #t_smi = 'O=C(c1ccc2c(c1)nc1CCCCCn21)N1CCN(c2ccccn2)CC1'

    # 8TZ
    #t_smi = 'O=C1CCc2ccc3[nH]ccc3c12'

    # Test
    t_smi = 'O=C(c1cc(C2CC2)nc2ccccc12)N1CCC1'
    '''
    tasmi = Make_Canonical_SMI(t_smi)
    tm_mol = Chem.MolFromSmiles(tasmi)
    patt = Chem.MolFromSmiles(re[0][0])
    match_list = tm_mol.GetSubstructMatches(patt)
    print match_list
    print len(match_list[0])
    match_set = set(match_list[0])

    smol = readstring('smi',re[0][0])
    list_ring=smol.sssr
    print 'Source ring numnber:',len(list_ring)


    amol = readstring('smi',tasmi)
    list_ring=amol.sssr
    #list_ring=tm_mol.sssr
    num_ring = len(list_ring)
    #print num_ring
    #print list_ring
    ring_count = 0 
    for aring in list_ring:
        #print 'ring idex:',aring._path
        aring_path = set(aring._path)
        #tmp_set = (aring_path & match_set)
        #print tmp_set
        if len(aring_path & match_set) > 0:
            #print 'found'
            ring_count +=1
    print ring_count
    '''
    print re[0][0]

    #print(Check_Same_Scaffold(re[0][0],t_smi))

    smi = 'Cc1cc(C)c2ccccc2n1'
    t_smi = 'c1ccc2c(c3ncn[nH]3)cc(C3CC3)nc2c1'
    print(Check_Same_Scaffold(smi,t_smi))





    #re = Extract_Atom_Neighbors(smi)
    '''
    my_smi='c1ccsc1C5Nc3ccc2ncccc2c3C4=C5(C(=O)CCC4)'
    m = Chem.MolFromSmiles(my_smi)
    c_smi_1='C=1C=CNCC=1'
    c_smi_2='C=1C=CNCC=1'
    patt = Chem.MolFromSmiles(c_smi)
    list_match = m.GetSubstructMatches(patt)

    t_smi = 'O=C1CCCC2=C1CNc1ccc3ncccc3c12'
    t_smi = 'O=C4C=3CNc2ccc1ncccc1c2C=3CCC4'
    print 'test smi:',t_smi 
    m = Chem.MolFromSmiles(t_smi)

    p_smi = 'O=C2C=1CNC=CC=1CCC2'  
    #p_smi = 'O=C1CCCC2=C1CNC=C2'
    #p_smi = Make_Canonical_SMI(p_smi)
    print p_smi
    patt = Chem.MolFromSmiles(p_smi)
    sma = Chem.MolToSmarts(patt)
    patt = Chem.MolFromSmarts(sma)
    print sma
    print type(patt)
    match_list = m.GetSubstructMatches(patt)
    print match_list


    print 'col_smiles_list[l_idx]:',col_smiles_list[l_idx]
    p_smi = Make_Canonical_SMI(col_smiles_list[l_idx])
    print p_smi
    patt = Chem.MolFromSmiles(p_smi)
    match_list = tmp_mol.GetSubstructMatches(patt)
    print match_list
    '''



if __name__=="__main__":

    # Ver 20210526, swshin
    # Ver 20210623, 'Extract_Inner_Scaffold_2(asmi)'
    # Ver 20210818, 'Extract_Inner_Scaffold_3(asmi) and Merge two scaffold into one scaffold!!!!'
    # Ver 20210831, 'modifiy the Extract_Inner_Scaffold_4(asmi) and add !!!!'
    # Ver 20210914, 'Bug Fix!'
    # Ver 20210928, 'Using NetworkX, substructure-matching works good!
    # Ver 20211004, 'completely reversion works perfect!
    main()

