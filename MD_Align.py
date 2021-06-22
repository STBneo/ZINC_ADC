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
    subprocess.call(['java','-jar','sng.jar','generate','-o', t_dir+lid+'.tmp',t_dir+f_name], stdout=FNULL, stderr=subprocess.STDOUT)

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
    subprocess.call(['java','-jar','./Data/Sub_P/sng.jar','generate','-o', t_dir+lid+'.tmp',t_dir+f_name], stdout=FNULL, stderr=subprocess.STDOUT)
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
        sys.exit(1)
    
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
def Old_Align3D(re_dic,OID,zid,tmp_sdf):
   
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
    subprocess.call(['java','-jar','sng.jar','generate','-o', str(proc)+'.tmp',f_name], stdout=FNULL, stderr=subprocess.STDOUT)

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
    subprocess.call(['java','-jar','sng.jar','generate','-o', str(proc)+'.tmp',f_name], stdout=FNULL, stderr=subprocess.STDOUT)

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
    subprocess.call(['java','-jar','sng.jar','generate','-o', str(proc)+'.tmp',f_name], stdout=FNULL, stderr=subprocess.STDOUT)

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

######################################################################################

def Check_LSAlign():
    if not os.path.exists('./LSalign'):
        print ' This python program needs \'LSalign\'.'
        print ' Move the \'LSalign\' program at this fold and re excute this program.'
        return -1
    else:
        return 1


def Align3D(t_id,t_smi,q_id,q_smi):

    if Check_LSAlign()==-1:
        sys.exit(1)

    # For Check '.' in smiles for disconnection
    
    q_smi= Check_SMI_Dot(q_smi)

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

    return pcscore


def AlignM3D(t_id,t_smi,q_ids,q_smis):

    time1=time()
    if Check_sng() ==-1:  
        return 

    nslist=zip(q_ids,q_smis)
    ln_nslist=len(nslist)

    print ' --> Starting LSalign........'
    # Mulitprocessing
    manager = Manager()
    #Re_dic = manager.dict()
    Main_list = manager.list()

    Num_Of_CPU=multiprocessing.cpu_count()
    #Num_Of_CPU=2
    pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
    func=partial(Align3D_Core,t_id,t_smi,ln_nslist,nslist,Main_list)
    pool.map(func,nslist)
    pool.close()
    pool.join()

    Main_list=list(Main_list)
    df = pd.DataFrame.from_records(Main_list,columns=['Template','Query','PCScore'])
    df=df.sort_values(by=['PCScore'],ascending=False)
    #print df.head()
    print '\n --> Ending LSaling........'
    #print df.head()

    time2=time()
    #print 'Alignment time: '+str('{:.2f}'.format(time2-time1))+' sec.'

    return df


def Align3D_Core(t_id,t_smi,ln_nslist,nslist,Main_list,comp):

    if Check_LSAlign()==-1:
        sys.exit(1)

    aidx=nslist.index(comp)
    q_id = comp[0]
    q_smi = comp[1]
    
    print '     Processing....:'+str(aidx+1)+'/'+str(ln_nslist)+' ,mol:'+q_id+'\r', 
    sys.stdout.flush()

    re = Align3D(t_id,t_smi,q_id,q_smi)
    tmp_list=[t_id,q_id,re]
    Main_list.append(tmp_list)

    return 


def Check_SMI_Dot(t_smi):

    # For ChEMBL
    l_dot=[]
    ln_t_smi=len(t_smi)

    token=t_smi.split('.')
    if len(token)==1:
        return t_smi
    if len(token)>=2:
        token=list(set(token))
        for ele in token:
            if len(ele)<=7:
                token.remove(ele)
        if len(token)>2:
            print 'Error:',t_smi
            sys.exit(1)
        return token[0]


def MD_Backbone_show():
    print 'Check_LSAlign()'
    print 'Align3D(t_id,t_smi,q_id,q_smi),str'
    print 'AlignM3D(t_id,t_smi,q_ids,q_smis),str'
    print 'Align3D_Core(t_id,t_smi,ln_nslist,nslist,Main_list,comp),'
    print 'Check_SMI_Dot(t_smi),str'
    return


'''
def main():

    # 2021.04.08.16:37

    parser=argparse.ArgumentParser()
    #parser.add_argument('-t',required=True, choices=['l','f'], default='n',  help='Input type: list(csv) or files(smi).')
    #parser.add_argument('-BB',required=True, choices=['y','n'], default='n',  help='Input type: Backbone or not backboen.')
    #parser.add_argument('-i',required=True, help='Input list or path.')
    args=parser.parse_args()
    #iPath = args.i
    #ExtractM_BB(iPath)

    t_smi ='c1cc(c[nH]1)Cc2ccccc2'
    t_id  ='t.smi'

    l_q_smi =['O=C(OC1CCCC1)Nc2ccccc2','O=C(OC1CCCC1)Nc2ccccc2','O=C(OC1CCCC1)Nc2ccccc2','O=C(OC1CCCC1)Nc2ccccc2']
    l_q_id  =['q1.smi','q2.smi','q3.smi','q4.smi']

    #re=Align3D(t_id,t_smi,q_id,q_smi)
    #re=AlignM3D(t_id,t_smi,l_q_id,l_q_smi)

    t_smi='[Na+].c12c(c(c(c(n2)C(C)C)/C=C/[C@H](C[C@H](CC(=O)[O-])O)O)c2ccc(cc2)F)c(nn1c1ccc(cc1)OC)C'
    t_smi='c12c(c(c(c(n2)C(C)C)/C=C/[C@H](C[C@H](CC(=O)[O-])O)O)c2ccc(cc2)F)c(nn1c1ccc(cc1)OC)C.[Na+]'
    t_smi='[Na+].c12c(c(c(c(n2)C(C)C)/C=C/[C@H](C[C@H](CC(=O)[O-])O)O)c2ccc(cc2)F)c(nn1c1ccc(cc1)OC)C.[Na+]'
    t_smi='[Al+3].O=C1c2c([O-])cc(O)cc2OC(C1O)c1c(O)cc(O)cc1.O=C1c2c([O-])cc(O)cc2OC(C1O)c1c(O)cc(O)cc1.O=C1c2c([O-])cc(O)cc2OC(C1O)c1c(O)cc(O)cc1'
    print t_smi
    re=Check_SMI_Dot(t_smi)
    print re



if __name__=="__main__":

    # 2021.04.05
    # python 3D_Scan_Make_Scaffold_Backbone_list.Memo.v2.py -input ./Data/Input/ -top_p 0 -mins 1 -img n
    # python Extract_CP_SMILES.v2.py -t f -BB n -i ./MOA/DMC_ligand/
    main()
'''
