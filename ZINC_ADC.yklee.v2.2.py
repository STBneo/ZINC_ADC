import glob,sys,os,random,argparse,subprocess,pickle
from time import time
from datetime import datetime
import random
from operator import itemgetter

from rdkit import RDConfig
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Scaffolds import MurckoScaffold

import multiprocessing
from multiprocessing import Process, Manager
from functools import partial
import json
import copy
import re
import pandas as pd 
import traceback
import matplotlib
import matplotlib.pyplot as plt
# For background plotting
matplotlib.use("Agg")
plt.switch_backend("agg")
from pybel import *

# For Clustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import spectral_embedding
from scipy.cluster.hierarchy import dendrogram,linkage

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
from ADC_tools.MD_Backbone import *
from ADC_tools.MD_DB import *
from ADC_tools.MD_Align import *
from ADC_tools.MD_BA_class2 import *
from ADC_tools.MD_Output import *
from ADC_tools.MD_Cluster import *

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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



def processing_backbone(scaffold_backbone):


    type1 = '[*]'
    type2 = '([*])'

    old_backbone=scaffold_backbone
    scaffold_backbone = scaffold_backbone.replace(type2,'')
    scaffold_backbone = scaffold_backbone.replace(type1,'')
    #print 'Before: '+old_backbone+' ---> After: '+scaffold_backbone

    return scaffold_backbone



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
def OLD_Align3D(re_dic,OID,zid,tmp_sdf):
    
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


def Make_BB_Core(ln_slist,slist,Main_list,tmp_list,casmi):

    aidx=slist.index(casmi)
    #return
    asmi=casmi[0]
    asmi_path=casmi[1]
    #print asmi,asmi_path
    #print '\n\nProcessing....:'+str(aidx+1)+'/'+str(ln_slist)+' ,mol:'+asmi 
    print ' --> Processing....:'+str(aidx+1)+'/'+str(ln_slist)+' ,mol:'+asmi+'\r', 
    sys.stdout.flush()

    t_dir='./sng_tmp/' 

    proc = os.getpid()
    f_name=str(proc)+'.smi'
    #proc_name = current_process().name

    #print proc_name
    #print proc,asmi,f_name
    #return
    
    # write smiles
    t_path=os.path.join(t_dir,f_name)
    fp_for_out=open(t_path,'w')
    fp_for_out.write(asmi)
    fp_for_out.close()

    # excuate sng
    FNULL = open(os.devnull, 'w')
    subprocess.call(['java','-jar','sng.jar','generate','-o', t_dir+str(proc)+'.tmp',t_dir+f_name], stdout=FNULL, stderr=subprocess.STDOUT)
    #subprocess.call(['java','-jar','sng.jar','generate','-o', t_dir+str(proc)+'.tmp',t_dir+f_name], stderr=subprocess.STDOUT)

    re_name=str(proc)+'.tmp'
    t_path=os.path.join(t_dir,re_name)
    fp_for_in=open(t_path,'r')
    lines=fp_for_in.readlines()
    fp_for_out.close()
    #print lines

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

    scaffold_backbone = processing_backbone(scaffold_backbone)
    #print 'Backbone:',scaffold_backbone
    tmp_list=[scaffold_backbone,asmi,asmi_path]
    Main_list.append(tmp_list)

    '''
    # Extract chemical perporty
    tmp_list=[]
    try:
        amol = readstring('smi',scaffold_backbone)
    except:
        print '\nReading smiles Error :'+backbone
        return

    desc = amol.calcdesc(descnames=['MW', 'logP', 'HBA1', 'HBD','TPSA'])
    #print desc
    t= amol.sssr
    Num_Ring = len(t)

    tmp_list=[asmi,desc['MW'],desc['logP'],desc['HBA1'],desc['HBD'],desc['TPSA'],Num_Ring]
    Main_list.append(tmp_list)
    '''

def processing_backbone(scaffold_backbone):

    type1 = '[*]'
    type2 = '([*])'

    old_backbone=scaffold_backbone
    scaffold_backbone = scaffold_backbone.replace(type2,'')
    scaffold_backbone = scaffold_backbone.replace(type1,'')
    #print 'Before: '+old_backbone+' ---> After: '+scaffold_backbone

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
def yklee_work_help():
	print("""
	=================================================================
	#	About a_type : It is Working type                           #
	=================================================================
	#################################################################
	# a_type == 0 : Tier 1 Exact Matching                           #
	# a_type == 1 : Tier 1.5 Matching --> Inter-Ring Search Method  #
	# a_type == 2 : Tier 1.6 Matching --> Backbone Alignment Method #
    # a_type == 3 : ZINC_ADC Matching --> Extract ADC compound      #
	#################################################################
""")
def yklee_work(a_type):
	global T1_pcscore_cutoff,N_ZIDs_cutoff,ZIDs,out_csv_path
	T1_pcscore_cutoff = 0.98
	ring_cutoff = 1
	N_ZIDs_cutoff = 10000 #1500
	global oPath,DB_Path
	iPath = "./Data/Input/"
	oPath = "./Data/ADC_Output/Files/"
	rei_img_Path = "./Data/Re_Input/IMG/"
	out_csv_path = "./Data/ADC_Output/"

	DB_Path = "/ssd/swshin/1D_Scan.v2/Data/DB_Table/"
	#DB_Path = "/lwork02/yklee/DB_Table/"
	
	TR_MW = 38.0
	if os.path.exists(oPath):
		pass
	else:
		os.makedirs(oPath)
	
	if os.path.exists(rei_img_Path):
		pass
	else:
		os.makedirs(rei_img_Path)

	if_list = sorted(glob.glob(iPath + "*.smi"))
	if len(if_list) == 0:
		print("No Input Files")
		sys.exit(1)
	
	Main_list = []
	re = []
	sum_df = []
	
	alp = 0

	out_file1 = "./Data/ADC_Output/Out_Summary.csv"
	out_file2 = "./Data/ADC_Output/Error_SMILES.txt"
	for afile in if_list:
		alp += 1
		M_MW = 0.1
		ZIDs = []
		asmi = Read_SMILES_FILE(afile)
		Input_CP = Extract_CP(asmi)

		fn = os.path.basename(afile)
		file_name = os.path.splitext(fn)[0]
		##################
		# Activaion Part #
		##################
		if a_type == 0:
			bblist = working_a0(asmi,file_name)
			if bblist == -1:
				print("No ZIDs Match \nSMILES : %s"%asmi)
				pass
		elif a_type == 1:
			bblist = working_a1(asmi,file_name,afile)
			if bblist == -1:
				print("No BB Match \nSMILES : %s"%asmi)
				pass
		elif a_type == 2:
			total_df = working_a2(asmi,file_name,Input_CP)
			if total_df.empty:
				bblist = -1
			else:
				bblist = 1
		elif a_type == 3:
			total_df = working_ZDC2(asmi,file_name,afile,Input_CP) #working ZADC
			if total_df.empty:
				bblist = -1
			else: 
				bblist = 1
		else:
			sys.exit(1)

		######################
		# Make Result Format #
		######################
		if bblist == -1:
			pass
		else:
			if a_type == 0 or a_type == 1:
				ZIDs,zdf = BB_Align_Class_Search(Extract_BB(asmi),file_name,bblist,ZIDs,N_ZIDs_cutoff)
				if zdf.empty or zdf is None:
					sum_df = pd.DataFrame()
				else:
					sum_df = Write_Out_Summary(file_name,Extract_BB(asmi),zdf)
			
			else:
				sum_df = Write_Out_Summary(file_name,Extract_BB(asmi),total_df)

		################
		# Make Summary #
		################
		if bblist == -1:
			pass
		else:
			if sum_df.empty:
				with open(out_file2,"a") as W:
					W.write(file_name + "\t" + asmi + '\n')
			else:
				if not os.path.exists(out_file1):
					sum_df.to_csv(out_file1,index=False,mode="w")
				else:
					sum_df.to_csv(out_file1,index=False,mode="a",header=False)
		
def working_a0(asmi,file_name):
	######################
	# Tier 1 Exact Match #
	######################
	tmp_list = []
	id_smi = {}
	re_list = set()
	aBB = Extract_BB(asmi)
	ZIDs = T1_Class_Search(aBB,file_name,m_type=1)

	if ZIDs == -1:
		return -1
	else:
		tmp_df = pd.DataFrame(ZIDs,columns="ZID")
		tmp_df["Purchasability"] = ["Purchasable"]

	for i in tmp_df["ZID"]:
		id_smi = Fetch_SMILES_by_ID(i,id_smi,DB_Path)
	for i in list(set(id_smi.values())):
		re_list.add(Extract_BB(i)) # Backbone list

	return list(re_list)
def working_a1(asmi,file_name,afile):
	######################################
	# Tier 1.5 Match , Inter-Ring Search #
	######################################
	M_MW = 0.1
	TR_MW = 38.0
	Ok_flag = 0
	idx = 0
	aBB = Extract_BB(asmi) 
	re_list = working_a0(asmi,file_name)
	if re_list == -1:
		re_list = set()
	else:
		re_list = set(re_list)
	Total_re_list = T15_Class_Search_yklee(afile,TR_MW,M_MW)
	return Total_re_list
def BB_Query_parameters(asmi,mw_percent):
	omol = readstring("smi",asmi)
	csmi = "C"*(len(omol.atoms)-4) + "NNOO"
	CsmiCP = Extract_CP(csmi)
	InputCP = Extract_CP(asmi)
	tmp_li = [CsmiCP["MW"],InputCP["MW"]]

	Input_Num_Ring = InputCP["Ring"]
	Input_MW = np.float64(max(tmp_li))

	per_MW = Input_MW*np.float64(mw_percent)/np.float64(100.0)
	tper_MW = Input_MW*np.float64(mw_percent-1.0)/np.float64(100.0)
	if mw_percent == 1.0:
		up_MW = Input_MW + per_MW
		low_MW = Input_MW - per_MW
		return (low_MW,up_MW,Input_Num_Ring)

	else:
		uu_MW = Input_MW + per_MW
		ul_MW = Input_MW + tper_MW

		lu_MW = Input_MW - tper_MW
		ll_MW = Input_MW - per_MW

		return (ll_MW,lu_MW,ul_MW,uu_MW,Input_Num_Ring)
def working_a2(asmi,file_name,InputCP):
	df_list = []
	df_list_t = []
	df_list2 = []
	df_list3 = []
	pur_ls = []
	#idid = []
	t_idset = set()
	id_BB = {}
	re_list = []
	t_smiset = set()
	tier_list = []

	BB_dic = {}
	Osmi_dic = {}
	df_list = []
	aBB = Extract_BB(asmi)
	out_csv_path = "./Data/ADC_Output/"
	
	wsmi_df = pd.DataFrame().from_dict(InputCP,orient="index").T
	wsmi_df["ZID"] = "* "+ file_name
	if not re_list :
		re_list = working_a0(asmi,file_name)
	elif re_list == -1:
		re_list = set()
	else:
		re_list = set(re_list)
	mw_percent = 1.0
	break_flag = 0
	n_bbs = 0
	while break_flag == 0:
		query_entities = BB_Query_parameters(asmi,mw_percent)
		break_flag,n_bbs = Fetch_PBB_BY_MW_RingNum_temp(BB_dic,Osmi_dic,query_entities,mw_percent,n_bbs,DB_Path)
		mw_percent += 1.0
	lenid = len(BB_dic)//9

	div_keys = list(divide_list(BB_dic.values(),lenid))
	div_ids = list(divide_list(BB_dic.keys(),lenid))
	a = 0
	for ids,keys in zip(div_ids,div_keys):
		a += 1
		smiles = []
		Align_DF = AlignM3D(file_name,aBB,ids,keys)
		for i in Align_DF["Query"]:
			smiles.append(Osmi_dic[i]["Osmi"])
		Align_DF["SMILES"] = smiles
		Align_DF.to_csv(file_name + ".%d"%a + ".csv",index=False)
	for i in glob.glob(file_name + "*.csv"):
		df_list.append(pd.read_csv(i))
		os.remove(i)
	fin_df = pd.concat(df_list).sort_values(by="PCScore",ascending=False)
	fin_df.to_csv(out_csv_path + file_name + ".BBAlign.total.csv",index=False) # Extract Backbone Align score file

	smi_list = fin_df["SMILES"].drop_duplicates()
	for smi in smi_list: 
		idid = []
		pcscore = np.float64(fin_df[fin_df["SMILES"] == smi]["PCScore"][:1])
		dd = Fetch_BB_smis(smi,t_smiset,DB_Path)
		if dd is None:
			id_df = None
			pass
		else:
			for t_smi in dd:
				Fetch_BB_IDs(t_smi,idid,id_BB,DB_Path)
			id_df = Final_Annot(idid,id_BB,pcscore,DB_Path)

		df_list_t.append(id_df)
		if id_df is None:
			pass
		else:
			t_idset = t_idset | set(id_df["ZID"].tolist())
		print(len(t_idset))
		if len(t_idset) >= int(N_ZIDs_cutoff):
			break
		else:
			pass
	fin_df = pd.concat(df_list_t).drop_duplicates()
	for i in t_idset:
		drop_df = fin_df[fin_df["ZID"] == i].sort_values(by="PCScore",ascending=False)[:1]
		df_list3.append(drop_df)
	fin_df = pd.concat(df_list3).sort_values(by="PCScore",ascending=False)

	za_df = AlignM3D(file_name,asmi,fin_df["ZID"],fin_df["SMILES"]).rename(columns={"Query":"ZID","PCScore":"Z_PCScore"}).drop(["Template"],axis=1)
	fin_df = reduce(lambda x,y : pd.merge(x,y,on="ZID"),[fin_df,za_df]).rename(columns={"PCScore":"BB_PCScore"}).sort_values(by="Z_PCScore",ascending=False)
	for i in fin_df["ZID"]:
		tier_list.append("T 1.6 Backbone Alignment Search")
	fin_df["Tier"] = tier_list
	fin_df = pd.concat([wsmi_df,fin_df])
	fin_df = fin_df[['ZID',"Z_PCScore",'BB_PCScore','MW','LogP','TPSA','RotatableB','HBD','HBA','Ring','Total_Charge','HeavyAtoms','CarBonAtoms','HeteroAtoms','Lipinski_Violation','VeBer_Violation','Egan_Violation','Toxicity','SMILES',"Purchasability","Tier"]]
	fin_df.reset_index(drop=True,inplace=True)
	fin_df1 = fin_df[fin_df["BB_PCScore"] >= 0.70][fin_df["Z_PCScore"] >= 0.70] # BB PCScore Cutoff & Z PCScore Cutoff
	pd.concat([wsmi_df,fin_df1]).to_csv(out_csv_path + file_name + ".fin_out.csv",index=False)
	pd.concat([wsmi_df,fin_df]).to_csv(out_csv_path + file_name + ".all_out.csv",index=False)

	return fin_df1
def working_ZADC(asmi,file_name,afile,InputCP):
	fin_df = pd.DataFrame()
	ZIDs = []
	aBB = Extract_BB(asmi)
	wsmi_df = pd.DataFrame().from_dict(InputCP,orient="index").T
	wsmi_df["ZID"] = "* " + file_name
	re_list = working_a0(asmi,file_name)
	if re_list == -1:
		re_list = set()
	else:
		re_list = set(re_list)
	re_list1 = working_a1(asmi,file_name,afile)
	if re_list1 == -1: # Check inner-Scaffold
		fin_df = working_a2(asmi,file_name,InputCP) # if mol don't have inner-Scaffold,it do Backbone Align
		#return fin_df
	else:
		re_list = re_list|set(re_list1)
	if not fin_df.empty:
		zdf = pd.concat([wsmi_df,fin_df])
	else:
		ZIDs,zdf = BB_Align_Class_Search_ForZADC(Extract_BB(asmi),file_name,re_list,ZIDs,N_ZIDs_cutoff)
		if len(ZIDs) >= N_ZIDs_cutoff:
			print(zdf)
			pass
		else:
			fin_df = working_a2(asmi,file_name,InputCP)
			tfin_df = pd.concat([wsmi_df,zdf,fin_df]).drop_duplicates().reset_index(drop=True)
			zdf = tfin_df
			print(zdf)
	
	zdf = zdf[['ZID',"Z_PCScore",'BB_PCScore','MW','LogP','TPSA','RotatableB','HBD','HBA','Ring','Total_Charge','HeavyAtoms','CarBonAtoms','HeteroAtoms','Lipinski_Violation','VeBer_Violation','Egan_Violation','Toxicity','SMILES',"Purchasability","Tier"]]
	zdf = pd.concat([zdf[:1],zdf[1:].sort_values(by="Z_PCScore",ascending=False)])
	print(zdf)
	zdf.to_csv(out_csv_path + file_name + ".ZADC.csv",index=False)
	return zdf

def working_ZDC(asmi,file_name,afile,InputCP):
	fin_df = pd.DataFrame()
	ZIDs = []
	aBB = Extract_BB(asmi)
	wsmi_df = pd.DataFrame().from_dict(InputCP,orient="index").T
	wsmi_df["ZID"] = "* " + file_name
	global CF_dic,finl_dic
	re_list = working_a0(asmi,file_name)
	if re_list == -1:
		re_list = set()
	else:
		re_list = set(re_list)

	re_list1 = working_a1(asmi,file_name,afile)
	if re_list1 == -1:
		fin_df = working_a2(asmi,file_name,InputCP)
	else:
		re_list = re_list|set(re_list1)
	print(fin_df)
	if not fin_df.empty:
		zdf = pd.concat([wsmi_df,fin_df])
	else:
		ZIDs,zdf = BB_Align_Class_Search_ForZADC(Extract_BB(asmi),file_name,re_list,ZIDs,N_ZIDs_cutoff)
		if len(ZIDs) >= N_ZIDs_cutoff:
			print(zdf)
			pass
		else:
			fin_df = working_a2(asmi,file_name,InputCP)
			tfin_df = pd.concat([wsmi_df,zdf,fin_df]).drop_duplicates().reset_index(drop=True)
			zdf = tfin_df
			print(zdf)

	zdf = zdf[["ZID","Z_PCScore","BB_PCScore","MW","LogP","TPSA","RotatableB","HBD","HBA","Ring","Total_Charge","HeavyAtoms","CarBonAtoms","HeteroAtoms","Lipinski_Violation","VeBer_Violation","Egan_Violation","Toxicity","SMILES","Purchasability","Tier"]]
	zdf = pd.concat([zdf[:1],zdf[1:].sort_values(by="Z_PCScore",ascending=False)])
	print(zdf)
	zdf.to_csv(out_csv_path + file_name + ".ZDC.csv",index=False)
	pre_bb,pre_id = Scaffolds_for_Clustering(zdf,file_name)
	Active_Clustering(file_name,pre_bb,pre_id)
	
	return zdf
def working_ZDC2(asmi,file_name,afile,InputCP):
	fin_df = pd.DataFrame()
	ZIDs = []
	aBB = Extract_BB(asmi)

	wsmi_df = pd.DataFrame().from_dict(InputCP,orient="index").T
	wsmi_df["ZID"] = "* "+file_name

	re_list = working_a0(asmi,file_name)

	if re_list == -1:
		re_list = set()
	else:
		re_list = set(re_list)

	re_list1 = working_a1(asmi,file_name,afile)
	if re_list1 == -1:
		return pd.DataFrame()
	else:
		re_list = re_list|set(re_list1)

	ZIDs,zdf = BB_Align_Class_Search_ForZADC(aBB,file_name,re_list,ZIDs,N_ZIDs_cutoff)

	zdf = zdf[['ZID',"Z_PCScore",'BB_PCScore','MW','LogP','TPSA','RotatableB','HBD','HBA','Ring','Total_Charge','HeavyAtoms','CarBonAtoms','HeteroAtoms','Lipinski_Violation','VeBer_Violation','Egan_Violation','Toxicity','SMILES',"Purchasability","Tier"]]
	zdf = pd.concat([zdf[:1],zdf[1:].sort_values(by="Z_PCScore",ascending=False)])
	print(zdf)
	zdf.to_csv(out_csv_path + file_name + ".ZDC.csv",index=False)

	return zdf

def Extract_ADC():

    global T1_pcscore_cutoff,N_ZIDs_cutoff
    T1_pcscore_cutoff=0.98
    ring_cutoff=1
    N_ZIDs_cutoff = 1500

    iPath='./Data/Input/'
    global oPath
    oPath='./Data/ADC_Output/Files/'
    rei_img_Path='./Data/Re_Input/IMG/'

    #DB_Path='./Data/DB_Tables/'
    #DB_Path='../3D_Scan/Data/DB_Table/'
    global DB_Path
    #DB_Path='/ssd/swshin/1D_Scan.v2/Data/DB_Table/'
    DB_Path = "/lwork01/yklee/DB_Table/"
    BB_ID_dic = load_BBID()
    #global BB_ID_dic

    # MW of template
    # 5*'C'(6)+1*'0'(8)
    TR_MW = 38
    
    # Margin of MW
    # 5*'C'(6)+1*'0'(8)
    #M_MW = 0.1
    #M_MW = 0.2

    # Remove the previous ADC output
    shutil.rmtree(oPath)
    os.mkdir(oPath)

    shutil.rmtree(rei_img_Path)
    os.mkdir(rei_img_Path)
    #return

    # Extract Backbone from input files
    print 'Starting ZINC-ADC...'
    split_BB_list(iPath,BB_ID_dic)
    if_list=glob.glob(iPath+'*.smi')
    #if_list = glob.glob(iPath + "STB_BB_1608376.smi")
    if_list.sort()

    # Check if there exists input files
    if len(if_list)==0:
        print 'There are no input(no smi files).....'
        return

    BB_list=[]
    Main_list =[]
    re =[]
    sum_df = []
    alp = 0
    #OK_flag=0
    # For one input
    out_file = "./Data/ADC_Output/Out_Summary.csv"
    """
    if not os.path.exists(out_file):
        pass
    else:
        os.remove(out_file)"""
    out_file2 = "./Data/ADC_Output/Error_SMILES.csv"

    
    # ############################################################################
    for afile in if_list:
        alp += 1
        M_MW = 0.075
        print(bcolors.WARNING + "Processing : %s - %d/%d \n"%(afile,alp,len(if_list)) + bcolors.ENDC)
        atmp_list =[]
        asmi = Read_SMILES_FILE(afile)
        Input_CP = Extract_CP(asmi)
        Input_Num_Ring = Input_CP["Ring"]
        OK_flag=0

        aBB = Extract_BB(asmi)
        if aBB in BB_list:
            continue
        BB_list.append(aBB)
        fn = os.path.basename(afile)
        file_name = os.path.splitext(fn)[0]

        Draw_BB_smi(rei_img_Path,fn,aBB)
        ZIDs = T1_Class_Search(aBB,file_name,m_type=1)
        tmp_list = []
        id_smi = {}
        re_list = set()
        if ZIDs == -1:
            idx =1
        else:
            for zids in ZIDs:
                tmp_df = Fetch_Purch_Annot(zids,DB_Path)
                if tmp_df is None:
                    pass
                elif ''.join(tmp_df["Purchasability"].tolist()) == "Unknown":
                    pass
                else:
                    tmp_list.append(tmp_df)
        if not tmp_list:
            idx = 1
            print(bcolors.WARNING + "Go To T 1.5" + bcolors.ENDC)
        else:
            tmp_df = pd.concat(tmp_list)
            idx = 0
            for i in tmp_df["ZID"]:
                Fetch_SMILES_by_ID(i,id_smi,DB_Path)
            for i in list(set(id_smi.values())):
                re_list.add(Extract_BB(i))
            re_list = list(re_list)
            print(bcolors.WARNING + "T 1.5 Pass" + bcolors.ENDC)
                
        #
        if idx == 1 :
            print(bcolors.WARNING + "\nTier 1.5 Start\n" + bcolors.ENDC)
            Total_re_list =set()
            re_list = []
            while(OK_flag!=1):
                print(bcolors.WARNING + 'Loop: %d'%idx + bcolors.ENDC)
                #re,tmp_mw = T15_Class_Search(afile,TR_MW,M_MW,Total_re_list)
                re,tmp_mw = T15_Class_Search_Type2(afile,TR_MW,M_MW,Total_re_list)
                if re == -1 or tmp_mw == -1 : # No Scaffold and inter-ring
                    break
                M_MW = M_MW+0.05
                re = set(list(re))
                Total_re_list = Total_re_list|re
                ltrl = len(Total_re_list)
                print len(Total_re_list),type(Total_re_list) 
                idx+=1
                if ltrl == 0 and idx==3: # Number of backbones is 0 and Number of loop is 4
                    break
                if len(list(Total_re_list))>=1 or tmp_mw >= np.float64(700.0) or idx == 3:
                    OK_flag=1
                    if len(list(Total_re_list)) >= 1:
                        re_list = Total_re_list #random.sample(list(Total_re_list),20)
                    else:
                        re_list = Total_re_list
            os.system("clear")
            print(bcolors.WARNING + "T 1.5 Process End : %s - %d/%d \n"%(afile,alp,len(if_list)) + bcolors.ENDC)
                
        if not re_list:
            ttmp_df = pd.DataFrame()
            pass
        else:
            print(bcolors.WARNING + '\nnTotal Num. of candidate ligand for '+afile+': '+str(len(re_list)) + bcolors.ENDC)
            #ZIDs,zdf = BB_Align_Class_Search(aBB,file_name,re_list,ZIDs,N_ZIDs_cutoff) # from MD_BA_class2
			#	print(len(t_idset))
            ZIDs,zdf = BB_Purch_Search(aBB,file_name,re_list)
            ttmp_df = zdf
            #if zdf.empty:
            #    ttmp_df = pd.DataFrame()
            #else:
            #    ttmp_df = Write_Out_Summary(file_name,aBB,zdf) # from MD_BA_class2
        #sum_df.append(ttmp_df)
        if ttmp_df.empty:
            with open(out_file2,"a") as W:
                W.write(asmi + '\n')
        if not os.path.exists(out_file):
            ttmp_df.to_csv(out_file,index=False,mode="w")
        else:
            ttmp_df.to_csv(out_file,index=False,mode="a",header=False)
        #sum_df.append(ttmp_df)

    # Write summary
    #out_file='./Data/ADC_Output/Out_Summary.csv'
    #sum_df1 = pd.concat(sum_df)
    #print(sum_df1)
    #sum_df1.to_csv(out_file,index=False)

    return



def T1_Class_Search(aBB,file_name,m_type):
    re = Search_ASMILES(aBB,m_type)
    if re == -1:
        return re
    zids=list(re)
    #list_SDFs = FetchM_SDF(zids,DB_Path)
    #Save_Extracted_SDF(list_SDFs,file_name,oPath)

    return zids

def BB_Align_Class_Search(aBB,file_name,re_list,ZIDs,N_ZIDs_cutoff): # from MD_BA_class2
    didi = {} # Manager().dict()
    smis = []
    zids = set(ZIDs)
    in_ID = file_name
    asmi = aBB
    for i,j in zip(re_list,range(len(re_list))):
        didi[str(j)] = i

    Align_DF = AlignM3D(in_ID,asmi,didi.keys(),didi.values())
    for i in Align_DF["Query"]:
        smis.append(didi[i])
    Align_DF["SMILES"] = smis
    Align_DF.to_csv("./Data/ADC_Output/" + in_ID + '.total.csv',index=False)

    mol2_list = glob.glob("*.mol2")
    Ncpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(Ncpu-2)
    pool.map(multi_file_remove_func,mol2_list)
    pool.close()
    pool.join()

    zzids,ddf = Make_BB_Align_result2(file_name,zids,N_ZIDs_cutoff,asmi,DB_Path)
    if zzids is None:
        return
    else:
        zzids = list(zzids)

        return zzids,ddf
def BB_Align_Class_Search_ForZADC(aBB,file_name,re_list,ZIDs,N_ZIDs_cutoff):
	didi = {}
	smis = []
	zids = set(ZIDs)
	in_ID = file_name
	asmi = aBB
	for i,j in zip(re_list,range(len(re_list))):
		didi[str(j)] = i
	Align_DF = AlignM3D(in_ID,asmi,didi.keys(),didi.values())
	for i in Align_DF["Query"]:
		smis.append(didi[i])
	Align_DF["SMILES"] = smis
	Align_DF2 = Align_DF #[Align_DF["PCScore"] >= 0.8] # Backbone PCScore Cutoff
	
	Align_DF2.to_csv("./Data/ADC_Output/" + in_ID + ".total.csv",index=False)
	zzids,ddf = Make_BB_Align_result2(file_name,zids,N_ZIDs_cutoff,asmi,DB_Path)
	if zzids is None:
		return
	else:
		zzids = list(zzids)
		return zzids,ddf

def Active_Clustering(fn,bb_list,id_list):
	df = pd.DataFrame()
	temp_list = []
	temp_list2 = []
	temp_list3 = []
	temp_list4 = []
	df["Backbone"] = bb_list
	df["ZID"] = id_list

	global temp_list

	temp_list = Manager().list()
	CF_dic = {}
	finl_dic = {}

	for i in bb_list:
		try:
			rm = Chem.MolFromSmiles(i)
		except:
			pass
		if not rm is None:
			try:
				fw = MurckoScaffold.MakeScaffoldGeneric(rm)
			except:
				pass
			if not fw is None:
				CF_dic[i] = Chem.MolToSmiles(fw)
			else:
				pass
		else:
			pass
	remain_smiles = set(CF_dic.keys())
	Scaffold_Clustering(CF_dic,finl_dic,remain_smiles)

	for i in finl_dic:
		finl_dic[i] = ','.join(finl_dic[i])
	pdf = pd.DataFrame.from_dict(finl_dic,orient="index").reset_index().rename(columns={"index":"Backbone",0:"Cluster_Members"})
	ddf = pd.merge(df,pdf,on="Backbone")

	for idx,line in ddf.iterrows():
		tnum = 0
		for i in line["Cluster_Members"].split(","):
			aa = df[df["Backbone"]==i]["ZID"].values[0].split(",")
			print(type(aa))
			print(aa)
			tsum = len(aa)
			tnum += tsum
			for j in aa: #df[df["Backbone"]==i]["ZID"].values[0]): #.split(","):
				temp_list3.append(j)
		temp_list.append(tnum)
		rm = Chem.MolFromSmiles(line["Backbone"])
		sssr = Chem.GetSSSR(rm)
		temp_list2.append(sssr)
		temp_list4.append(",".join(temp_list3))
	ddf.drop("ZID",axis=1,inplace=True)
	ddf["Total of ZIDs"] = temp_list
	ddf["Num of Rings"] = temp_list2
	ddf["ZID"] = temp_list4
	ddf = ddf[["Backbone","Num of Rings","Total of ZIDs","Cluster_Members","ZID"]]
	ddf.sort_values(by="Total of ZIDs",ascending=False,inplace=True)
	ddf.to_csv(out_csv_path + "%s.Cluster.csv"%fn,index=False)
	
def BB_Purch_Search(aBB,file_name,re_list):
	######################################
	# Bio Active Backbone Analysis Tools #
	######################################
	id_BB = {}
	for bb in re_list:
		idid = []
		Fetch_BB_IDs(bb,idid,id_BB,DB_Path)
		for ids in idid:
			t_list = []
			t_df = pd.DataFrame()
			t_id = ids
			purch_df = Fetch_Purch_Annot(ids,DB_Path)
			if purch_df is None:
				pass
			elif ''.join(purch_df["Purchasability"].tolist()) == "Unknown":
				pass
			else:
				t_list.append(t_id)
				cp_df = Fetch_CP_Annot(t_id,DB_Path)
				t_df = pd.merge(cp_df,purch_df)
				t_df["File_Name"] = [file_name]
				t_df["Backbone"] = [aBB]
				t_df = t_df[["File_Name","ZID","SMILES","Backbone","MW","LogP","TPSA","RotatableB","HBD","HBA","Ring","Total_Charge","HeavyAtoms","CarBonAtoms","HeteroAtoms","Lipinski_Violation","VeBer_Violation","Egan_Violation","Toxicity","Purchasability"]]
				return t_list,t_df
	return t_list,t_df

	
	
	




def T2_Class_Search(df_subscaffold):
    return


def T3_Class_Search(df_subscaffold):
    return


def T4_Class_Search(df_subscaffold):
    return


def T5_Class_Search(df_subscaffold):
   
    for index,row in df_subscaffold.iterrows():
        #print row[0],row[1]
        # Ring Num check
        if int(row[0]) > 1:
            asmi=row[1]
            print asmi
            Search_ASMILES(asmi)
            break

    return

def T15_Class_Search(afile,TR_MW,M_MW,MW_Index_list):


    Re_ID_List=[]

    # Max_Num= 500
    Max_Num= 500

    # For one input
    print '\n'+'For file:',afile
    print 'Start processing T1.5....'
    atmp_list=[]
    asmi = Read_SMILES_FILE(afile)
    asmi = Make_Canonical_SMI(asmi)
    #print 'rdkit smi:',asmi
    if asmi == -1:
        return -1,-1
    pmol = readstring('smi',asmi)
    psmi = pmol.write(format='smi')
    #print 'pybel smi:',psmi
    pBB = Extract_BB(psmi)
    #print 'PBB:',pBB
    re_list = Extract_Inner_Scaffold(pBB)

    #if len(re_list) == 0:
    if re_list == -1 or not re_list:
        print 'There is no \'Scaffold\' in the mol'
        return -1,-1
    t_mw = 0
    manager = Manager()
    Re_Zid_list = manager.list()
    for alist in re_list:
        Scaffold = alist[0]
        BB = alist[1]
        Num_Brench = alist[2]
        try:
            pmol = readstring('smi',BB)
        except:
            print '  -> pybel reading input error'
            return -1,-1

        list_ring=pmol.sssr
        num_ring = len(list_ring)

        cp = Extract_CP(Scaffold)

        #entry=(cp['MW'],cp['LogP'],cp['HBA'],cp['HBD'],cp['TPSA'],cp['Ring'])
        #print entry
        #Same_CPs = Fetch_BB_CP(entry,DB_Path)

        print '\nScaffold: '+Scaffold+',Num. Brench: '+str(Num_Brench)
        Total_Ring = num_ring + Num_Brench
        print '  --> Querying samiles to CP DB'

        S_MW = cp['MW']
        IMW = (TR_MW*Num_Brench) + S_MW
        print(IMW)
        if M_MW == 0.1:
            MMW = IMW * M_MW
            FIMW_Max = IMW + MMW
            FIMW_Min = IMW - MMW

            #print FIMW_Min, FIMW_Max

            entry=(FIMW_Min, FIMW_Max, Total_Ring)
            t_mw = FIMW_Max
            print entry
            retri_list = Fetch_BB_BY_MW_RingNum(entry,DB_Path)
            #print retri_list
            #sys.exit(1)

        else:
            tM_MW =M_MW-0.05
            MMW = IMW * tM_MW
            PFIMW_Max = IMW + MMW
            PFIMW_Min = IMW - MMW

            MMW = IMW * M_MW
            CFIMW_Max = IMW + MMW
            CFIMW_Min = IMW - MMW

            #print PFIMW_Min, PFIMW_Max, CFIMW_Min, CFIMW_Max

            entry=(CFIMW_Min, PFIMW_Min, PFIMW_Max, CFIMW_Max,Total_Ring)
            t_mw = CFIMW_Max
            print entry
            retri_list = Fetch_BB_BY_MW_RingNum(entry,DB_Path)

        #return
        if type(retri_list) == int:
            pass
        else:
            print '  --> The number of retrieval: '+str(len(retri_list))

        if retri_list == -1:
            print '  --> There is no result of query and pass'
            return -1,-1



        # Multi Version
        #manager = Manager()
        #Re_Zid_list = manager.list()
        Num_Of_CPU=multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
        func=partial(MatchM_Substructre,Re_Zid_list,Scaffold)
        #pool.map(func,tmp_retri_list)
        pool.map(func,retri_list)
        pool.close()
        pool.join()


    print '\n\nTotal Num. of candidate ligand for '+afile+': '+str(len(Re_Zid_list))
    return Re_Zid_list,t_mw

#def T15_Class_Search_Type2(afile,TR_MW,M_MW,MW_Index_list):
def T15_Class_Search(afile,TR_MW,M_MW,MW_Index_list):


    Re_ID_List=[]

    # Max_Num= 500
    Max_Num= 500

    # For one input
    print '\n'+'For file:',afile
    print 'Start processing T1.5....'
    atmp_list=[]
    asmi = Read_SMILES_FILE(afile)
    asmi = Make_Canonical_SMI(asmi)
    if asmi == -1 :
        print("\nIt is Impossible to change Canonical SMILES\n")
        return -1,-1
    #print 'rdkit smi:',asmi
    try:
        pmol = readstring('smi',asmi)
        psmi = pmol.write(format='smi')
    except:
        print("Error SMILES change : %s"%asmi)
        return -1,-1
    #print 'pybel smi:',psmi
    pBB = Extract_BB(psmi)
    #print 'PBB:',pBB
    re_list = Extract_Inner_Scaffold(pBB)

    #if len(re_list) == 0:
    if re_list == -1 or not re_list:
        print 'There is no \'Scaffold\' in the mol'
        return -1,-1
    t_mw = 0
    manager = Manager()
    Re_Zid_list = manager.list()
    for alist in re_list:
        Scaffold = alist[0]
        BB = alist[1]
        Num_Brench = alist[2]
        try:
            pmol = readstring('smi',BB)
        except:
            print '  -> pybel reading input error'
            return -1,-1

        list_ring=pmol.sssr
        num_ring = len(list_ring)

        cp = Extract_CP(Scaffold)

        #entry=(cp['MW'],cp['LogP'],cp['HBA'],cp['HBD'],cp['TPSA'],cp['Ring'])
        #print entry
        #Same_CPs = Fetch_BB_CP(entry,DB_Path)

        print '\nScaffold: '+Scaffold+',Num. Brench: '+str(Num_Brench)
        Total_Ring = num_ring + Num_Brench
        print '  --> Querying samiles to CP DB'

        S_MW = cp['MW']
        IMW = (TR_MW*Num_Brench) + S_MW
        print(IMW)
        if M_MW == 0.075:
            MMW = IMW * M_MW
            FIMW_Max = IMW + MMW
            FIMW_Min = IMW - MMW

            #print FIMW_Min, FIMW_Max

            entry=(FIMW_Min, FIMW_Max, Total_Ring)
            t_mw = FIMW_Max
            print entry
            retri_list = Fetch_BB_BY_MW_RingNum(entry,DB_Path)
            #print retri_list
            #sys.exit(1)

        else:
            tM_MW =M_MW-0.05
            MMW = IMW * tM_MW
            PFIMW_Max = IMW + MMW
            PFIMW_Min = IMW - MMW

            MMW = IMW * M_MW
            CFIMW_Max = IMW + MMW
            CFIMW_Min = IMW - MMW

            #print PFIMW_Min, PFIMW_Max, CFIMW_Min, CFIMW_Max

            entry=(CFIMW_Min, PFIMW_Min, PFIMW_Max, CFIMW_Max,Total_Ring)
            t_mw = CFIMW_Max
            print entry
            retri_list = Fetch_BB_BY_MW_RingNum(entry,DB_Path)

        #return
        if type(retri_list) == int:
            pass
        else:
            print '  --> The number of retrieval: '+str(len(retri_list))

        if retri_list == -1:
            print '  --> There is no result of query and pass'
            return -1,-1


        #print retri_list
        #sys.exit(1)

        # Multi Version
        #manager = Manager()
        #Re_Zid_list = manager.list()
        Num_Of_CPU=multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
        func=partial(MatchM_Substructre,Re_Zid_list,Scaffold)
        #pool.map(func,tmp_retri_list)
        pool.map(func,retri_list)
        #pool.map(func,retri_list[0:1000])
        pool.close()
        pool.join()

        '''
        # Serial Version
        for relist in retri_list:
            #print relist,Scaffold
            #print relist[0],relist[1]

            try:
                tasmi = Make_Canonical_SMI(relist[1])
            except:
                continue
            tm_mol = Chem.MolFromSmiles(tasmi)

            try:
                patt = Chem.MolFromSmiles(Scaffold)
            except:
                continue

            match_list = tm_mol.GetSubstructMatches(patt)
            if len(match_list)>0:
                print 'Matched ID:',relist[0],relist[1],'                                                     \r',
                sys.stdout.flush()
                Re_ID_List.append(relist[0])
        '''

    print '\n\nTotal Num. of candidate ligand for '+afile+': '+str(len(Re_Zid_list))
    Re_Zid_list = Check_Re_Zid_list(IMW,Re_Zid_list)
    return Re_Zid_list,t_mw

def T15_query_parameters(ainput,TR_MW,mw_percent):
	Scaffold = ainput[0]
	BB = ainput[1]
	Num_Brench = ainput[2]
	InputCP = Extract_CP(Scaffold)
	# Count Total Ring
	try :
		pmol = readstring("smi",BB)
	except:
		print("Pybel Reading Error")
	list_ring = pmol.sssr
	num_ring = len(list_ring)
	
	Total_Ring = num_ring + Num_Brench
	# Calculate MW
	S_MW = InputCP["MW"]
	IMW = (TR_MW*Num_Brench) + S_MW
	per_MW = IMW*np.float64(mw_percent)/np.float64(100)
	nper_MW = IMW*np.float64(mw_percent-5.0)/np.float64(100)
	if mw_percent == 7.5:
		up_MW = IMW + per_MW
		low_MW = IMW - per_MW

		return IMW,(low_MW,up_MW,Total_Ring)
	else:
		uu_MW = IMW + per_MW
		ul_MW = IMW + nper_MW #per_MW/2.0
		
		lu_MW = IMW - nper_MW #per_MW/2.0
		ll_MW = IMW - per_MW
		return IMW,(ll_MW,lu_MW,ul_MW,uu_MW,Total_Ring)

	
def T15_Class_Search_yklee(afile,TR_MW,M_MW):
	Re_ID_List = []
	atmp_list = []

	Max_Num = 500

	print("\nFor file : %s"%afile)
	print("Start Processing T1.5....")
	
	asmi = Read_SMILES_FILE(afile)
	asmi = Make_Canonical_SMI(asmi)
	if asmi == -1: # pass point 1
		print("\nIt is Impossible to change Canonical SMILES\n")
		return -1
	try: # pass point 2
		pmol = readstring("smi",asmi)
		psmi = pmol.write("smi")
	except:
		print("\nIt is Impossible to change SMILES\n")
		return -1
	pBB = Extract_BB(psmi)
	
	#InSCF_list = Extract_Inner_Scaffold(pBB)
	#InSCF_list = Extract_Inner_Scaffold_3(pBB)
	#print(InSCF_list)
	if Check_Ring_Total_Ring_Num(pBB) <=2:
		InSCF_list = Less_Than_Two_Ring(pBB)
	else:
		InSCF_list = Extract_Inner_Scaffold_3(pBB)
	if InSCF_list == -1 or not InSCF_list: # pass point 3
		print("There is no \"Scaffold\" in the Mol")
		return -1
	t_mw = 0
	manager = Manager()
	Re_Zid_list = manager.list()
	n_bbs = 0
	for alist in InSCF_list:
		Scaffold = alist[0]
		break_flag = 0
		mw_percent = 7.5
		while break_flag == 0:

			IMW,entities = T15_query_parameters(alist,TR_MW,mw_percent)
			
			#break_flag,n_bbs,retri_list = Fetch_BB_BY_MW_RingNum_temp(entities,mw_percent,n_bbs,DB_Path)
			retri_list = Fetch_PBB_BY_MW_RingNum(entities,DB_Path)
			if type(retri_list) == int:
				pass
			else:
				print '  --> The number of retrieval: '+str(len(retri_list))

			if retri_list == -1:
				print '  --> There is no result of query and pass'
				return -1

			Num_Of_CPU=multiprocessing.cpu_count()
			pool = multiprocessing.Pool(processes=(Num_Of_CPU-1))
			func=partial(MatchM_Substructre,Re_Zid_list,Scaffold)
			pool.map(func,retri_list)
			pool.close()
			pool.join()
			print(len(Re_Zid_list))
			if len(Re_Zid_list) >= 100 or np.float64(entities[-2]) >= 700.0 or mw_percent >= 50.0:
				break_flag = 1
			else:
				break_flag = 0
				mw_percent += 5.0

	print '\n\nTotal Num. of candidate ligand for '+afile+': '+str(len(Re_Zid_list))
	Re_Zid_list = Check_Re_Zid_list(IMW,Re_Zid_list)
	return Re_Zid_list
def T15_Class_Search_yklee_temp(afile,TR_MW,M_MW):
	Re_ID_List = []
	atmp_list = []
	temp_dic = {}
	Max_Num = 500

	print("For File : %s"%afile)
	print("Start Processing T1.5......")

	asmi = Read_SMILES_FILE(afile)
	asmi = Make_Canonical_SMI(asmi)

	if asmi == -1:
		print("It is Impossible to change Canonical SMILES")
		return -1
	try:
		pmol = readstring("smi",asmi)
		psmi = pmol.write("smi")
	except:
		print("It is Impossible to change SMILES")
		return -1

	pBB = Extract_BB(psmi)
	InSCF_list = Extract_Inner_Scaffold(pBB)

	if InSCF_list == -1 or not InSCF_list:
		print("There is no \"Scaffold\" in the Mol")
		return -1 
	t_mw = 0
	manager = Manager()
	Re_Zid_list = manager.list()
	n_bbs = 0
	for alist in InSCF_list:
		Scaffold = alist[0]
		break_flag = 0
		mw_percent = 7.5
		while break_flag ==0:
			IMW,entities = T15_query_parameters(alist,TR_MW,mw_percent)
			retri_list = Fetch_PBB_BY_MW_RingNum(entities,DB_Path)
			if type(retri_list) == int:
				pass
			else:
				print("--> The number of retrieval: " + str(len(retri_list)))
			if retri_list == -1:
				print("--> There is no result of query and pass")
				return -1

			Ncpu = multiprocessing.cpu_count()
			pool =multiprocessing.Pool(Ncpu-2)
			func = partial(MatchM_Substructre,Re_Zid_list,Scaffold)
			pool.map(func,retri_list)
			pool.close()
			pool.join()
			if len(Re_Zid_list) >= 250 or np.float64(entities[-2]) >= 700.0 or mw_percent >= 50.0:
				break_flag = 1
			else:
				break_flag = 0
				mw_percent += 5.0
	print("Total Num. of candidiate ligand for " + afile + ": " + str(len(Re_Zid_list)))
	Re_Zid_list = Check_Re_Zid_list(IMW,Re_Zid_list)

	return Re_Zid_list
####################################
# Extract Scaffold For Clustering  #
####################################	
def Scaffolds_for_Clustering(in_df,f_name):
	#global Re_Dic
	idx = 0
	lines = []
	for ii, row in in_df.iterrows():
		if idx == 0:
			pass
		else:
			t_ZID = row.ZID.encode("ascii","ignore")
			t_SMILES = row.SMILES.encode("ascii","ignore")
			tmp_str = t_ZID + "," + t_SMILES
			lines.append(tmp_str)
		idx +=1
	Re_Dic = R_Decomp(lines)

	for key in Re_Dic:
		lst_zid = Re_Dic[key]
		str_zid = ",".join(lst_zid)
		Re_Dic[key] = str_zid
	df = pd.DataFrame(list(Re_Dic.items()),columns=["Backbone","ZID"])
	out_f = out_csv_path + f_name + ".ZID.csv"
	df.to_csv(out_f,index=False)
	return df["Backbone"].tolist(),df["ZID"].tolist()

def R_Decomp(lines):
	ln_lines = len(lines)

	manager = Manager()
	Re_Dic = manager.dict()
	Main_list = manager.list()

	Ncpu = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(Ncpu-2)
	func = partial(R_Decomp_Core,lines,ln_lines,Main_list,Re_Dic)
	pool.map(func,lines)
	pool.close()
	pool.join()

	return dict(Re_Dic)
def R_Decomp_Core(lines,ln_lines,Main_list,Re_Dic,aline):
	idx = lines.index(aline)

	token = aline.split(",")
	zid = token[0]
	z_smiles = token[1]
	zBB = Extract_BB(z_smiles)

	proc = os.getpid()

	re_list = Extract_Inner_Scaffold(zBB)

	if type(re_list) == int:
		return
	else:
		pass

	for are_list in re_list:
		BB= are_list[1]
		if BB not in Re_Dic:
			tmp_list = [zid]
			Re_Dic[BB] = tmp_list

		else:
			tmp_list = Re_Dic[BB]
			tmp_list.append(zid)
			tmp_list = list(set(tmp_list))
			Re_Dic[BB] = list(tmp_list)
#########################################
def diff(first, second):
    return [item for item in first if item not in second]



def MatchM_Substructre(Re_Zid_list,Scaffold,relist):

    #print Scaffold,relist 

    aBB = relist[0]

    if aBB in Re_Zid_list:
        return -1

    try:
        tasmi = Make_Canonical_SMI(aBB)
    except:
        return -1
    tm_mol = Chem.MolFromSmiles(tasmi)

    try: 
        patt = Chem.MolFromSmiles(Scaffold)
    except:
        return -1
    try:
        match_list = tm_mol.GetSubstructMatches(patt)
    except:
        return -1
    if len(match_list)>0:
        print 'Matched ID:',aBB,'                                                             \r',
        sys.stdout.flush()
        Re_Zid_list.append(aBB)


def Search_ASMILES(asmi,m_type):

    T_ZIDs=set()

    if m_type==1 or m_type==3: 
        #########################################
        # 1. Extract ZIDs by Backbone using BB DB
        print ' -> Starting search: BB exact matching'

        T_ZIDs_BB=set()

        #re=Fetch_BB(asmi,DB_Path)
        re = Fetch_PBB(asmi,DB_Path)
        #print re
        if re is None:
            print("SMILES %s , There is no result for extract BB matching")
            return -1
        elif len(re)>0:
            candi =re[-1]
            candi = set(candi)
            T_ZIDs_BB=T_ZIDs_BB.union(candi)
        else:
            print 'SMILES: '+asmi+', There is no result for extact BB matching.'
            return -1
               
        cp = Extract_CP(asmi)
        #print cp

        entry=(cp['MW'],cp['LogP'],cp['HBA'],cp['HBD'],cp['TPSA'],cp['Ring'])
        #Same_CPs = Fetch_BB_CP(entry,DB_Path)
        Same_CPs = Fetch_PBB_CP(entry,DB_Path)
        print  ' --> Same_CPs num: ',len(Same_CPs)

        # Extract smiles for same CP
        list_smi=[x[0] for x in Same_CPs]

        #print list_smi
        #print len(list_smi)
        #print len(set(list_smi))

        # Make id list for AlignM3D
        list_smi_id=[range(0,len(list_smi),1)]
        list_smi_id = list_smi_id[0]
        list_smi_id = [str(x) for x in list_smi_id]
        #print asmi
        #print list_smi
        #print list_smi_id
        #sys.exit(1)
        #return
        #print len(list_smi_id)
        df = AlignM3D('template',asmi,list_smi_id,list_smi)
        df = df.drop(df[df.PCScore<T1_pcscore_cutoff].index)
        #print df
        #print Same_CPs
        
        for index,row in df.iterrows():
            #print row[1]
            #sys.exit(1)
            #print list_smi[int(row[1])],Same_CPs[int(row[1])][0]
            candi = Same_CPs[int(row[1])][-1]  
            candi = set(candi) 
            T_ZIDs_BB=T_ZIDs_BB.union(candi)
        #print len(T_ZIDs_BB),T_ZIDs_BB


    if m_type==2 or m_type==3: 

        ############################################
        # 2. Extract ZIDs by Scaffold using Scaffold DB
        print '\n -> Starting search: Scaffold exact matching'
        T_ZIDs_SF=set()
        list_zids = Fetch_Scaffold(asmi,DB_Path)

        if len(list_zids)==0:
            print ' --> There is no result for scaffold matching'
        else:
            #list_zids = list_zids[0:50]
            # list of ZID:['ZINC000845648520', 'ZINC000330252169', 'ZINC000418997081']
            #print len(list_zids)

            re = FetchM_SDF_to_SMI(list_zids,DB_Path)
            list_smi = [x[0] for x in re]
            list_smi_id = [x[1] for x in re]
            #print len(list_smi),list_smi
            #print len(list_smi_id),list_smi_id

            df = AlignM3D('template',asmi,list_smi_id,list_smi)
            df = df.drop(df[df.PCScore<T1_pcscore_cutoff].index)

            # Test_Test
            #df = df.drop(df[df.PCScore<0.5].index)
            tmp_list=[]
            for index,row in df.iterrows():
                tmp_list.append(row[1])
            #print tmp_list
            T_ZIDs_SF = T_ZIDs_SF.union(set(tmp_list))
            #print len(T_ZIDs_SF),T_ZIDs_SF

    if m_type==1:
        return T_ZIDs_BB 
    if m_type==2:
        return T_ZIDs_SF 
    if m_type==3:
        T_ZIDs = T_ZIDs.union(T_ZIDs_SF)
        T_ZIDs = T_ZIDs.union(T_ZIDs_BB)
        return T_ZIDs 

    return
def Check_Re_Zid_list(IMW,Re_Zid_list):

    Re_Zid_list=list(Re_Zid_list[0:20])
    #Re_Zid_list=list(Re_Zid_list)

    re_tmp_list =[]
    #print Re_Zid_list
    #sys.exit(1)

    for aele in Re_Zid_list:
        #print aele
        #re = Check_IS_Purchasable(aele,DB_Path)
        #print re
        #if re ==-1:
            #continue
        tmp_list=[]
        pmol = readstring('smi',aele)
        desc = pmol.calcdesc(descnames=['MW'])
        pMW = desc['MW']
        diff = abs(IMW-pMW)
        tmp_list.append(diff)
        tmp_list.append(aele)
        re_tmp_list.append(tmp_list)
        #print tmp_list

    re_tmp_list = sorted(re_tmp_list, key=itemgetter(0))
    #print re_tmp_list
    re = [x[1] for x in re_tmp_list]
    #print re
    #sys.exit(1)

    return re




def Save_Extracted_SDF(list_SDFs,t_fn,oPath):
   
    #print ' -> Writing the SDF files'
    t_dir = os.path.join(oPath,t_fn)
    #print t_dir 

    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    else:
        FNULL = open(os.devnull, 'w')
        arg='rm '+t_dir+'*'
        process = subprocess.Popen(arg, shell=True, stdout=FNULL,stderr=subprocess.STDOUT)

    for asdf in list_SDFs:
        token=asdf[0].split('\n')
        #print asdf
        ofn = token[0]+'.sdf'
        t1_dir = os.path.join(t_dir,ofn)
        #print t_dir
        print ' -> Writing file: ',t1_dir+'                     \r',
        sys.stdout.flush()
        fp_for_out = open(t1_dir,'w')
        fp_for_out.write(asdf[0])
        fp_for_out.close()

    return


################################################################################
def main():

    parser=argparse.ArgumentParser()
    #parser.add_argument('-t',required=True, choices=['l','f'], default='n',  help='Input type: list(csv) or files(smi).')
    #parser.add_argument('-BB',required=True, choices=['y','n'], default='n',  help='Input type: Backbone or not backboen.')
    #parser.add_argument('-i',required=True, help='Input list or path.')
    #parser.add_argument('-top_p',required=True, default=0, help='Select the top X percent of scaffod frequency.')
    #parser.add_argument('-mins',required=True, default=1, help='The minimum number of scaffold.')
    #parser.add_argument('-img',required=True, choices=['y','n'], default='n',  help='Making the png files')
    args=parser.parse_args()

    #print i_type,i_content,i_BB

    time1=time()
    os.system('clear')
    #Extract_Mol()
    #df = Extract_Chemical_Feature(i_path,m_type)
    #Query_DB(df)
    #Query_ScaffoldDB(in_file)
    #Make_PDB_from_ADC()
    #Make_BBR(in_file)

    #df,f_list  = Processing_input(i_type,i_content)
    #print df.head()
    #return
    #if i_BB =='n':
        #df = Make_BB(df,f_list)
    #else:
        #pass
        
    #print df.head()
    #Extract_CP_SMILES2(i_type,i_content,df)

    #Extract_ADC()
    yklee_work(3)


    time2=time()
    print '\n\nTotal excution time: '+str('{:.2f}'.format(time2-time1))+' sec.'
    print_date_time()
    print '\n\n'



if __name__=="__main__":

    # 2021.04.05
    # python 3D_Scan_Make_Scaffold_Backbone_list.Memo.v2.py -input ./Data/Input/ -top_p 0 -mins 1 -img n
    # python Extract_CP_SMILES.v2.py -t f -BB n -i ./MOA/DMC_ligand/
    main()
