ZINC_ADC
==========
# Project Goal : Findig similarly SMILES by input SMILES

## Usage : python ZINC_ADC.yklee.v2.py

## Help:
* working type 0:   
> Tier 1 Exact Match   

* working type 1:   
> Tier 1.5 Match   

* working type 2:   
> Tier 1.6 Match   

* working type 3:   
> Extract ZINC ADC   
<hr/>

### Data:    
+ **Input Path** : ./Data/Input   
  + SMILES string
+ **Output Path** : ./Data/ADC_Output   
  + Out_Summary.csv
  + Error_SMILES.csv
  
  **Exact Match & Inner-Scaffold Search**   
  + [file_name].[N.ZIDs.Cutoff].csv
  
  **Backbone Alignment Result**   
  + [file_name].fin_out.csv
  + [file_name].BBAlign.total.csv
  + [file_name].all_out.csv   
  
  **Extract ZINC ADC**
  + [file_name].[N.ZIDs.Cutoff].csv
  + [file_name].BBAlign.total.csv
  + [file_name].fin_out.csv
  + [file_name].all_out.csv
  + [file_name].ZADC.csv

### Default DB Path :
* /ssd/swshin/1D_Scan.v2/Data/DB_Table/   

##Annotation Images   
<img src="./imgs/ZINC_ADC.VS_concept.jpg" width="75%" height="50%"></img><br/>
<img src="./imgs/ZINC_ADC.workflow.jpg" width="75%" height="50%"></img><br/>
<img src="./imgs/ZINC_ADC.ExampleOfResults.jpg" width="75%" height="50%"></img><br/>
