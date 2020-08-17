

import random
import os
import sys
import vegas
import math
import torch
import datetime
import matrix2py
import numpy as np
from inspect import getsourcefile
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 





from nisrep.normalizing_flows.manager import *
from nisrep.PhaseSpace.flat_phase_space_generator import *


torch.set_default_dtype(torch.double)

class Colour:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

global counter
counter=0

E_cm=1000
pdf=True
pT_cut=0
delR_cut=0.4  
rap_maxcut=2.4
var_n=40000
nitn=20
neval=2000
n_bins=5
NN=[11]*10
lr=0.19e-3
wd=3.8e-4
batch_size=40000
preburn_time=40
dev = torch.device("cuda:"+str(0)) if torch.cuda.is_available() else torch.device("cpu")
ngrid=40

names=["d","u","s","c","b","t","g"]
pdgs=[1,2,3,4,5,6,21]

matrix2py.initialisemodel("param_card.dat")

file = open("nexternal.inc", "r") 
file.readline()

n_external=file.readline()
n_external = int((n_external.split("=")[1]).split(")")[0])
file.readline()
n_incoming=file.readline()
n_incoming = int((n_incoming.split("=")[1]).split(")")[0])
file.close

file = open("pmass.inc", "r") 
z=file.readlines()

z=[x.split("=")[1].split("\n")[0] for x in z]
z=[x[::-1].split("(")[0][::-1].split(")")[0] for x in z]
file.close

file = open("param.log", "r")
p=file.readlines()
file.close

masses=[0]*len(z)
external_masses=[0]*2
Gf=  float([i for i in p if " mdl_gf " in i][0].split()[7])
aEW=  1/float([i for i in p if " aewm1 " in i][0].split()[7])
MZ= float( [i for i in p if " mdl_mz " in i][0].split()[7])

for ider,x in enumerate(z):
    if x=="ZERO":
        masses[ider]=0.0
    elif x=="MDL_MW":
        
        masses[ider]=np.sqrt(MZ**2/2. + np.sqrt(MZ**4/4. - (aEW*np.pi*MZ**2)/(Gf*np.sqrt(2))))
       
    else:
        res = [i for i in p if " "+x.lower()+" " in i][0]
        masses[ider]=float(res.split()[7])
external_masses[0]=masses[:n_incoming]
external_masses[1]=masses[n_incoming:]

path=os.path.abspath(getsourcefile(lambda:0))
path=path.split("P1_")[1]
particles=path.split("_")[0]

pdg=[0]*len(external_masses[0])
offset1=0
offset2=0
for ide,x in enumerate(names):
    ider=ide-offset1-offset2
    marker=particles.find(x)
    
    if marker==0 and (x!='t' or(len(particles)<=2 or (particles[2]!='-' and particles[2]!='+' ))):
        
        pdg[offset1]=pdgs[ider]
        particles=particles[1:]
        
        if len(particles)>0 and particles[0]=="x":
            pdg[offset1]*=-1
            particles=particles[1:]
            
        names.insert(ide,x)
        offset1+=1
       
        
       
    elif marker!=-1 and (x!='t' or(len(particles)<=2+marker or (particles[marker+2]!='p' and particles[marker+2]!='m' ))) :
        particles=particles[:marker]+particles[marker+1 :]
        pdg[1]=pdgs[ider]
        
        if len(particles)>marker and particles[marker]=="x":
            particles=particles[:marker]+particles[marker+1 :]
            pdg[1]*=-1
        names.insert(ide,x)
        offset2+=1
    if offset1+offset2==2:
        break
        

print("Ingoing particles: "+str(len(external_masses[0])))
print("Ingoing pdg codes: "+str(pdg[0])+" "+str(pdg[1]))
print("PDFs active: "+str(pdf))
print("Outgoing particles: "+str(len(external_masses[1])))


if((len(external_masses[0]))!=2 and pdf):
    print("No PDF support for other cases than 2 body collision")
   

p=None

if pdf:
    import lhapdf
    lhapdf.pathsPrepend("/home/niklas/Desktop/Thesis_code/MG5_aMC_v3_0_2_py3/HEPTools/lhapdf6/share/LHAPDF") 
    p = lhapdf.mkPDF("NNPDF23_nlo_as_0119", 0)
    

this_process_E_cm = max( E_cm, sum(external_masses[1])*2. )
my_ps_generator=FlatInvertiblePhasespace(external_masses[0], external_masses[1],pdf=p,pdf_active=pdf)

s=this_process_E_cm**2
element=0

def fv(x):
    try:
        dev = torch.device("cuda:"+str(0)) if torch.cuda.is_available() else torch.device("cpu")
        momenta, jac = my_ps_generator.generateKinematics_batch(this_process_E_cm, torch.tensor(x).unsqueeze(0), pT_mincut=pT_cut, delR_mincut=delR_cut, rap_maxcut=rap_maxcut, pdgs=pdg)

        
        momenta=momenta.squeeze(0).t().cpu().tolist()
       
        jac=jac.cpu()

        element=matrix2py.smatrix(momenta)*jac[0]
  
    except Exception as e:
            global counter
            counter=counter+1
            return 0
   
    return (element)

def f(x):
        
        momenta, jac = my_ps_generator.generateKinematics_batch(this_process_E_cm, x,pT_mincut=pT_cut, delR_mincut=delR_cut, rap_maxcut=rap_maxcut, pdgs=pdg)
        momenta=momenta.cpu()
        jac=jac.cpu()
        #print(momenta)
        q=0
        element=[0]*momenta.shape[0]
       
        element=[matrix2py.smatrix(momenta[ind,:,:].t().tolist())*jac[ind]
                 for ind, q in enumerate(element)]
        end_time=datetime.datetime.utcnow()
       
        
        return torch.tensor(element,device=x.device)
    
if not pdf:
    n_flow = my_ps_generator.nDimPhaseSpace() # number of dimensions
else:
     n_flow = my_ps_generator.nDimPhaseSpace()+2 # number of dimensions
w = torch.empty(var_n, n_flow)
torch.nn.init.uniform_(w)
v_var_int=torch.var(f(w))

print("--------")
print("Initial variance: " +str(v_var_int))
print("E_cm: "+str(this_process_E_cm))
print("CUTS: min pT: "+str(pT_cut)+" min deltaR: "+str(delR_cut)+" max rapidity: "+str(rap_maxcut))
print("--------")
print("-----")

NF =  PWQuadManager(n_flow=n_flow)
NF.create_model(n_cells=2, n_bins=n_bins, NN=NN,dev=0)


optim = torch.optim.Adamax(NF._model.parameters(),lr=lr, weight_decay=wd) 
print("Batchsize: "+ str(batch_size) + " n_bins: " +str(n_bins) + " NN_length: "+str(len(NN)) + " NN_width: "+str(NN[0]))
print("LR: "+str(lr)+" weight decay: "+str(wd)+ " preburn_time: "+str(preburn_time))
start_time=datetime.datetime.utcnow()

sig,sig_err=NF._train_variance_forward_seq(f,optim,False,"./logs/tmp/",batch_size,
                                           1000,0,pretty_progressbar=False,save_best=True,run=None,dev=dev,
                                               integrate=True,preburn_time=100)
w = torch.empty(var_n, NF.n_flow)
torch.nn.init.uniform_(w)
Y=NF.format_input(w, dev=dev)
X=NF.best_model(Y)
v_var=torch.var(f(X[:,:-1])*X[:,-1])
w_max=torch.max(f(X[:,:-1])*X[:,-1]).cpu().tolist()
w_mean=torch.mean(f(X[:,:-1])*X[:,-1]).cpu().tolist()
print("--------")

print("-----------")
print("NIS")
print(str(sig) + " +/- " +str(sig_err)+" GeV^-2")
sig=sig/(2.5681894616*10**(-9)) #GeV**2 -> pb
sig_err=sig_err/(2.5681894616*10**(-9))
print(str(sig) + " +/- " +str(sig_err)+" pb")
end_time=datetime.datetime.utcnow()
print("Final Variance: "+str(v_var))
print("Variance reduction: "+str(v_var/v_var_int))
print("Unweighting efficiency: "+str(w_mean/w_max))
print("Duration:")
print((end_time-start_time).total_seconds())
print("-----------")
print('Initial loss')
print(NF.int_loss)
print('Best loss')
print(NF.best_loss)
print('Best loss relative')
print(NF.best_loss_rel)
print('Evaluations')
print(NF.best_func_count)
print("---------------")
print("---------------")

print("FINAL STEP INTEGRATION")
w = torch.empty(var_n, NF.n_flow)
torch.nn.init.uniform_(w)
Y=NF.format_input(w, dev=dev)
X=NF.best_model(Y)
v_var=torch.var(f(X[:,:-1])*X[:,-1])
mean=torch.mean(f(X[:,:-1])*X[:,-1])
err=v_var/math.sqrt(var_n)
print(mean/(2.5681894616*10**(-9)))
print(err/(2.5681894616*10**(-9)))

