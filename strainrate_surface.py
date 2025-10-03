#!/Users/victorsacek/anaconda3/bin/ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob


step_initial = int(sys.argv[1])
step_final = int(sys.argv[2])

if (len(sys.argv)>3): d_step = int(sys.argv[3])
else: d_step = 25

with open("param.txt","r") as f:
	line = f.readline()
	line = line.split()
	Nx = int(line[2])
	line = f.readline()
	line = line.split()
	Nz = int(line[2])
	line = f.readline()
	line = line.split()
	Lx = float(line[2])
	line = f.readline()
	line = line.split()
	Lz = float(line[2])
	
print(Nx,Nz,Lx,Lz) #Lz/(Nz-1)

#xx,zz = np.mgrid[0:Lx:(Nx)*1j,-Lz:0:(Nz)*1j]

xi = np.linspace(0,Lx/1000,Nx)
zi = np.linspace(-Lz/1000,0,Nz)
xx,zz = np.meshgrid(xi,zi)


ts = glob.glob("time_*.txt")
total_curves = len(ts) #total_steps/print_step
n_curves = total_curves/2

val = 100

air = 40.0

for cont in range(step_initial,step_final,d_step):#
	print(cont)

	"""
	#A = np.loadtxt("strain_"+str(cont)+".txt",unpack=True,comments="P",skiprows=2)
	A = pd.read_csv("strain_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = TT[:,:]
	TTT[TTT==0] = 1.0E-18 #evitar log0
	TTT = np.log10(TTT) 
	plt.close()
	plt.figure(figsize=(10*2,2.5*2))
	plt.contourf(xx,zz,np.transpose(TTT),100)
	#plt.contour(xx,zz,np.transpose(rho),levels=[1000,2900,3360],colors="k")
	
	plt.xlim(250,1750)
	
	"""
	
	#PLOT DOS PONTOS
	#plt.plot(x/1000,z/1000,"c.",color=cor,markersize=0.3)
	"""
	cond = (cc>difere2) & (cc<difere)
	plt.plot(x[cond]/1000,z[cond]/1000,"c.",color=cor,markersize=0.3)
	cond = (cc<difere2)
	plt.plot(x[cond]/1000,z[cond]/1000,"c.",color=cor3,markersize=0.3)
	plt.plot(x[cc>difere]/1000,z[cc>difere]/1000,"c.",color=cor2,markersize=0.3)
	"""
	
	#plt.savefig("strain_{:05}.png".format(cont))
	
	

	
	#A = np.loadtxt("rho_"+str(cont)+".txt",unpack=True,comments="P",skiprows=2)
	A = pd.read_csv("density_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None)
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = TT[:,:]
	TTT = TTT[:,::-1]
	rho = np.copy(TTT)

	"""
	plt.close()
	plt.figure(figsize=(10*2,2.5*2))
	plt.contourf(xx,zz,np.transpose(TTT))
	#PLOT DOS PONTOS
	#plt.plot(x/1000,z/1000,"c.",color=cor,markersize=0.3)
	
	
	plt.savefig("H_{:05}.png".format(cont*10))
	"""

	A = np.loadtxt("time_"+str(cont)+".txt",dtype='str')  
	AA = A[:,2:]
	AAA = AA.astype("float") 
	tempo = np.copy(AAA)
	
	print("Time = %.1lf Myr\n\n"%(tempo[0]/1.0E6))

	A = pd.read_csv("strain_rate_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = TT[:,:]
	TTT = TTT[:,::-1]
	TTT[rho<200]=0
	TTT = np.log10(TTT)
	TTT = np.transpose(TTT)
	plt.close()
	plt.figure(figsize=(10*2,2.5*2))

	b1 = [0.1,0.05,0.876,0.65]
	bv1 = plt.axes(b1)

	plt.imshow(TTT,extent=[0,Lx/1000,-Lz/1000+air,0+air],vmin=-19,vmax=-14) #100 deixar a escala de cor mais suave, com 100 cores 
	plt.colorbar()

	plt.text(100,10,"%.1lf Myr"%(tempo[0]/1.0E6))
	
	#PLOT DOS PONTOS
	#plt.plot(x/1000,z/1000,"c.",color=cor,markersize=0.3)
	"""
	cond = (cc>difere2) & (cc<difere)
	plt.plot(x[cond]/1000,z[cond]/1000,"c.",color=cor,markersize=0.3)
	cond = (cc<difere2)
	plt.plot(x[cond]/1000,z[cond]/1000,"c.",color=cor3,markersize=0.3)
	plt.plot(x[cc>difere]/1000,z[cc>difere]/1000,"c.",color=cor2,markersize=0.3)
	"""

	"""
	x=[]
	z=[]
	cc=[]
	
	for rank in range(4):
		
		A = pd.read_csv("step_"+str(cont)+"-rank_new"+str(rank)+".txt",delimiter = " ",header=None)
		A = A.to_numpy()
		x1 = A[:,0]
		z1 = A[:,1]
		c1 = A[:,3]

		cor =  (0,0,0)
		cor2 = (0,0,0)
		cor3 = (0,0,0)
		#print(cor)
		
		cc = np.append(cc,c1)
		x = np.append(x,x1)
		z = np.append(z,z1)


	condp = (cc>0) & (cc<7)
	x = x[condp]
	z = z[condp]

	#plt.close()
	#plt.figure()
	"""

#Surface descomentar aqui até a linha 189

	zs = np.loadtxt("sp_surface_global_"+str(cont)+".txt",unpack=True,skiprows=2,comments="P")

    
	n = np.size(zs)
    
	zmean = np.mean(zs[xi<600.])
	zs=zs-zmean


	b1 = [0.1,0.80,0.7,0.15]
	bv1 = plt.axes(b1)

	plt.plot(xi,zs)
	plt.plot(xi,xi*0-1500.0)



	zmean = 0.0#np.mean(z)
	
	plt.xlim(0,Lx/1000)
	plt.ylim(-6000,+8000)


	plt.savefig("rs2_{:05}.png".format(cont*1))

	#plt.plot(x/1000,z/1000,"c.",color=cor,markersize=0.3)

	#plt.xlim(500,1000)

	#plt.ylim(-200,-20)
	
	#plt.savefig("Gr2_rate_{:05}.png".format(cont*1))
	




