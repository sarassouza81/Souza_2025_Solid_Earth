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


ts = glob.glob("Tempo_*.txt")
total_curves = len(ts) #total_steps/print_step
n_curves = total_curves/2

val = 100

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
	
	

	"""
	A = np.loadtxt("H_"+str(cont)+".txt",unpack=True,comments="P",skiprows=2)
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = TT[:,:]
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


	A = pd.read_csv("density_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = TT[:,:]
	TTT = np.transpose(TTT)
	rho = np.copy(TTT)

	A = pd.read_csv("temperature_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = np.transpose(TT)
	temper = np.copy(TTT)


	A = pd.read_csv("strain_"+str(cont)+".txt",delimiter = " ",comment="P",skiprows=2,header=None) 
	A = A.to_numpy()
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT,(Nx,Nz),order='F')
	TTT = np.transpose(TT)
	TTT[rho<200]=0
	#TTT[rho>3365]=0
	#TTT[TTT<1.0E-1]=0
	TTT = np.log10(TTT)
	stc = np.copy(TTT)
	
	


	plt.close()
	plt.figure(figsize=(10*2,2.5*2))
	#plt.imshow(rho2,extent=[0,Lx/1000,-Lz/1000,0]) #100 deixar a escala de cor mais suave, com 100 cores 
	#plt.plot([0,1600],[-40,-40],"r")

	cr = 255.
	color_sed = (241./cr,184./cr,68./cr)
	color_dec = (137./cr,81./cr,151./cr)
	color_uc = (228./cr,156./cr,124./cr)
	color_lc = (240./cr,209./cr,188./cr)
	color_lit = (155./cr,194./cr,155./cr)
	color_ast = (207./cr,226./cr,205./cr)

	plt.contourf(xx,zz,rho,levels=[200.,2750,2900,3365,3900],
		colors=[color_uc,color_lc,color_lit,color_ast])
	#plt.contourf(xx,zz,rho,levels=[200.,2750,2900,3365,3900],
		#colors=[color_uc,color_lc,color_lit,color_ast])

	#plt.contour(xx,zz,temper,levels=[700,1000,1200],
		#colors=[(0,0,0)])
	#para plotar as isotermas em vermelho
	plt.contour(xx,zz,temper,levels=[550,750,850,950,1200,1300],
		colors=[(1,0,0),(1,0,0),(1,0,0)],linewidths=0.6)

	print("stc",np.min(stc),np.max(stc))

	#stc = np.log10(stc)
	print("stc(log)",np.min(stc),np.max(stc))
	plt.imshow(stc[::-1,:],extent=[0,Lx/1000,-Lz/1000,0],
		zorder=100,alpha=0.2,cmap=plt.get_cmap("Greys"),vmin=-0.5,vmax=0.9)

	plt.text(100,10,"%.1lf Myr"%(tempo[0]/1.0E6))

	#st = np.log10(st)
	#print("st(log)",np.min(st),np.max(st))
	#plt.imshow(st[::-1,:],extent=[0,Lx/1000,-Lz/1000,0],
	#	zorder=101,alpha=0.5,cmap=plt.get_cmap("Oranges"))#,vmin=-1.14,vmax=0.14)

	#ds = 0.4
	#for i in range(10):
	#	plt.contourf(xx,zz,stc,levels=[-14+i*ds,-14+(i+1)*ds],
	#		colors=["k"],alpha=0.2+0.07*i)

	#plt.contourf(xx,zz,stc,levels=[-0.1,0,1,2,10,100],
	#	colors=["k","k","k","k","k"],alpha=0.5)

	#plt.colorbar()
	
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
	idc=[]
	cc=[]
	
	for rank in range(4):
		
		A = pd.read_csv("step_"+str(cont)+"-rank_new"+str(rank)+".txt",delimiter = " ",header=None)
		A = A.to_numpy()
		x1 = A[:,0]
		z1 = A[:,1]
		idc1 = A[:,2]
		c1 = A[:,3]

		cor =  (0,0,0)
		cor2 = (0,0,0)
		cor3 = (0,0,0)
		#print(cor)
		
		cc = np.append(cc,c1)
		idc = np.append(idc,idc1)
		x = np.append(x,x1)
		z = np.append(z,z1)
	"""

	#condp = (cc==0) & (idc%2==0)
	#x = x[condp]
	#z = z[condp]

	#plt.plot(x/1000,z/1000,"o",ms=0.1,color=(0.5,0.5,0.5))
	

	#plt.close()
	#plt.figure()

	b1 = [0.74,0.41,0.2,0.2]
	bv1 = plt.axes(b1)

	A = np.zeros((100,10))

	A[:25,:]=2700
	A[25:50,:]=2800
	A[50:75,:]=3300
	A[75:100,:]=3400

	A = A[::-1,:]

	xA = np.linspace(-0.5,0.9,10)
	yA = np.linspace(0,1.5,100)

	xxA,yyA = np.meshgrid(xA,yA)
	air_threshold = 200
	plt.contourf(xxA,yyA,A,levels=[air_threshold,2750,2900,3365,3900],
			colors=[color_uc,color_lc,color_lit,color_ast])

	plt.imshow(xxA[::-1,:],extent=[-0.5,0.9,0,1.5],
			zorder=100,alpha=0.2,cmap=plt.get_cmap("Greys"),vmin=-0.5,vmax=0.9)

	bv1.set_yticklabels([])

	plt.xlabel(r"log$(\epsilon_{II})$",size=18)

	plt.savefig("litho_temper_{:05}.png".format(cont*1), dpi=300)

	#plt.plot(x/1000,z/1000,"c.",color=cor,markersize=0.3)

	#plt.xlim(500,1000)
	#
	#plt.ylim(-200,-20)
	#
	#plt.savefig("zoom_litho_{:05}.png".format(cont*1))
	




