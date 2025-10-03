#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 19 10:18:56 2024

@author: jobueno
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

# Diretório e cenários
cdir = 'C:/Users/laudi/OneDrive/Desktop'
cenarios = [
    #'MR_v1_30_15_ris0.6_hkoff_1350', 
    #'MR_v1.5_30_15_ris0.6_hkoff_1350','MR_v2_30_15_ris0.6_hkoff_1350',
    #'MR_v2.5_30_15_ris0.6_hkoff_1350', 'MR_v3_30_15_ris0.6_hkoff_1350',
    #'MR_v3.5_30_15_ris0.6_hkoff_1350', 'MR_v4_30_15_ris0.6_hkoff_1350',
    #'MR_v4.5_30_15_ris0.6_hkoff_1350', 
    'MR_v5_30_15_ris0.6_hkoff_1350',
    'MR_va_30_15_ris0.6_hkoff_1350', 
]

Nx = 1601
Lx = 1600000.0
xi = np.linspace(0, Lx / 1e3, Nx)
maxtimes = []

# Paleta personalizada
colors = [
    '#FF4500', '#F0E68C', '#FFFF00', '#FFD700', '#FF8C00', '#CD5C5C', '#B22222',
    '#FA8072', '#FFC0CB', '#FF69B4', '#C71585', '#FF1493', '#DC143C', '#FF0000',   # quentes (14)
    '#1E90FF', '#0000CD', '#191970', '#228B22', '#20B2AA', '#32CD32', '#66CDAA',  # frias (7)
]

colormap_contraste = ListedColormap(colors, name='quente_para_frio')

# Criando os boundaries para 21 cores
boundaries = np.concatenate([
    np.linspace(-7000, 0, 15),     # 14 intervalos frios
    np.linspace(0, 3500, 8)[1:]  # 7 intervalos quentes (exclui zero duplicado)
])

# BoundaryNorm substitui o TwoSlopeNorm
toponorm = BoundaryNorm(boundaries, ncolors=len(colors))
levels_topo = boundaries  # Usamos os próprios boundaries como níveis

# Coletando tempos máximos
for c in cenarios:
    timesteps = np.loadtxt(os.path.join(cdir, c, 'strainrateplot-ts.txt'), float)
    maxtimes.append(np.max(timesteps))

# Subplots
fig, axshf = plt.subplots(len(cenarios), 1, sharex=True, figsize=(4.268, 2*len(cenarios)),
                          gridspec_kw={'height_ratios': maxtimes/np.min(maxtimes)}, dpi=600)

fig2, axstopo = plt.subplots(len(cenarios), 1, sharex=True, figsize=(4.268, 2*len(cenarios)),
                             gridspec_kw={'height_ratios': maxtimes/np.min(maxtimes)}, dpi=600)

fig3, axssr = plt.subplots(len(cenarios), 1, sharex=True, figsize=(4.268, 2*len(cenarios)),
                           gridspec_kw={'height_ratios': maxtimes/np.min(maxtimes)}, dpi=600)

# Loop principal
for i in range(len(cenarios)):
    c = cenarios[i]
    hf_t = np.loadtxt(os.path.join(cdir, c, 'hfluxplot-ts.txt'), float)
    hf = np.loadtxt(os.path.join(cdir, c, 'hfluxplot-hf.txt'), float)
    topo = np.loadtxt(os.path.join(cdir, c, 'topoplot-tp-mil.txt'), float)
    sr = np.loadtxt(os.path.join(cdir, c, 'strainrateplot-sr.txt'), float)
    sr_t = np.loadtxt(os.path.join(cdir, c, 'strainrateplot-ts.txt'), float)

    xxt, ttt = np.meshgrid(xi, sr_t)

    hf_plot = axshf[i].contourf(xxt, ttt, hf * 1e3, cmap='magma', levels=50, vmin=30, vmax=720)
    
    topo_plot = axstopo[i].contourf(xxt, ttt, topo, cmap=colormap_contraste,
                                    norm=toponorm, levels=levels_topo)
    
    sr_plot = axssr[i].contourf(xxt, ttt, np.log10(sr), cmap='viridis', levels=50,
                                vmin=-19, vmax=-14)

# Colorbars
#smhf = ScalarMappable(cmap=hf_plot.cmap)
#smhf.set_clim(30, 720)

smtp = ScalarMappable(cmap=colormap_contraste, norm=toponorm)
smtp.set_clim(-7000, 3500)

smsr = ScalarMappable(cmap=sr_plot.cmap)
smsr.set_clim(-19, -12)

#cbar_hf = fig.colorbar(smhf,ax=axshf[0], location='top',
#              label='Heat flux (mW/m²)', orientation='horizontal')

cbar_topo = fig2.colorbar(smtp, ax=axstopo[0], location='top',
                          label='z (m)', orientation='horizontal',
                          ticks=[-7000, -4500, -3000, -1500, 0, 1500, 3500])
              
              
cbar_strain = fig3.colorbar(smsr,ax=axssr[0], location='top',
              label='$log10(\dot{ \\varepsilon})$', orientation='horizontal')

#cbar_topo.set_ticks([-7000,-4500, -3000, -1500, 0, 1500, 3500])
cbar_topo.set_ticklabels(['-7000','-4500', '-3000', '-1500', '0', '1500', '3500'])

axshf[len(cenarios)//2].set_ylabel('Time (Myr)')
axshf[-1].set_xlabel('Distance (km)')

axstopo[len(cenarios)//2].set_ylabel('Time (Myr)')
axstopo[-1].set_xlabel('Distance (km)')

axssr[len(cenarios)//2].set_ylabel('Time (Myr)')
axssr[-1].set_xlabel('Distance (km)')

for ax in axshf:
    ax.set_xlim(400, 1600)
    #ax.set_ylim(0, 20) 
    
for ax in axstopo:
    ax.set_xlim(400, 1600)
    #ax.set_ylim(0, 20)

for ax in axssr:
    ax.set_xlim(400, 1600)
    #ax.set_ylim(0, 20)

#fig.suptitle('Heat Flux (cenários)', fontsize=16)
#fig2.suptitle('Topografia (cenários)', fontsize=16)
#fig3.suptitle('Strain Rate - 10km (cenários)', fontsize=16)

plt.show()
fig.savefig(os.path.join(cdir,'hfplot-scaled.png'), bbox_inches='tight')
fig2.savefig(os.path.join(cdir,'topoplot-scaled.png'), bbox_inches='tight')
fig3.savefig(os.path.join(cdir,'srplot-scaled.png'), bbox_inches='tight')
