#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 15:40:27 2021

#GlowScript 3.0 VPython

# Hard-sphere gas.

# Bruce Sherwood
# Claudine Allen
"""

from vpython import *
import numpy as np
import math
import matplotlib.pyplot as plt
import random as rd
from scipy.stats import maxwell
from scipy.optimize import curve_fit


# Déclaration de variables influençant le temps d'exécution de la simulation
Natoms = 200  # change this to have more or fewer atoms
dt = 1E-7  # pas d'incrémentation temporel

mass = 9.109e-31 # Pour le choix de la distribution de maxwell

Ratom = 0.01
k = 1.4E-23 # Boltzmann constant
T = 300 # around room temperature

#### CANEVAS DE FOND ####
L = 1 # container is a cube L on a side
gray = color.gray(0.7) # color of edges of container and spheres below
animation = canvas( width=750, height=500) # , align='left')
animation.range = L

#### ARÊTES DE BOÎTE 2D ####
d = L/2+Ratom
r = 0.005
cadre = curve(color=gray, radius=r)
cadre.append([vector(-d,-d,0), vector(d,-d,0), vector(d,d,0), vector(-d,d,0), vector(-d,-d,0)])

#### POSITION ET QUANTITÉ DE MOUVEMENT INITIALE DES SPHÈRES ####
Atoms = [] # Objet qui contiendra les sphères pour l'animation
p = [] # quantité de mouvement des sphères
apos = [] # position des sphères


# Génération des électrons
for i in range(Natoms):
    x = L*random()-L/2
    y = L*random()-L/2
    z = 0
    if i == 0:  # garde une sphère plus grosse et colorée parmis toutes les grises
        Atoms.append(simple_sphere(pos=vector(x,y,z), radius=0.02, color=color.magenta))
    else: Atoms.append(simple_sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
    apos.append(vec(x,y,z)) # liste de la position initiale de toutes les sphères

    pavg = mass * maxwell.rvs(scale=sqrt(k * T / mass))

    phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
    px = pavg*cos(phi)  # quantité de mvt initiale selon l'équipartition
    py = pavg*sin(phi)
    pz = 0
    p.append(vector(px,py,pz)) # liste de la quantité de mvt initiale de toutes les sphères

# Génération des noyaux
npos = []
Noyaux = []
steps = [0.45,0.27,0.09,-0.09,-0.27,-0.45]
for i, x in enumerate(steps):
    for j, y in enumerate(steps):
        Noyaux.append(simple_sphere(pos=vector(x,y,0), radius=0.03, color=color.green))
        npos.append(vec(x,y,0))

steps = [0.36,0.18,0,-0.18,-0.36]
for i, x in enumerate(steps):
    for j, y in enumerate(steps):
        Noyaux.append(simple_sphere(pos=vector(x,y,0), radius=0.03, color=color.green))
        npos.append(vec(x,y,0))

#### FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions():
    hitlist = []   # initialisation
    r2 = 2*Ratom   # distance critique où les 2 sphères entre en contact à la limite de leur rayon
    r2 *= r2
    for i in range(Natoms):
        ai = apos[i]
        for j in range(len(Noyaux)) :
            aj = npos[j]
            dr = ai - aj
            if mag2(dr) < r2:
                hitlist.append([i,j])
    return hitlist


liste_p_moyenne = []
liste_p_electron = []

atom_id = 1
reset = False
tau_mesure = []
tickParticule = 1

#### BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt ####
for i in range(2000):
    rate(300)

    # Calculer les magnitudes des vecteurs
    p_norm = [mag(vecteur) for vecteur in p]
    pavg = np.mean(p_norm)
    liste_p_moyenne.append(pavg)
    liste_p_electron.append(mag(p[0])) # qdm pour un seul électron

    #### DÉPLACE TOUTES LES SPHÈRES D'UN PAS SPATIAL deltax
    vitesse = []   # vitesse instantanée de chaque sphère
    deltax = []  # pas de position de chaque sphère correspondant à l'incrément de temps dt
    for i in range(Natoms):
        vitesse.append(p[i]/mass)   # par définition de la quantité de nouvement pour chaque sphère
        deltax.append(vitesse[i] * dt)   # différence avant pour calculer l'incrément de position
        Atoms[i].pos = apos[i] = apos[i] + deltax[i]  # nouvelle position de l'atome après l'incrément de temps dt

    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS AVEC LES PAROIS DE LA BOÎTE ####
    for i in range(Natoms):
        loc = apos[i]
        if abs(loc.x) > L/2:
            if loc.x < 0: p[i].x =  abs(p[i].x)  # renverse composante x à la paroi de gauche
            else: p[i].x =  -abs(p[i].x)   # renverse composante x à la paroi de droite
        if abs(loc.y) > L/2:
            if loc.y < 0: p[i].y = abs(p[i].y)  # renverse composante y à la paroi du bas
            else: p[i].y =  -abs(p[i].y)  # renverse composante y à la paroi du haut

    #### LET'S FIND THESE COLLISIONS!!! ####
    hitlist = checkCollisions()
    T = (pavg**2)/(3 * mass * k)
    
    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS ENTRE SPHÈRES ####
    for ij in hitlist:
        # définition de nouvelles variables pour chaque paire de sphères en collision
        i = ij[0]
        j = ij[1]

        # Définir la collision selon le modèle de Drudes
        phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
        pNorm = mass * maxwell.rvs(scale=sqrt(k * T / mass))
        px = pNorm*cos(phi)
        py = pNorm*sin(phi)

        # Nouvelle quantité de mouvement
        p[i] = vector(px,py,0)
        # Nouvelle position pour évité les doublons de collision
        apos[i] = vector(2*Ratom*cos(phi), 2*Ratom*sin(phi), 0) + npos[j]

        if (i or j) == atom_id:
            tau_mesure.append(tickParticule)
            reset = True
    if reset:
        tickParticule = 0
        reset = False
    tickParticule += 1
    
tau_moyen = np.mean(tau_mesure)


def exponential_fit(t, p0, tau):
    return p0 * np.exp(-t / tau)

# Curve fiting exponentielle
t_data = np.arange(0,len(liste_p_moyenne), 1) 
popt, pcov = curve_fit(exponential_fit, t_data, liste_p_moyenne, p0=(liste_p_moyenne[0], 1.0))
p0_fit, tau_fit = popt
p_fit = exponential_fit(t_data, p0_fit, tau_fit)

# Graphique p(t) moyen
plt.subplot(2,1,1)
plt.plot(t_data, liste_p_moyenne, label='p(t)', color='black')
plt.plot(t_data, p_fit, label=f'Curve fit (τ ={(tau_fit * dt):.5f})', color='red', linestyle='--')
plt.xlabel('Temps (10e-7) [-]')
plt.ylabel('p moyenne [-]')
plt.title('p moyenne des électrons en fonction du temps')
plt.legend()
plt.grid(True)

# Graphique p(t) pour un seul électron
plt.subplot(2,1,2)
plt.plot(t_data, liste_p_electron, label='p(t)', color='black')
plt.plot(t_data, (liste_p_electron[0] * np.exp(- t_data / tau_moyen)), label=f"Temps libre moyen de l'électron (τ ={(tau_moyen * dt):.5f})", color='blue', linestyle='--')
plt.xlabel('Temps (10e-7) [-]')
plt.ylabel("p d'un électron [-]")
plt.title("p d'un électron en fonction du temps")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()