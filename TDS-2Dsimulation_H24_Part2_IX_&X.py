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

def maxwell_distribution(v,m,t):
    # Génère une distribution de maxwell selon le script: https://ballen95.pythonanywhere.com/Week2-Notebook/
    data = []
    vx = []
    r = 8.31446261815324 # J / (mol * K)
    distribution = (4.*np.pi * ((m / (2 * np.pi * r * t))**1.5) * (v**2) * np.exp(- (m * v**2)/(2 * r * t)))
    distribution = (np.array(distribution)/np.amax(distribution))*(2**8)
    distribution = distribution.astype("uint16")
    for i, x in enumerate(v):
        data += [x]*distribution[i]
    return data

def random_maxwell(maxwell):
    # Petit hic, je dois adapté l'ordre de grandeur pour que ça fonctionne, la distribution est défini par le gaz ET la température
    # Retourne une valeur de vitesse aléatoire selon la distribution de maxwell préalablement choisie.
    return maxwell[rd.randint(0,len(maxwell)-1)]

# Déclaration de variables influençant le temps d'exécution de la simulation
Natoms = 100  # change this to have more or fewer atoms
dt = 1E-5  # pas d'incrémentation temporel

mass = 9.109e-31 # Pour le choix de la distribution de maxwell

Ratom = 0.01
k = 1.4E-23 # Boltzmann constant
T = 300 # around room temperature

# Composantes du champ uniforme (respecté pas plus haut que 1e-4 sinon parfois il y a des sorties du cadre)
dpx = 1e-4
dpy = 1e-4

# ref: https://ballen95.pythonanywhere.com/Week2-Notebook/
#         maxwell_distribution(array_vitesse_m/s, masse_gaz(ici He), Température)                    
maxwell = maxwell_distribution(np.linspace(0,4000,1000), 4/1000, T)

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

    pavg = random_maxwell(maxwell)*mass

    phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
    px = pavg*cos(phi)  # quantité de mvt initiale selon l'équipartition
    py = pavg*sin(phi)
    pz = 0
    p.append(vector(px,py,pz)) # liste de la quantité de mvt initiale de toutes les sphères

# Génération des noyaux
npos = []
Noyaux = []
steps = [0.3,0.1,-0.1,-0.3]
for i, x in enumerate(steps):
    for j, y in enumerate(steps):
        Noyaux.append(simple_sphere(pos=vector(x,y,0), radius=0.03, color=color.green))
        npos.append(vec(x,y,0))

steps = [0.4,0.2,0,-0.2,-0.4]
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

#### BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt ####
for i in range(1000):
    rate(300)

    #### DÉPLACE TOUTES LES SPHÈRES D'UN PAS SPATIAL deltax
    vitesse = []   # vitesse instantanée de chaque sphère
    deltax = []  # pas de position de chaque sphère correspondant à l'incrément de temps dt
    for i in range(Natoms):
        champE_uniforme = vector(dpx,dpy,0)
        vitesse.append(p[i]/mass)   # par définition de la quantité de nouvement pour chaque sphère
        deltax.append(vitesse[i] * dt + champE_uniforme)   # différence avant pour calculer l'incrément de position
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

    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS ENTRE SPHÈRES ####
    for ij in hitlist:
        # définition de nouvelles variables pour chaque paire de sphères en collision
        i = ij[0]
        j = ij[1]

        # Définir la collision selon le modèle de Drudes
        phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
        pNorm = random_maxwell(maxwell)*mass
        px = pNorm*cos(phi)
        py = pNorm*sin(phi)

        # Nouvelle quantité de mouvement
        p[i] = vector(px,py,0)
        # Nouvelle position pour évité les doublons de collision
        apos[i] = vector(2*Ratom*cos(phi), 2*Ratom*sin(phi), 0) + npos[j]

