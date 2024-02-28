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
from scipy.stats import mode, norm
from sigfig import round
from scipy.optimize import curve_fit

reset = False
distParcouru = []
dtCollision = []
posParticule = []
tickParticule = 1
vParticule = []
atom_id = 197

def followParticule(posParticule):
    dTot = 0
    for i, vector in enumerate(posParticule):
        if i+1 != len(posParticule):
            dX = np.abs(vector.x - posParticule[i+1].x)
            dY = np.abs(vector.y - posParticule[i+1].y)
            dTot += np.sqrt(dX**2+dY**2)
    return dTot

# win = 500 # peut aider à définir la taille d'un autre objet visuel comme un histogramme proportionnellement à la taille du canevas.

# Déclaration de variables influençant le temps d'exécution de la simulation
Natoms = 300  # change this to have more or fewer atoms
dt = 1E-5  # pas d'incrémentation temporel

# Déclaration de variables physiques "Typical values"
mass = 4E-3/6E23 # helium mass
Ratom = 0.01 # wildly exaggerated size of an atom
k = 1.4E-23 # Boltzmann constant
T = 300 # around room temperature

#### CANEVAS DE FOND ####
L = 1 # container is a cube L on a side
gray = color.gray(0.7) # color of edges of container and spheres below
animation = canvas( width=750, height=500) # , align='left')
animation.range = L
# animation.title = 'Cinétique des gaz parfaits'
# s = """  Simulation de particules modélisées en sphères dures pour représenter leur trajectoire ballistique avec collisions. Une sphère est colorée et grossie seulement pour l’effet visuel permettant de suivre sa trajectoire plus facilement dans l'animation, sa cinétique est identique à toutes les autres particules.

# """
# animation.caption = s

#### ARÊTES DE BOÎTE 2D ####
d = L/2+Ratom
r = 0.005
cadre = curve(color=gray, radius=r)
cadre.append([vector(-d,-d,0), vector(d,-d,0), vector(d,d,0), vector(-d,d,0), vector(-d,-d,0)])

#### POSITION ET QUANTITÉ DE MOUVEMENT INITIALE DES SPHÈRES ####
Atoms = [] # Objet qui contiendra les sphères pour l'animation
p = [] # quantité de mouvement des sphères
apos = [] # position des sphères
pavg = sqrt(2*mass*1.5*k*T) #Principe de l'équipartition de l'énergie en thermodynamique statistique classique

for i in range(Natoms):
    x = L*random()-L/2 # position aléatoire qui tient compte que l'origine est au centre de la boîte
    y = L*random()-L/2
    z = 0
    if i == 0:  # garde une sphère plus grosse et colorée parmis toutes les grises
        Atoms.append(simple_sphere(pos=vector(x,y,z), radius=0.03, color=color.magenta))
    else: Atoms.append(simple_sphere(pos=vector(x,y,z), radius=Ratom, color=gray))
    apos.append(vec(x,y,z)) # liste de la position initiale de toutes les sphères
#    theta = pi*random() # direction de coordonnées sphériques, superflue en 2D
    phi = 2*pi*random() # direction aléatoire pour la quantité de mouvement
    px = pavg*cos(phi)  # quantité de mvt initiale selon l'équipartition
    py = pavg*sin(phi)
    pz = 0
    p.append(vector(px,py,pz)) # liste de la quantité de mvt initiale de toutes les sphères

#### FONCTION POUR IDENTIFIER LES COLLISIONS, I.E. LORSQUE LA DISTANCE ENTRE LES CENTRES DE 2 SPHÈRES EST À LA LIMITE DE S'INTERPÉNÉTRER ####
def checkCollisions():
    hitlist = []   # initialisation
    r2 = 2*Ratom   # distance critique où les 2 sphères entre en contact à la limite de leur rayon
    r2 *= r2   # produit scalaire pour éviter une comparaison vectorielle ci-dessous
    for i in range(Natoms):
        ai = apos[i]
        for j in range(i) :
            aj = apos[j]
            dr = ai - aj   # la boucle dans une boucle itère pour calculer cette distance vectorielle dr entre chaque paire de sphère
            if mag2(dr) < r2:   # test de collision où mag2(dr) qui retourne la norme élevée au carré de la distance intersphère dr
                hitlist.append([i,j]) # liste numérotant toutes les paires de sphères en collision
    return hitlist

#### BOUCLE PRINCIPALE POUR L'ÉVOLUTION TEMPORELLE DE PAS dt ####
## ATTENTION : la boucle laisse aller l'animation aussi longtemps que souhaité, assurez-vous de savoir comment interrompre vous-même correctement (souvent `ctrl+c`, mais peut varier)
## ALTERNATIVE : vous pouvez bien sûr remplacer la boucle "while" par une boucle "for" avec un nombre d'itérations suffisant pour obtenir une bonne distribution statistique à l'équilibre


# while True:
for temps in range(20000):
    rate(300)  # limite la vitesse de calcul de la simulation pour que l'animation soit visible à l'oeil humain!
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

    #### CONSERVE LA QUANTITÉ DE MOUVEMENT AUX COLLISIONS ENTRE SPHÈRES ####
    for ij in hitlist:

        # définition de nouvelles variables pour chaque paire de sphères en collision
        i = ij[0]  # extraction du numéro des 2 sphères impliquées à cette itération
        j = ij[1]
        ptot = p[i]+p[j]   # quantité de mouvement totale des 2 sphères
        mtot = 2*mass    # masse totale des 2 sphères
        Vcom = ptot/mtot   # vitesse du référentiel barycentrique/center-of-momentum (com) frame
        posi = apos[i]   # position de chacune des 2 sphères
        posj = apos[j]
        vi = p[i]/mass   # vitesse de chacune des 2 sphères
        vj = p[j]/mass
        rrel = posi-posj  # vecteur pour la distance entre les centres des 2 sphères
        vrel = vj-vi   # vecteur pour la différence de vitesse entre les 2 sphères

        # exclusion de cas où il n'y a pas de changements à faire
        if vrel.mag2 == 0: continue  # exactly same velocities si et seulement si le vecteur vrel devient nul, la trajectoire des 2 sphères continue alors côte à côte
        if rrel.mag > Ratom: continue  # one atom went all the way through another, la collision a été "manquée" à l'intérieur du pas deltax

        # calcule la distance et temps d'interpénétration des sphères dures qui ne doit pas se produire dans ce modèle
        dx = dot(rrel, vrel.hat)       # rrel.mag*cos(theta) où theta is the angle between vrel and rrel:
        dy = cross(rrel, vrel.hat).mag # rrel.mag*sin(theta)
        alpha = asin(dy/(2*Ratom))  # alpha is the angle of the triangle composed of rrel, path of atom j, and a line from the center of atom i to the center of atom j where atome j hits atom i
        d = (2*Ratom)*cos(alpha)-dx # distance traveled into the atom from first contact
        deltat = d/vrel.mag         # time spent moving from first contact to position inside atom

        #### CHANGE L'INTERPÉNÉTRATION DES SPHÈRES PAR LA CINÉTIQUE DE COLLISION ####
        posi = posi-vi*deltat   # back up to contact configuration
        posj = posj-vj*deltat
        pcomi = p[i]-mass*Vcom  # transform momenta to center-of-momentum (com) frame
        pcomj = p[j]-mass*Vcom
        rrel = hat(rrel)    # vecteur unitaire aligné avec rrel
        pcomi = pcomi-2*dot(pcomi,rrel)*rrel # bounce in center-of-momentum (com) frame
        pcomj = pcomj-2*dot(pcomj,rrel)*rrel

        p[i] = pcomi+mass*Vcom # transform momenta back to lab frame
        p[j] = pcomj+mass*Vcom
        apos[i] = posi+(p[i]/mass)*deltat # move forward deltat in time, ramenant au même temps où sont rendues les autres sphères dans l'itération
        apos[j] = posj+(p[j]/mass)*deltat
        if (i or j) == atom_id:
            distParcouru.append(followParticule(posParticule))
            dtCollision.append(tickParticule)
            if i == atom_id:
                vParticule.append(vi)
            elif j == atom_id:
                vParticule.append(vj)
            reset = True
    if reset:
        posParticule = []
        tickParticule = 0
        reset = False
    posParticule.append(apos[atom_id])
    tickParticule += 1


p_2 = []
for vec in p:
    p_2.append(mag2(vec))

avg_p_2 = np.mean(p_2)
print(avg_p_2)

T_final = (2/3)*avg_p_2/(2*mass*k)
print("Température initiale =", T)
print("Température finale =", T_final)
print("Rapport T/T_final =", T/T_final)

# Einit = (3/2)*k*T
# Efinal = avg_p_2/(2*mass)
# print("Einit =", Einit)
# print("Efinal =", Efinal)
# print("Einit/Efinal =", Einit/Efinal)

print("Distance parcourue entre chaque collision :", distParcouru)
print("Nombre de ticks entre chaque collision :", dtCollision)

print("I_moy = ", np.mean(distParcouru), "unités")
print("τ =", np.mean(dtCollision), "ticks")

vParticule_avg = np.mean(vParticule)
print("Vitesse vectorielle de la particule à chaque collision :", vParticule)
print("Vitesse vectorielle moyenne de la particule :", vParticule_avg)

v_norm = []
v_2 = []
v_x = []
for elem in vParticule:
    v_norm.append(mag(elem))
    v_2.append(mag2(elem))
    v_x.append(elem.x)

v_x = np.array(v_x)
v_x_2 = v_x**2

v_norm_normalized = np.array(v_norm)/max(v_norm)
v_2_normalized = np.array(v_2)/max(v_2)
v_x_2_normalized = v_x_2/max(v_x_2)

bin_nb = 24
bin_range = (0, 1)

param_list = []
for arr in [v_norm_normalized, v_2_normalized, v_x_2_normalized]:
    rms = np.sqrt(np.mean(arr**2))
    var_list = [np.mean(arr), mode(arr, axis=None, keepdims=False)[0], np.median(arr), rms]
    param_list.append(var_list)

i=0
for var in param_list:

    rounding_list = []
    for elem in var:

        rounding_list.append(round(elem, 3))
    param_list[i] = rounding_list
    i += 1

fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
ax[0].hist(v_norm_normalized, bins=bin_nb, alpha=0.5, label=f" Moyenne = {param_list[0][0]}\n"
                                               f" Mode = {param_list[0][1]}\n Médiane = {param_list[0][2]}"
            f"\n RMS = {param_list[0][3]}", color="b")
ax[0].legend(loc="upper right", fontsize="11")
ax[0].set_title(r"Distribution de $||\vec{v}||$", fontsize="14")
ax[1].hist(v_2_normalized, bins=bin_nb, alpha=0.5, label=f" Moyenne = {param_list[1][0]}\n"
                                               f" Mode = {param_list[1][1]}\n Médiane = {param_list[1][2]}"
            f"\n RMS = {param_list[1][3]}", color="orange")
ax[1].legend(loc="upper right", fontsize="11")
ax[1].set_title(r"Distribution de $v^2$", fontsize="14")
ax[2].hist(v_x_2_normalized, bins=bin_nb, alpha=0.5, label=f" Moyenne = {param_list[2][0]}\n"
                                               f" Mode = {param_list[2][1]}\n Médiane = {param_list[2][2]}"
            f"\n RMS = {param_list[2][3]}", color="g")
ax[2].legend(loc="upper right", fontsize="11")
ax[2].set_title(r"Distribution de $v_x^2$", fontsize="14")
fig.text(0.5, 0.04, 'Amplitude normalisée [-]', ha='center', fontsize="16")
fig.text(0.04, 0.5, 'Fréquence [-]', va='center', rotation='vertical', fontsize="16")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

v_x_normalized = v_x/max(v_x)

avg = np.mean(v_x_normalized)
var = np.var(v_x_normalized)
pdf_x = np.linspace(-1, 1)
pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)

mu, std = norm.fit(v_x_normalized)
xmin, xmax = -1, 1
x_fitted = np.linspace(xmin, xmax, 100)
y_fitted = norm.pdf(x_fitted, mu, std)

plt.figure()
plt.hist(v_x_normalized, alpha=0.5, bins=bin_nb, range=(-1, 1), density=True, label="Données simulées")
plt.plot(pdf_x, pdf_y, '--', color="k", label="Ajustement de courbe gaussien")
plt.title(r"Distribution de $v_x$ pour une particule", fontsize="14")
plt.xlabel('Amplitude normalisée [-]', fontsize="16")
plt.ylabel('Fréquence normalisée[-]', fontsize="16")
plt.legend(fontsize="12")
plt.xticks(fontsize="12")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

rms = np.sqrt(np.mean(v_x_normalized**2))
var_list = [np.mean(arr), mode(arr, axis=None, keepdims=False)[0], np.median(arr), rms]


print(f"Moyenne = {avg}, Mode = {mode(v_x_normalized, axis=None, keepdims=False)[0]},"
      f" Médiane = {np.median(v_x_normalized)}, RMS = {rms}, STD = {sqrt(var)}")

v_x_all = []

for elem in vitesse:
    v_x_all.append(elem.x)

v_x_all_normalized = np.array(v_x_all)/max(v_x_all)

avg = np.mean(v_x_all_normalized)
var = np.var(v_x_all_normalized)
pdf_x = np.linspace(-1, 1)
pdf_y = 1.0 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (pdf_x - avg) ** 2 / var)

mu, std = norm.fit(v_x_all_normalized)
xmin, xmax = -1, 1
x_fitted = np.linspace(xmin, xmax, 100)
y_fitted = norm.pdf(x_fitted, mu, std)

plt.figure()
plt.hist(v_x_all_normalized, alpha=0.5, bins=bin_nb, range=(-1, 1), density=True, label="Données simulées")
plt.plot(pdf_x, pdf_y, '--', color="k", label="Ajustement de courbe gaussien")
plt.title(r"Distribution de $v_x$ à un instant pour toutes les particules", fontsize="14")
plt.xlabel('Amplitude normalisée [-]', fontsize="16")
plt.ylabel('Fréquence normalisée[-]', fontsize="16")
plt.legend(fontsize="12")
plt.xticks(fontsize="12")
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

rms = np.sqrt(np.mean(v_x_all_normalized**2))
var_list = [np.mean(arr), mode(arr, axis=None, keepdims=False)[0], np.median(arr), rms]


print(f"Moyenne = {avg}, Mode = {mode(v_x_all_normalized, axis=None, keepdims=False)[0]},"
      f" Médiane = {np.median(v_x_all_normalized)}, RMS = {rms}, STD = {sqrt(var)}")