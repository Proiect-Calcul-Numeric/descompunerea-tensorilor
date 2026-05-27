#va urasc din tot sufletul meu nu mai suport am stat 5 ore sa fac un 
#amarat de svd sa mearga pentru descompunerea tucker si acum imi da eroare 
#la testul 5 pentru marginea teoretica ,nu stiu de ce mai stau o ora sa mi dau seama 
#(mi am dat seama)

import numpy as np
import time
import matplotlib.pyplot as plt

def norma_frobenius(A):
    return np.sqrt(np.sum(A ** 2))

def QR_Gram_Schmidt(A):
    m, n = A.shape
    Q = np.copy(A).astype(float)
    R = np.zeros((n, n))

    if(np.all(A == 0)): #caz special pentru matricea nula , toate coloanele sunt deja ortogonale si norma e zero
        return Q, R
    
    for k in range(n):
        for i in range(k):
            R[i, k] = np.dot(Q[:, i], Q[:, k])
            Q[:, k] = Q[:, k] - R[i, k] * Q[:, i]

        R[k, k] = norma_frobenius(Q[:, k])
        
        if R[k, k] > 1e-14:
            Q[:, k] = Q[:, k] / R[k, k]
            
    return Q, R

def norma_extra_diag(A):
    A_fara_diag = A - np.diag(np.diagonal(A))
    return norma_frobenius(A_fara_diag)

def QR_Householder(A,TOL=1e-4):
    m, n = A.shape
    R = np.copy(A).astype(float)
    Q = np.eye(m)
    
    prag = max(TOL * norma_frobenius(R), 1e-30)

    for k in range(min(m,n)):
        x = np.copy(R[k:, k])
        norm_x = norma_frobenius(x)
        
        semn = 1 if x[0] >= 0 else -1
        alpha = -semn * norm_x
        
        e0 = np.zeros_like(x)
        e0[0] = 1.0
        v = x - alpha * e0
        
        norm_v = norma_frobenius(v)
        if norm_v > prag:
            v = v / norm_v
            H_k = np.eye(m - k) - 2 * np.outer(v, v)
        else:
            H_k = np.eye(m - k)
            
        H = np.eye(m)
        H[k:, k:] = H_k
        
        R = H @ R
        Q = Q @ H 
        
    return Q, R
def Tridiag_Householder(A):
    n = np.shape(A)[0]
    T = np.copy(A)
    Q = np.eye(n)
    
    for k in range(n - 2):
        x = np.copy(T[k + 1:, k]) 
        norm_x = norma_frobenius(x) 
        
        semn = 1 if x[0] >= 0 else -1
        alpha = -semn * norm_x
        
        e0 = np.zeros_like(x)
        e0[0] = 1.0
        v = x - alpha * e0
        
        norm_v = norma_frobenius(v) 
        if norm_v > 1e-14: 
            v = v / norm_v
            H_k = np.eye(n - k - 1) - 2 * np.outer(v, v)
        else:
            H_k = np.eye(n - k - 1)
            
        H = np.eye(n)
        H[k + 1:, k + 1:] = H_k
        
        T = H @ T @ H
        Q = Q @ H
        
    return Q, T

def toate_extra_diag_sub_prag(A, prag):
    A_fara_diag = A - np.diag(np.diagonal(A))
    return np.all(np.abs(A_fara_diag) <= prag)

def QR_iteration(A, Q, TOL=1e-4, max_iter=1000):
    T = Q.T @ A @ Q
    V = np.copy(Q)
    n = T.shape[0]
    I = np.eye(n)

    prag = max(TOL * norma_frobenius(T), 1e-30)
    iteratii = 0
    while not toate_extra_diag_sub_prag(T, prag) and iteratii < max_iter:
        # shift Wilkinson pentru a evita stagnarea pe blocuri 2x2
        if n >= 2:
            if abs(T[n - 1, n - 2]) < 1e-15:
                mu = T[n - 1, n - 1]
            else:
                delta = (T[n - 2, n - 2] - T[n - 1, n - 1]) / 2.0
                semn = 1.0 if delta >= 0 else -1.0
                numitor = abs(delta) + np.sqrt(delta ** 2 + T[n - 1, n - 2] ** 2)
                if numitor < 1e-15:
                    mu = T[n - 1, n - 1]
                else:
                    mu = T[n - 1, n - 1] - semn * T[n - 1, n - 2] ** 2 / numitor
        else:
            mu = T[n - 1, n - 1]

        Q_k, R_k = QR_Householder(T - mu * I)
        T = R_k @ Q_k + mu * I

        V = V @ Q_k
        iteratii += 1
    return T, V #T este matricea tridiagonalizata care contine valorile proprii pe diagonala, iar V contine vectorii proprii ai lui A


def SVD(A, TOL=1e-14):
    m, n=A.shape

    def completeaza_baza_ortonormala(B, dimensiune, start_col):
        I=np.eye(dimensiune)
        curent_col = start_col
        for k in range(dimensiune):
            if curent_col >= dimensiune:
                break
            v_posibil= I[:, k].copy()
            for i in range(curent_col):
                R_ik = np.dot(B[:, i], v_posibil)
                v_posibil = v_posibil - R_ik * B[:, i]
            norm_v = norma_frobenius(v_posibil)
            if norm_v > TOL:
                B[:, curent_col] = v_posibil / norm_v
                curent_col += 1
        return B

    if n <= m:
        ATA = A.T @ A
        Q, T = Tridiag_Householder(ATA)
        T_final, V = QR_iteration(ATA, Q, TOL)
        lambdas=np.diag(T_final) #valorile proprii sunt pe diagonala, V=vectorii proprii ai lui ATA
        sigmas=np.sqrt(np.maximum(lambdas,0)) #pentru a evita erori de tip NaN folosim np.maximum care forteaza valorile mici (ex:-1.2e-16) sa devina zero
        idx = np.argsort(sigmas)[::-1] #ordonam valorile proprii in ordine descrescatoare
        sigmas = sigmas[idx]
        V = V[:, idx]

        U = np.zeros((m, m))
        prag_sigma = TOL * max(A.shape) * sigmas[0] if len(sigmas) > 0 else TOL
        r = 0
        for i in range(min(n, m)):
            if sigmas[i] > prag_sigma:
                u = (A @ V[:, i]) / sigmas[i]
                for j in range(r):
                    u = u - np.dot(U[:, j], u) * U[:, j]
                norm_u = norma_frobenius(u)
                if norm_u > TOL:
                    U[:, r] = u / norm_u
                    r += 1
        U = completeaza_baza_ortonormala(U, m, r)
    else:
        AAT = A @ A.T
        Q, T = Tridiag_Householder(AAT)
        T_final, U = QR_iteration(AAT, Q, TOL)
        lambdas=np.diag(T_final) #valorile proprii sunt pe diagonala, U=vectorii proprii ai lui AAT
        sigmas=np.sqrt(np.maximum(lambdas,0))
        idx = np.argsort(sigmas)[::-1]
        sigmas = sigmas[idx]
        U = U[:, idx]

        V = np.zeros((n, n))
        prag_sigma = TOL * max(A.shape) * sigmas[0] if len(sigmas) > 0 else TOL
        r = 0
        for i in range(min(n, m)):
            if sigmas[i] > prag_sigma:
                v = (A.T @ U[:, i]) / sigmas[i]
                for j in range(r):
                    v = v - np.dot(V[:, j], v) * V[:, j]
                norm_v = norma_frobenius(v)
                if norm_v > TOL:
                    V[:, r] = v / norm_v
                    r += 1
        V = completeaza_baza_ortonormala(V, n, r)

    S = np.zeros((m, n))
    np.fill_diagonal(S, sigmas)
    return U, S , V.T

def SVD_redus(A,r,TOL=1e-14):
    m, n = A.shape
    rang_maxim = min(m, n)
    r = min(r, rang_maxim)

    if n <= m:
        ATA = A.T @ A
        Q, T = Tridiag_Householder(ATA)
        T_final, V = QR_iteration(ATA, Q, TOL)
        lambdas = np.diag(T_final)
        sigmas = np.sqrt(np.maximum(lambdas, 0))
        idx = np.argsort(sigmas)[::-1][:r]
        sigmas_r = sigmas[idx]
        V_r = V[:, idx]

        U_r = np.zeros((m, r))
        prag_sigma = TOL * max(A.shape) * sigmas_r[0] if len(sigmas_r) > 0 else TOL
        for i in range(r):
            if sigmas_r[i] > prag_sigma:
                u = (A @ V_r[:, i]) / sigmas_r[i]
                for j in range(i):
                    u = u - np.dot(U_r[:, j], u) * U_r[:, j]
                norm_u = norma_frobenius(u)
                if norm_u > TOL:
                    U_r[:, i] = u / norm_u
    else:
        AAT = A @ A.T
        Q, T = Tridiag_Householder(AAT)
        T_final, U = QR_iteration(AAT, Q, TOL)
        lambdas = np.diag(T_final)
        sigmas = np.sqrt(np.maximum(lambdas, 0))
        idx = np.argsort(sigmas)[::-1][:r]
        sigmas_r = sigmas[idx]
        U_r = U[:, idx]

        V_r = np.zeros((n, r))
        prag_sigma = TOL * max(A.shape) * sigmas_r[0] if len(sigmas_r) > 0 else TOL
        for i in range(r):
            if sigmas_r[i] > prag_sigma:
                v = (A.T @ U_r[:, i]) / sigmas_r[i]
                for j in range(i):
                    v = v - np.dot(V_r[:, j], v) * V_r[:, j]
                norm_v = norma_frobenius(v)
                if norm_v > TOL:
                    V_r[:, i] = v / norm_v

    S_r = np.diag(sigmas_r)
    return U_r, S_r, V_r.T

#functii pentru tensori
#matricizarea foloseste ca argumente tensorul T si modul in care vrem sa il matricizam (1,2,3)
#permutare construieste permutarea de axe care muta "mode" pe pozitia 0 , lasand restul in ordine crescatoare
#de ex daca dimensiunea e 3 : mod 0 -> (0,1,2); mod 1 -> (1,0,2); mod 2 -> (2,0,1)
def permutare(N,mode):
    axes = [mode]               # modul ales vine primul
    for i in range(N):
        if i != mode:           # adaugam toti ceilalti, in ordine
            axes.append(i)
    return axes
def permutare_inv(N,mode):
    axes = permutare(N, mode)
    inversa = [0] * N
    for pozitie in range(N):
        ax = axes[pozitie]      # axa originala care a ajuns pe `pozitie`
        inversa[ax] = pozitie   # ea trebuie sa se intoarca pe pozitia `pozitie`
    return inversa
def matricize(T,mode):
    N = T.ndim
    # permutarea care pune `mode` pe pozitia 0
    axes = permutare(N, mode)
    # permutarea axelor tensorilor
    T_permutat = np.transpose(T, axes)
    # calculam numarul de coloane
    # produsul tuturor dimensiunilor in afara de `mode`
    nr_coloane = 1
    for i in range(N):
        if i != mode:
            nr_coloane = nr_coloane * T.shape[i]
    nr_linii = T.shape[mode]
    # aplatizam tensorul intr-o matrice (nr_linii x nr_coloane)
    T_matricizat = T_permutat.reshape(nr_linii, nr_coloane)
    return T_matricizat
# exemplu produsul mode-n : fie un tensor A care apartine lui R^(2x3x2) si o matrice M care apartine lui R^(4x2)
# rezultatul este un tensor B
# de fapt, matricizam tensorul pe modul n , inmultim matriceal si apoi refacem tensorul
# adica spre exemplu : matricizam A pe modul 1,apoi inmultesc M cu A(1) si ne va da o matrice B de mod 1 care apartine lui R^(4x6)
# apoi refacem tensorul B , primele 3 coloane -> slice 1, urmatoarele 3 coloane -> slice 2;
# functia : T tensor de forma I_0 ... I_N-1 ; M matrice de forma J , I_mode ; mode : indicele modului pe care aplicam M
# rezultatul are aceeasi forma ca T cu exceptia dimensiunii mode care se schimba din I_mode in J
# se bazeaza pe proprietatea: (T x_n M)_(n) = M @ T_(n)
# adica produsul mod-n e echivalent cu o inmultire matriceala aplicata pe matricizarea tensorului.
def mode_n_product(T,M,mode):
    N   = T.ndim #ndim - number of array dimensions
    I_n = T.shape[mode] #dimensiunea modului pe care aplicam M
    J   = M.shape[0] #numarul de linii al matricei M
    assert M.shape[1] == I_n, (
        f"Dimensiunile nu se potrivesc: "
        f"M are {M.shape[1]} coloane dar modul {mode} are dimensiunea {I_n}"
    )
    T_matricizat = matricize(T, mode)
    B_matricizat = M @ T_matricizat
    axes = permutare(N,mode) #reconstruim tensorul din matrice , mode ul e pe pozitia 0 cu noua dimensiune J
    forma_permutata = [J]
    for i in range(1,N): #restul dimensiunilor in ordinea permutarii
        ax = axes[i]
        forma_permutata.append(T.shape[ax])
    B_permutat = B_matricizat.reshape(forma_permutata)
    axes_inv = permutare_inv(N, mode) #aducem modul inapoi pe pozitia lui originala
    B = np.transpose(B_permutat, axes_inv)
    return B
#tucker zice : fie un tensor A, exista un tensor mic G si trei matrici U^(1), U^(2), U^(3) din care se poate reconstrui o aproximare buna a lui A
#hosvd parcurge fiecare mod , matricizeaza, aplica SVD , pastreaza primele R_n coloane si apoi calculeaza core tensor ul cu produse mod-n succesive
#hosvd nu da aproximarea optima (pt obtim trebuie HOOI - higher order orthogonal iteration)
#fie o matrice A care apartine lui R ^ (I1,I2,I3). tucker decomposition ne da:
# A = G x1 U1 x2 U2 x3 U3
# G este core tensor, iar U^(n) sunt matrici cu coloane ortonormate
# coloanele lui U^(n) sunt vectori proprii ai lui A*A.T; coloanele R_n din U reprezinta cata informatie vreau sa pastrez pe modul n. daca pastrez tot , constructia e exacta
# de ce primele Rn coloane din U(n)? dupa ce facem SVD pe A(n) coloanele lui U(n) ies automat ordonate dupa importanta , prima coloana capteaza cea mai mare variatie, a doua mai putina si tot asa.
#valorile singulare sigma spun cat de importanta este fiecare coloana. daca de exemplu sigma_3 e aproape 0 inseamna ca a treia coloana nu contribuie aproape deloc
# rezumat : R_n coloane din U(n) = cei mai buni R_n vectori ortonormati care descriu cum variaza tensorul de a lungul modului n
# core tensor ul G e tensorul original exprimat intr un nou sistem de coordonate comprimat

def HOSVD(tensor, ranguri):
    N = tensor.ndim
    assert len(ranguri) == N, "Numarul de ranguri trebuie sa fie egal cu numarul de moduri"
    for n in range(N):
        assert ranguri[n] <= tensor.shape[n], "Rangul trebuie sa fie mai mic sau egal cu dimensiunea modului"
    Us = [] 
    for n in range(N):
        A_n = matricize(tensor, n)
        # retinem doar U , V si S nu ne intereseaza pentru HOSVD
        U_trunchiat, _, _ = SVD_redus(A_n, ranguri[n])
        Us.append(U_trunchiat)
    G = tensor.copy().astype(float)
    for n in range(N):
        # U_n^T are shape (R_n, I_n) — comprima dimensiunea n din I_n in R_n
        G = mode_n_product(G, Us[n].T, n)
    return G, Us

#G si Us sunt o reprezentare comprimata a tensorului original
#acum trebuie reconstruit tensorul aproximat din G si Us
#hosvd comprima , reconstruct decomprima

def reconstruct(G, Us):
    N = G.ndim
    A_aprox = G.copy().astype(float)
    for n in range(N):
        # U_n are shape (I_n, R_n) — redimensioneaza dimensiunea modului n din R_n in I_n
        A_aprox = mode_n_product(A_aprox, Us[n], n)
    return A_aprox

#calculam si eroare de reconstructie in norma frobenius : eroare = || A - A_aprox ||_F

def tucker_error(tensor_original, G, Us):
    tensor_aprox = reconstruct(G, Us)
    diferenta = tensor_original - tensor_aprox
    return norma_frobenius(diferenta.ravel())

# pentru tensori mari e de preferat sa calculam eroarea inainte de a reconstrui
# folosim teorema eckhart-young care zice ca suma patratelor valorilor singulare de la r+1 incolo sub radical este egala cu eroarea
# marginea teoretica permite estimarea erorii fara reconstructie

def margine_teoretica(tensor, ranguri):
    N = tensor.ndim
    suma_patrate = 0.0
    for n in range(N):
        A_n = matricize(tensor, n)
        # valorile singulare ale matricizarii
        U, S, Vt = SVD(A_n)
        sigmas = np.diag(S)
        R_n = ranguri[n]
        for i in range(R_n, len(sigmas)):
            suma_patrate = suma_patrate + sigmas[i] ** 2
    return np.sqrt(suma_patrate)            
