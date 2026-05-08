import numpy as np
import time
import matplotlib.pyplot as plt

def norma_frobenius(A):
    return np.sqrt(np.sum(A ** 2))

def QR_Gram_Schmidt(A):
    m, n = A.shape
    Q = np.copy(A).astype(float)
    R = np.zeros((n, n))
    
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

def QR_Householder(A):
    m, n = A.shape
    R = np.copy(A).astype(float)
    Q = np.eye(m)
    
    for k in range(n):
        x = np.copy(R[k:, k])
        norm_x = norma_frobenius(x)
        
        semn = 1 if x[0] >= 0 else -1
        alpha = -semn * norm_x
        
        e0 = np.zeros_like(x)
        e0[0] = 1.0
        v = x - alpha * e0
        
        norm_v = norma_frobenius(v)
        if norm_v > 1e-14:
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

def QR_iteration(A, Q, TOL=1e-6):
    T = Q.T @ A @ Q
    V = np.copy(Q)

    while norma_extra_diag(T) > TOL: 
        Q_k, R_k = QR_Gram_Schmidt(T)
        T = R_k @ Q_k
        V = V @ Q_k  

    return T, V

def SVD(A, TOL=1e-14):
    ATA = A.T @ A
    n = np.shape(ATA)[0]
    Q, T = Tridiag_Householder(ATA)
    T_final, V = QR_iteration(ATA, Q, TOL)
    lambdas=np.diag(T_final) #valorile proprii sunt pe diagonala, V=vectorii proprii ai lui ATA
    sigmas=np.sqrt(np.maximum(lambdas,0)) #pentru a evita erori de tip NaN folosim np.maximum care forteaza valorile mici (ex:-1.2e-16) sa devina zero
    idx = np.argsort(sigmas)[::-1] #ordonam valorile proprii in ordine descrescatoare , ::-1 inverseaza ordinea si obtinem indicii pt sortarea descrescatoare
    sigmas = sigmas[idx]  #sortam vectorii proprii in aceeasi ordine descrescatoare
    V = V[:, idx] 
    S=np.diag(sigmas) #matricea sigma
    m, n=A.shape
    U = np.zeros((m, m)) 
    r = 0 # contor pt vectorii deja adaugati in U
    for i in range(len(sigmas)):
        if sigmas[i] > TOL:
            U[:, i] = (A @ V[:, i]) / sigmas[i]
            r += 1
    # gram schmidt - folosesc vectorii bazei canonice (matricea identitate) pe care ii ortogonalizez
    I=np.eye(m)
    curent_col = r
    for k in range(m):
        if curent_col >= m:
            break 
        v_posibil= I[:, k].copy()
        for i in range(curent_col):
            R_ik = np.dot(U[:, i], v_posibil)
            v_posibil = v_posibil - R_ik * U[:, i] 
        norm_v = norma_frobenius(v_posibil)
        if norm_v > TOL:
            U[:, curent_col] = v_posibil / norm_v
            curent_col += 1
    S = np.zeros((m, n))
    np.fill_diagonal(S, sigmas)
    return U, S , V.T

#functii pentru tensori
#matricizarea foloseste ca argumente tensorul T si modul in care vrem sa il matricizam (1,2,3)

def transpose_axes()