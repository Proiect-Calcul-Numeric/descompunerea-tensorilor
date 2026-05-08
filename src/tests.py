from matematica import *
import matplotlib.pyplot as plt

# Matrici de test : 
# 1. Matrice zero
# 2. Matrice cu un singur element
# 3. Vector coloană
# 4. Vector linie
# 5. Matrice cu valori negative (invarianta la semn)
# 6. Matrice rară
# 7. Matrice cu valori foarte mari
# 8. Matrice cu valori foarte mici
# 9. Matrice cu valori mixte
# 10. Matrice cu NaN
# 11. Matrice cu Inf
# 12. Matrice simetrică

def genereaza_matrici_test():
    
    matrici = []
    matrici.append(np.array([[0,0,0], [0,0,0], [0,0,0]])) #matrice zero
    matrici.append(np.array([[5]])) #matrice cu un singur element
    matrici.append(np.array([[1], [2], [3]])) #vector coloana
    matrici.append(np.array([[1, 2, 3]])) #vector linie
    matrici.append(np.array([[-1, -2], [-3, -4]])) #matrice cu valori negative
    matrici.append(np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])) #matrice rara
    matrici.append(np.array([[1e10, 2e10], [3e10, 4e10]])) #matrice cu valori foarte mari
    matrici.append(np.array([[1e-10, 2e-10], [3e-10, 4e-10]])) #matrice cu valori foarte mici
    matrici.append(np.array([[1e-10, 2e10], [3e-10, 4e10]])) #matrice cu valori mixte
    matrici.append(np.array([[np.nan, 1], [2, 3]])) #matrice cu NaN
    matrici.append(np.array([[np.inf, 1], [2, 3]])) #matrice cu Inf
    matrici.append(np.array([[1, 2], [2, 1]])) #matrice simetrică

    nume_matrici = [
        "Matrice zero",
        "Matrice cu un singur element",
        "Vector coloană",
        "Vector linie",
        "Matrice cu valori negative/invarianta la semn",
        "Matrice rară",
        "Matrice cu valori foarte mari",
        "Matrice cu valori foarte mici",
        "Matrice cu valori mixte",
        "Matrice cu NaN",
        "Matrice cu Inf",
        "Matrice simetrică"
    ]
    return matrici
   

def test_norma(nr_teste_randomizate=10 , matrici_test=genereaza_matrici_test()):
    for i in range(nr_teste_randomizate):
        A = np.random.rand(10, 10)
        assert np.isclose(norma_frobenius(A), np.linalg.norm(A, 'fro')), f"Testul {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, np.linalg.norm: {np.linalg.norm(A, 'fro')}, matrice: {A}"
    for i, A in enumerate(matrici_test):
        if(np.isnan(A).any()): #caz special cu NaN
            assert np.isnan(norma_frobenius(A)), f"Testul pentru matricea {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, matrice: {A}"
        elif(np.isinf(A).any()): #caz special cu Inf
            assert np.isinf(norma_frobenius(A)), f"Testul pentru matricea {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, matrice: {A}"
        else:    
            assert np.isclose(norma_frobenius(A), np.linalg.norm(A, 'fro')), f"Testul pentru matricea {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, np.linalg.norm: {np.linalg.norm(A, 'fro')}, matrice: {A}"
        

    print("Toate testele pentru norma frobenius au trecut")

def test_QR_Gram_Schmidt(nr_teste_randomizate=10,matrici_test=genereaza_matrici_test()):
    for i in range(nr_teste_randomizate):
        A = np.random.rand(10, 10)
        Q, R = QR_Gram_Schmidt(A)
        assert np.allclose(Q @ R, A), f"Testul {i+1} a eșuat, Q @ R: {Q @ R}, A: {A}"
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1])), f"Testul {i+1} a eșuat, Q.T @ Q: {Q.T @ Q}, I: {np.eye(Q.shape[1])}"
    for i, A in enumerate(matrici_test):
        if np.isnan(A).any():
            continue      
        if np.isinf(A).any():
            continue  
        Q, R = QR_Gram_Schmidt(A)
        assert np.allclose(Q @ R, A), f"Testul pentru matricea {i+1} a eșuat, Q @ R: {Q @ R}, A: {A}"
        m, n = A.shape
        rang = np.linalg.matrix_rank(A)
        if m >= n and rang == n:
            assert np.allclose(Q.T @ Q, np.eye(Q.shape[1])), f"Testul pentru matricea {i+1} a eșuat, Q.T @ Q: {Q.T @ Q}, I: {np.eye(Q.shape[1])}"
        else:
            norme = np.linalg.norm(Q, axis=0)
            coloane_nenule = norme > 1e-14

            Q_nenul = Q[:, coloane_nenule]

            if Q_nenul.shape[1] > 0:
                assert np.allclose(Q_nenul.T @ Q_nenul,np.eye(Q_nenul.shape[1])), f"Coloanele nenule din Q nu sunt ortonormale:\n{Q_nenul.T @ Q_nenul}"
    print("Toate testele pentru QR_Gram_Schmidt au trecut")

def __main__():
    test_norma()
    test_QR_Gram_Schmidt()

__main__()