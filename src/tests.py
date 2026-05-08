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

def test_QR_Householder(nr_teste_randomizate=10,matrici_test=genereaza_matrici_test()):
    for i in range(nr_teste_randomizate):
        A = np.random.rand(10, 10)
        Q, R = QR_Householder(A)
        assert np.allclose(Q @ R, A), f"Testul {i+1} a eșuat, Q @ R: {Q @ R}, A: {A}"
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1])), f"Testul {i+1} a eșuat, Q.T @ Q: {Q.T @ Q}, I: {np.eye(Q.shape[1])}"
    for i, A in enumerate(matrici_test):
        if np.isnan(A).any():
            continue      
        if np.isinf(A).any():
            continue  
        Q, R = QR_Householder(A)
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
    print("Toate testele pentru QR_Householder au trecut")

#tridiagonalizare
def este_tridiagonala(A, tol=1e-10):
    n, m = A.shape

    if n != m:
        return False

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and not np.isclose(A[i, j], 0, atol=tol):
                return False

    return True

def test_tridiagonalizare(nr_teste_randomizate=10):
    for i in range(nr_teste_randomizate):
        A = np.random.rand(10, 10)
        A= (A + A.T) / 2

        Q,T = Tridiag_Householder(A)
        assert np.allclose(Q @ T @ Q.T, A), f"Testul {i+1} a eșuat, Q @ T @ Q.T: {Q @ T @ Q.T}, A: {A}"
        assert este_tridiagonala(T), f"Testul {i+1} a eșuat, T nu este tridiagonală:\n{T}"
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1])), f"Testul {i+1} a eșuat, Q.T @ Q: {Q.T @ Q}, I: {np.eye(Q.shape[1])}"
        assert np.allclose(T, T.T), f"Testul {i+1} a eșuat, T nu este simetrică:\n{T}"
        assert np.allclose(np.sort(np.linalg.eigvals(T)),np.sort(np.linalg.eigvals(A))), f"Testul {i+1} a eșuat, valorile proprii ale lui T nu sunt egale cu cele ale lui A:\nValorile proprii T:\n {np.linalg.eigvals(T)}, \nValorile proprii A: \n{np.linalg.eigvals(A)}"
    print("Toate testele pentru tridiagonalizare au trecut")

def test_QR_iteration_aproximare_valori_proprii(nr_teste_randomizate=10):
    for i in range(nr_teste_randomizate):
        A = np.random.rand(100, 100)
        A= (A + A.T) / 2
        Q, T = Tridiag_Householder(A)
        T_final, Q_final = QR_iteration(A, Q)
        assert np.allclose(np.sort(np.linalg.eigvals(T_final)),np.sort(np.linalg.eigvals(A))), f"Testul {i+1} a eșuat, valorile proprii ale lui T_final nu sunt egale cu cele ale lui A:\nValorile proprii T_final:\n {np.linalg.eigvals(T_final)}, \nValorile proprii A: \n{np.linalg.eigvals(A)}"
    print("Toate testele pentru QR_iteration_aproximare_valori_proprii au trecut")
    
def __main__():
    teste=1
    test_norma(teste)
    test_QR_Gram_Schmidt(teste)
    test_QR_Householder(teste)
    test_tridiagonalizare(teste)
    start_time = time.time()
    test_QR_iteration_aproximare_valori_proprii(teste)
    end_time = time.time()
    print(f"Testele {teste} pentru QR_iteration_aproximare_valori_proprii au durat {end_time - start_time:.4f} secunde")
__main__()