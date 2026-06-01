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
   

#------------------------------------------------------------------------------------------------


def test_norma(nr_teste_randomizate=5 , matrici_test=genereaza_matrici_test()):
    rng = np.random.default_rng(0)
    for i in range(nr_teste_randomizate):
        A = rng.random((10, 10))
        assert np.isclose(norma_frobenius(A), np.linalg.norm(A, 'fro')), f"Testul {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, np.linalg.norm: {np.linalg.norm(A, 'fro')}, matrice: {A}"
    for i, A in enumerate(matrici_test):
        if(np.isnan(A).any()): #caz special cu NaN
            assert np.isnan(norma_frobenius(A)), f"Testul pentru matricea {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, matrice: {A}"
        elif(np.isinf(A).any()): #caz special cu Inf
            assert np.isinf(norma_frobenius(A)), f"Testul pentru matricea {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, matrice: {A}"
        else:    
            assert np.isclose(norma_frobenius(A), np.linalg.norm(A, 'fro')), f"Testul pentru matricea {i+1} a eșuat, norma_frobenius: {norma_frobenius(A)}, np.linalg.norm: {np.linalg.norm(A, 'fro')}, matrice: {A}"
        

    print("Toate testele pentru norma frobenius au trecut")


#------------------------------------------------------------------------------------------------


def test_QR_Gram_Schmidt(nr_teste_randomizate=5,matrici_test=genereaza_matrici_test()):
    rng = np.random.default_rng(1)
    for i in range(nr_teste_randomizate):
        A = rng.random((10, 10))
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


#------------------------------------------------------------------------------------------------


def test_QR_Householder(nr_teste_randomizate=5,matrici_test=genereaza_matrici_test()):
    rng = np.random.default_rng(2)
    for i in range(nr_teste_randomizate):
        A = rng.random((10, 10))
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


#------------------------------------------------------------------------------------------------


def este_tridiagonala(A, tol=1e-10):
    n, m = A.shape

    if n != m:
        return False

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and not np.isclose(A[i, j], 0, atol=tol):
                return False

    return True


#-------------------------------------------------------------------------------------------------


def test_tridiagonalizare(nr_teste_randomizate=5):
    rng = np.random.default_rng(3)
    for i in range(nr_teste_randomizate):
        A = rng.random((10, 10))
        A= (A + A.T) / 2

        Q,T = Tridiag_Householder(A)
        assert np.allclose(Q @ T @ Q.T, A), f"Testul {i+1} a eșuat, Q @ T @ Q.T: {Q @ T @ Q.T}, A: {A}"
        assert este_tridiagonala(T), f"Testul {i+1} a eșuat, T nu este tridiagonală:\n{T}"
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1])), f"Testul {i+1} a eșuat, Q.T @ Q: {Q.T @ Q}, I: {np.eye(Q.shape[1])}"
        assert np.allclose(T, T.T), f"Testul {i+1} a eșuat, T nu este simetrică:\n{T}"
        assert np.allclose(np.sort(np.linalg.eigvals(T)),np.sort(np.linalg.eigvals(A))), f"Testul {i+1} a eșuat, valorile proprii ale lui T nu sunt egale cu cele ale lui A:\nValorile proprii T:\n {np.linalg.eigvals(T)}, \nValorile proprii A: \n{np.linalg.eigvals(A)}"
    print("Toate testele pentru tridiagonalizare au trecut")


#------------------------------------------------------------------------------------------------


def test_QR_iteration_aproximare_valori_proprii(nr_teste_randomizate=3):
    TOL_QR = 2e-1
    rng = np.random.default_rng(4)
    for i in range(nr_teste_randomizate):
        A = rng.random((30, 30))
        A= (A + A.T) / 2
        Q, T = Tridiag_Householder(A)
        T_final, Q_final = QR_iteration(A, Q, max_iter=2000)
        valori_qr = np.sort(np.diag(T_final))
        valori_np = np.sort(np.linalg.eigvalsh(A))
        assert np.allclose(valori_qr, valori_np, atol=TOL_QR), f"Testul {i+1} a eșuat, valorile proprii aproximative nu sunt suficient de apropiate:\nQR diag:\n {valori_qr}, \nNumPy: \n{valori_np}"
        reziduu = np.linalg.norm(A @ Q_final - Q_final @ np.diag(np.diag(T_final)), ord="fro") / np.linalg.norm(A, ord="fro")
        assert reziduu < TOL_QR, f"Testul {i+1} a eșuat, vectorii proprii nu sunt suficient de buni:\nReziduu: {reziduu}"
    print("Toate testele pentru QR_iteration_aproximare_valori_proprii au trecut")


#------------------------------------------------------------------------------------------------


def test_SVD(nr_teste_randomizate=5,matrici_test=genereaza_matrici_test()):
    TOL_SVD = 1e-1
    rng = np.random.default_rng(5)
    for i in range(nr_teste_randomizate):
        A = rng.random((10, 10))
        U, S, Vt = SVD(A)
        sigmas = np.diag(S)
        assert np.allclose(U @ S @ Vt, A, atol=TOL_SVD), f"Testul {i+1} a eșuat, \nU @ S @ Vt: \n{U @ S @ Vt}, \nA: \n{A}"
        assert np.allclose(U.T @ U, np.eye(U.shape[1])), f"Testul {i+1} a eșuat, U.T @ U: {U.T @ U}, I: {np.eye(U.shape[1])}"
        assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0])), f"Testul {i+1} a eșuat, Vt @ Vt.T: {Vt @ Vt.T}, I: {np.eye(Vt.shape[0])}"
        assert np.all(sigmas >= -1e10), f"Testul {i+1} a eșuat, valorile singulare nu sunt nenegative: {S}"
        assert np.all(np.diff(sigmas) <= 0), f"Testul {i+1} a eșuat, valorile singulare nu sunt în ordine descrescătoare: {S}"
        assert np.allclose(sigmas, np.linalg.svd(A, compute_uv=False), atol=TOL_SVD), f"Testul {i+1} a eșuat, valorile singulare nu sunt corecte: {S}, np.linalg.svd: {np.linalg.svd(A, compute_uv=False)}"    
        assert (S.shape == A.shape), f"Testul {i+1} a eșuat, forma valorilor singulare nu este corectă: {S.shape}, așteptat: {A.shape}"

    for i, A in enumerate(matrici_test):
        if np.isnan(A).any():
            continue      
        if np.isinf(A).any():
            continue  
        U, S, Vt = SVD(A)
        assert np.allclose(U @ S @ Vt, A, atol=TOL_SVD), f"Testul pentru matricea {i+1} a eșuat, \n U @ S_matrix @ Vt:\n {U @ S @ Vt},\n A: \n{A}"
        assert np.allclose(U.T @ U, np.eye(U.shape[1])), f"Testul pentru matricea {i+1} a eșuat, U.T @ U: {U.T @ U}, I: {np.eye(U.shape[1])}"
        assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0])), f"Testul pentru matricea {i+1} a eșuat, Vt @ Vt.T: {Vt @ Vt.T}, I: {np.eye(Vt.shape[0])}"
    print("Toate testele pentru SVD au trecut")


#------------------------------------------------------------------------------------------------


def test_SVD_redus(nr_teste_randomizate=5,matrici_test=genereaza_matrici_test()):
    TOL_SVD = 1e-1
    rng = np.random.default_rng(6)

    def verifica_ortonormalitate_componente_active(U, S, Vt, A, descriere):
        sigmas = np.diag(S)
        prag_sigma = TOL_SVD * max(A.shape) * sigmas[0] if len(sigmas) > 0 else TOL_SVD
        componente_active = sigmas > prag_sigma

        U_activ = U[:, componente_active]
        Vt_activ = Vt[componente_active, :]

        if U_activ.shape[1] > 0:
            assert np.allclose(U_activ.T @ U_activ, np.eye(U_activ.shape[1]), atol=TOL_SVD), f"{descriere}: componentele active din U nu sunt ortonormale: {U_activ.T @ U_activ}"
        if Vt_activ.shape[0] > 0:
            assert np.allclose(Vt_activ @ Vt_activ.T, np.eye(Vt_activ.shape[0]), atol=TOL_SVD), f"{descriere}: componentele active din Vt nu sunt ortonormale: {Vt_activ @ Vt_activ.T}"

    for i in range(nr_teste_randomizate):
        A = rng.random((6, 6))
        r = min(A.shape)
        U, S, Vt = SVD_redus(A, r)
        sigmas = np.diag(S)
        assert np.allclose(U @ S @ Vt, A, atol=TOL_SVD), f"Testul {i+1} a eșuat, \nU @ S @ Vt: \n{U @ S @ Vt}, \nA: \n{A}"
        verifica_ortonormalitate_componente_active(U, S, Vt, A, f"Testul {i+1}")
        assert np.all(sigmas >= -1e10), f"Testul {i+1} a eșuat, valorile singulare nu sunt nenegative: {S}"
        assert np.all(np.diff(sigmas) <= 0), f"Testul {i+1} a eșuat, valorile singulare nu sunt în ordine descrescătoare: {S}"
        assert np.allclose(sigmas, np.linalg.svd(A, compute_uv=False), atol=TOL_SVD), f"Testul {i+1} a eșuat, valorile singulare nu sunt corecte: {S}, np.linalg.svd: {np.linalg.svd(A, compute_uv=False)}"    
        assert U.shape == (A.shape[0], r), f"Testul {i+1} a eșuat, forma lui U este incorectă: {U.shape}"
        assert S.shape == (r, r), f"Testul {i+1} a eșuat, forma lui S este incorectă: {S.shape}"
        assert Vt.shape == (r, A.shape[1]), f"Testul {i+1} a eșuat, forma lui Vt este incorectă: {Vt.shape}"

    for i, A in enumerate(matrici_test):
        if np.isnan(A).any():
            continue      
        if np.isinf(A).any():
            continue  
        r = min(A.shape)
        U, S, Vt = SVD_redus(A, r)
        assert np.allclose(U @ S @ Vt, A, atol=TOL_SVD), f"Testul pentru matricea {i+1} a eșuat, \n U @ S_matrix @ Vt:\n {U @ S @ Vt},\n A: \n{A}"
        verifica_ortonormalitate_componente_active(U, S, Vt, A, f"Testul pentru matricea {i+1}")

    A = rng.random((6, 4))
    r = 2
    U, S, Vt = SVD_redus(A, r)
    sigmas = np.diag(S)
    reconstructie = U @ S @ Vt
    assert U.shape == (A.shape[0], r), f"Forma lui U pentru cazul redus este incorectă: {U.shape}"
    assert S.shape == (r, r), f"Forma lui S pentru cazul redus este incorectă: {S.shape}"
    assert Vt.shape == (r, A.shape[1]), f"Forma lui Vt pentru cazul redus este incorectă: {Vt.shape}"
    assert reconstructie.shape == A.shape, f"Reconstructia redusă are forma incorectă: {reconstructie.shape}"
    verifica_ortonormalitate_componente_active(U, S, Vt, A, "Cazul redus")
    assert np.allclose(sigmas, np.linalg.svd(A, compute_uv=False)[:r], atol=TOL_SVD), f"Valorile singulare reduse nu sunt corecte: {sigmas}"
    print("Toate testele pentru SVD_redus au trecut")   


#------------------------------------------------------------------------------------------------


def test_norma_extra_diag(nr_teste_randomizate=5):
    rng = np.random.default_rng(7)
    for i in range(nr_teste_randomizate):
        A = rng.random((10, 10))
        A_fara_diag = A - np.diag(np.diagonal(A))
        assert np.isclose(norma_extra_diag(A), np.linalg.norm(A_fara_diag, 'fro')), f"Testul {i+1} a eșuat, norma_extra_diag: {norma_extra_diag(A)}, np.linalg.norm: {np.linalg.norm(A_fara_diag, 'fro')}, matrice: {A}"
    print("Toate testele pentru norma extra-diagonală au trecut")


#------------------------------------------------------------------------------------------------


def test_toate_extra_diag_sub_prag(nr_teste_randomizate=5):
    rng = np.random.default_rng(8)
    for i in range(nr_teste_randomizate):
        A = rng.random((10, 10))
        prag = 0.5
        A_fara_diag = A - np.diag(np.diagonal(A))
        assert toate_extra_diag_sub_prag(A, prag) == np.all(np.abs(A_fara_diag) <= prag), f"Testul {i+1} a eșuat, toate_extra_diag_sub_prag: {toate_extra_diag_sub_prag(A, prag)}, np.all: {np.all(np.abs(A_fara_diag) <= prag)}, matrice: {A}"
    print("Toate testele pentru toate_extra_diag_sub_prag au trecut")


#------------------------------------------------------------------------------------------------


def test_permutare(nr_teste_randomizate=5):
    cazuri = [
        (3, 0, [0, 1, 2]),
        (3, 1, [1, 0, 2]),
        (3, 2, [2, 0, 1]),
        (4, 2, [2, 0, 1, 3]),
    ]
    for N, mode, rezultat_asteptat in cazuri:
        rezultat = permutare(N, mode)
        assert rezultat == rezultat_asteptat, f"Permutarea pentru N={N}, mode={mode} este {rezultat}, asteptat {rezultat_asteptat}"
    print("Toate testele pentru permutare au trecut")


#------------------------------------------------------------------------------------------------


def test_permutare_inversa(nr_teste_randomizate=5):
    tensor = np.arange(24).reshape(2, 3, 4)
    for mode in range(tensor.ndim):
        axes = permutare(tensor.ndim, mode)
        axes_inv = permutare_inv(tensor.ndim, mode)
        tensor_refacut = np.transpose(np.transpose(tensor, axes), axes_inv)
        assert np.array_equal(tensor_refacut, tensor), f"Permutarea inversa pentru mode={mode} nu reface tensorul initial"
    print("Toate testele pentru permutare inversa au trecut")


#------------------------------------------------------------------------------------------------


def test_matricize(nr_teste_randomizate=5):
    rng = np.random.default_rng(11)
    for i in range(nr_teste_randomizate):
        tensor = rng.random((3, 4, 5))
        for mode in range(tensor.ndim):
            rezultat = matricize(tensor, mode=mode)
            axes = permutare(tensor.ndim, mode)
            rezultat_asteptat = np.transpose(tensor, axes).reshape(tensor.shape[mode], -1)
            assert rezultat.shape == rezultat_asteptat.shape, f"Testul {i+1} a eșuat, forma matricizarii pe mode={mode} este incorectă: {rezultat.shape}"
            assert np.allclose(rezultat, rezultat_asteptat), f"Testul {i+1} a eșuat, valorile matricizarii pe mode={mode} sunt incorecte"
    print("Toate testele pentru matricize au trecut")


#------------------------------------------------------------------------------------------------


def test_mode_n_product(nr_teste_randomizate=5):
    rng = np.random.default_rng(12)
    for i in range(nr_teste_randomizate):
        tensor = rng.random((3, 4, 5))
        for mode in range(tensor.ndim):
            matrice = rng.random((6, tensor.shape[mode]))
            rezultat = mode_n_product(tensor, matrice, mode=mode)
            rezultat_asteptat = np.moveaxis(np.tensordot(matrice, tensor, axes=([1], [mode])), 0, mode)
            forma_asteptata = list(tensor.shape)
            forma_asteptata[mode] = matrice.shape[0]
            assert rezultat.shape == tuple(forma_asteptata), f"Testul {i+1} a eșuat, forma produsului mode={mode} este incorectă: {rezultat.shape}"
            assert np.allclose(rezultat, rezultat_asteptat), f"Testul {i+1} a eșuat, valorile produsului mode={mode} sunt incorecte"
    print("Toate testele pentru mode_n_product au trecut")


#------------------------------------------------------------------------------------------------


def test_HOSVD(nr_teste_randomizate=5):
    rng = np.random.default_rng(13)
    for i in range(nr_teste_randomizate):
        tensor = rng.random((3, 4, 5))
        G, Us = HOSVD(tensor, tensor.shape)
        U1, U2, U3 = Us
        assert U1.shape == (3, 3), f"Testul {i+1} a eșuat, forma lui U1 este incorectă: {U1.shape}"
        assert U2.shape == (4, 4), f"Testul {i+1} a eșuat, forma lui U2 este incorectă: {U2.shape}"
        assert U3.shape == (5, 5), f"Testul {i+1} a eșuat, forma lui U3 este incorectă: {U3.shape}"
        assert G.shape == (3, 4, 5), f"Testul {i+1} a eșuat, forma lui G este incorectă: {G.shape}"
        reconstructie = reconstruct(G, Us)
        assert np.allclose(reconstructie, tensor), f"Testul {i+1} a eșuat, reconstructia nu este suficient de apropiată:\nReconstructie:\n{reconstructie},\nTensor:\n{tensor}"
    print("Toate testele pentru HOSVD au trecut")


#------------------------------------------------------------------------------------------------


def test_reconstruct(nr_teste_randomizate=5):
    rng = np.random.default_rng(14)
    for i in range(nr_teste_randomizate):
        tensor = rng.random((3, 4, 5))
        G, Us = HOSVD(tensor, tensor.shape)
        reconstructie = reconstruct(G, Us)
        assert np.allclose(reconstructie, tensor), f"Testul {i+1} a eșuat, reconstructia nu este suficient de apropiată:\nReconstructie:\n{reconstructie},\nTensor:\n{tensor}"
    print("Toate testele pentru reconstruct au trecut")


#------------------------------------------------------------------------------------------------


def test_tucker_error(nr_teste_randomizate=5):
    rng = np.random.default_rng(4)
    for i in range(nr_teste_randomizate):
        tensor = rng.random((3, 4, 5))
        G, Us = HOSVD(tensor, tensor.shape)
        reconstructie = reconstruct(G, Us)
        eroare_calculata = tucker_error(tensor, G, Us)
        eroare_asteptata = norma_frobenius((tensor - reconstructie).ravel())
        assert np.isclose(eroare_calculata, eroare_asteptata), f"Testul {i+1} a eșuat, tucker_error={eroare_calculata}, asteptat={eroare_asteptata}"
        assert eroare_calculata < 1e-10, f"Testul {i+1} a eșuat, eroarea full-rank este prea mare: {eroare_calculata}"

        G_redus, Us_redus = HOSVD(tensor, (2, 3, 3))
        reconstructie_redusa = reconstruct(G_redus, Us_redus)
        eroare_redusa = tucker_error(tensor, G_redus, Us_redus)
        eroare_redusa_asteptata = norma_frobenius((tensor - reconstructie_redusa).ravel())
        assert np.isclose(eroare_redusa, eroare_redusa_asteptata), f"Testul redus {i+1} a eșuat, tucker_error={eroare_redusa}, asteptat={eroare_redusa_asteptata}"
        assert eroare_redusa >= 0, f"Testul redus {i+1} a eșuat, eroarea nu poate fi negativa: {eroare_redusa}"
    print("Toate testele pentru tucker_error au trecut")


#------------------------------------------------------------------------------------------------


def test_margine_teoretica(nr_teste_randomizate=5):
    rng = np.random.default_rng(15)
    for i in range(nr_teste_randomizate):
        tensor = rng.random((3, 4, 5))
        margine_full = margine_teoretica(tensor, tensor.shape)
        margine_redusa = margine_teoretica(tensor, (2, 3, 3))

        assert np.isfinite(margine_full), f"Testul {i+1} a eșuat, marginea full-rank nu este finita: {margine_full}"
        assert np.isfinite(margine_redusa), f"Testul {i+1} a eșuat, marginea redusa nu este finita: {margine_redusa}"
        assert margine_full >= -1e-12, f"Testul {i+1} a eșuat, marginea full-rank este negativa: {margine_full}"
        assert margine_redusa >= -1e-12, f"Testul {i+1} a eșuat, marginea redusa este negativa: {margine_redusa}"
        assert margine_full < 1e-8, f"Testul {i+1} a eșuat, marginea full-rank trebuie sa fie aproape zero: {margine_full}"
    print("Toate testele pentru margine_teoretica au trecut")


#------------------------------------------------------------------------------------------------


def __main__():
    teste=3

    def ruleaza_test(nume_test, functie_test, *args):
        separator = "=" * 72
        print(f"\n{separator}")
        print(f"Rulez testul: {nume_test}")
        print(separator)
        start_time = time.time()
        functie_test(*args)
        end_time = time.time()
        print("-" * 72)
        print(f"Testele {teste} pentru {nume_test} au durat {end_time - start_time:.4f} secunde")
        print(separator)

    ruleaza_test("norma frobenius", test_norma, teste)
    ruleaza_test("norma extra-diagonală", test_norma_extra_diag, teste)
    ruleaza_test("toate_extra_diag_sub_prag", test_toate_extra_diag_sub_prag, teste)
    ruleaza_test("permutare", test_permutare, teste)
    ruleaza_test("matricize", test_matricize, teste)
    ruleaza_test("mode_n_product", test_mode_n_product, teste)
    ruleaza_test("HOSVD", test_HOSVD, teste)
    ruleaza_test("reconstruct", test_reconstruct, teste)
    ruleaza_test("permutare inversa", test_permutare_inversa, teste)
    ruleaza_test("tucker_error", test_tucker_error, teste)
    ruleaza_test("margine teoretica", test_margine_teoretica, teste)
    ruleaza_test("QR_Gram_Schmidt", test_QR_Gram_Schmidt, teste)
    ruleaza_test("QR_Householder", test_QR_Householder, teste)
    ruleaza_test("tridiagonalizare", test_tridiagonalizare, teste)
    ruleaza_test("QR_iteration_aproximare_valori_proprii", test_QR_iteration_aproximare_valori_proprii,teste)
    ruleaza_test("SVD", test_SVD, teste)
    ruleaza_test("SVD_redus", test_SVD_redus, teste)
__main__()
