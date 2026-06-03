# Descompunerea Tensorilor cu SVD si Tucker

Proiect de Calcul Numeric pentru implementarea manuala a descompunerii valorilor singulare (SVD) si folosirea ei intr-o aplicatie de compresie a imaginilor prin HOSVD/Tucker Decomposition.

Implementarea SVD nu foloseste functii NumPy predefinite pentru descompuneri, diagonalizari sau factorizari in codul principal.

## Structura

- `src/matematica.py` - implementarea metodelor numerice: QR, tridiagonalizare, QR iteration, SVD, SVD redus, HOSVD/Tucker.
- `src/image_compressor.py` - compresia imaginilor folosind descompunerea Tucker.
- `src/main.py` - interfata Streamlit pentru incarcarea si comprimarea imaginilor.
- `src/tests.py` - teste numerice pentru functiile implementate.
- `src/comparison.py` - comparatii intre functiile manuale si functiile NumPy.
- `docs/` - documentatia matematica in LaTeX/PDF.
- `comparison_results/` - grafice cu acuratetea si timpul de rulare.

## Instalare

```bash
pip install -r requirements.txt
```

## Rulare aplicatie

```bash
streamlit run src/main.py
```

Aplicatia permite incarcarea unei imagini, alegerea factorului de compresie si a dimensiunii blocului, apoi afiseaza imaginea originala, imaginea comprimata, raportul de compresie, eroarea relativa si timpul de executie.

## Rulare teste

```bash
python src/tests.py
```

## Rulare comparatii

```bash
python src/comparison.py
```

Graficele generate se salveaza in `comparison_results/`.
