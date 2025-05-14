import math
import argparse
from collections import Counter

def calculate_entropy(file_path):
    """
    Oblicza entropię Shannona dla danego pliku binarnego.

    Entropia jest miarą nieprzewidywalności informacji.
    Dla pliku binarnego, obliczamy ją na podstawie częstotliwości występowania
    poszczególnych bajtów (wartości od 0 do 255).

    Formuła entropii Shannona H:
    H = - Σ (p_i * log2(p_i))
    gdzie p_i to prawdopodobieństwo wystąpienia i-tego symbolu (bajtu).

    Wynik jest podawany w bitach na bajt.
    Teoretyczny zakres dla bajtów to od 0 (gdy plik zawiera tylko jeden powtarzający się bajt)
    do 8 (gdy wszystkie 256 możliwych wartości bajtów występują z jednakowym prawdopodobieństwem).
    """
    try:
        with open(file_path, 'rb') as f:
            # Wczytaj całą zawartość pliku jako ciąg bajtów
            data = f.read()
    except FileNotFoundError:
        print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
        return None
    except Exception as e:
        print(f"Wystąpił błąd podczas otwierania/czytania pliku: {e}")
        return None

    if not data:
        print(f"Plik '{file_path}' jest pusty. Entropia wynosi 0.")
        return 0.0

    # Zlicz wystąpienia każdego bajtu
    # Counter jest wydajniejszy niż ręczne tworzenie słownika lub listy
    byte_counts = Counter(data)
    total_bytes = len(data)
    
    entropy = 0.0
    for byte_value in byte_counts:
        # Prawdopodobieństwo wystąpienia danego bajtu
        probability = byte_counts[byte_value] / total_bytes
        
        # Składnik entropii dla tego bajtu
        # math.log2(x) to logarytm o podstawie 2 z x
        if probability > 0: # log2(0) jest niezdefiniowany
            entropy -= probability * math.log2(probability)
            
    return entropy

def main():
    parser = argparse.ArgumentParser(
        description="Oblicza entropię Shannona dla pliku binarnego (.bin lub dowolnego innego).",
        formatter_class=argparse.RawTextHelpFormatter # Dla lepszego formatowania opisu
    )
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Ścieżka do pliku, dla którego ma być obliczona entropia."
    )
    
    args = parser.parse_args()
    
    entropy_value = calculate_entropy(args.file_path)
    
    if entropy_value is not None:
        print(f"Entropia pliku '{args.file_path}': {entropy_value:.4f} bitów na bajt.")
        if entropy_value > 7.9:
            print("Wysoka entropia sugeruje dane dobrze skompresowane lub zaszyfrowane (lub losowe).")
        elif entropy_value < 2.0:
            print("Niska entropia sugeruje dane o dużej redundancji, np. tekst, nieskompresowane obrazy z dużymi jednolitymi obszarami.")
        else:
            print("Entropia w średnim zakresie.")

if __name__ == "__main__":
    main()