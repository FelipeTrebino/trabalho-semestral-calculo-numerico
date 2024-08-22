import numpy as np

def verificar_determinante(A):
    # Calcula o determinante da matriz A
    det = np.linalg.det(A)
    # Retorna True se o determinante for diferente de zero (matriz não singular), e o valor do determinante
    return det != 0, det

def verificar_diagonal_dominante_linhas(A):
    # Obtém o número de linhas (ou colunas, já que é quadrada) da matriz A
    n = A.shape[0]
    # Verifica a dominância diagonal por linhas
    for i in range(n):
        # Soma os valores absolutos dos elementos da linha i, exceto o elemento diagonal
        soma = sum(abs(A[i, j]) for j in range(n) if j != i)
        # Se o valor absoluto do elemento diagonal for menor que a soma, a matriz não é diagonalmente dominante
        if abs(A[i, i]) < soma:
            return False
    # Se passar por todas as linhas sem problemas, retorna True
    return True

def verificar_diagonal_dominante_colunas(A):
    # Obtém o número de linhas (ou colunas) da matriz A
    n = A.shape[0]
    # Verifica a dominância diagonal por colunas
    for j in range(n):
        # Soma os valores absolutos dos elementos da coluna j, exceto o elemento diagonal
        soma = sum(abs(A[i, j]) for i in range(n) if i != j)
        # Se o valor absoluto do elemento diagonal for menor que a soma, a matriz não é diagonalmente dominante
        if abs(A[j, j]) < soma:
            return False
    # Se passar por todas as colunas sem problemas, retorna True
    return True

def verificar_criterio_sassenfeld(A):
    # Obtém o número de linhas (ou colunas) da matriz A
    n = A.shape[0]
    # Inicializa um vetor beta de zeros
    beta = np.zeros(n)
    # Verifica o critério de Sassenfeld para cada linha
    for i in range(n):
        # Calcula a soma ponderada dos elementos das colunas anteriores (usando os valores de beta já calculados)
        soma1 = sum(abs(A[i, j]) * beta[j] for j in range(i))
        # Calcula a soma dos elementos das colunas seguintes
        soma2 = sum(abs(A[i, j]) for j in range(i + 1, n))
        # Calcula o valor de beta para a linha i
        beta[i] = (soma1 + soma2) / abs(A[i, i])
        # Se qualquer beta for maior ou igual a 1, o critério de Sassenfeld não é satisfeito
        if beta[i] >= 1:
            return False, beta
    # Se todos os betas forem menores que 1, o critério de Sassenfeld é satisfeito
    return True, beta

def calcula_jacobi(A, b, tol=0.01, max_iter=100):
    # Determinar o número de variáveis (que é o número de linhas ou colunas da matriz A)
    n_variaveis = A.shape[0]
    
    # Inicializar x com zeros
    x = np.zeros(n_variaveis)
    
    # Vetor para armazenar a nova aproximação
    x_new = np.zeros(n_variaveis)

    total_iteracoes = 0    

    # Iteração do método de Jacobi
    for it in range(max_iter):
        for i in range(n_variaveis):
            # Soma dos termos A[i,j] * x[j] para j != i
            s = sum(A[i, j] * x[j] for j in range(n_variaveis) if j != i)
            # Calcular a nova aproximação x_new[i]
            x_new[i] = (b[i] - s) / A[i, i]
        
        # Calcular a norma infinito de x_new e (x_new - x)
        norm_x_new = max(abs(xi) for xi in x_new)
        norm_diff = max(abs(x_new[i] - x[i]) for i in range(n_variaveis))

        # Calcular o erro relativo
        error = norm_diff / norm_x_new

        # Atualizar x com a nova aproximação
        x = x_new.copy()
        
        total_iteracoes = it + 1

        # Verificar se o erro é menor que a tolerância
        if error < tol:
            break
    
    # Retornar o vetor solução x e o número de iterações realizadas
    return x, total_iteracoes

def calcula_gauss_seidel(A, b, tol = 0.01, max_iterations = 100):

    n = len(b) # Define o tamanho do vetor b (ou seja o número de variáveis)

    # Inicializar x com zeros
    x = np.zeros(n)

    # Vetor para armazenar a nova aproximação
    x_new = np.zeros(n)

    for k in range(max_iterations):        
        for i in range(n):
            # Calcula a soma dos produtos dos elementos anteriores da linha 𝑖da matriz 𝐴 com os elementos correspondentes do vetor solução atual.
            s1 = np.dot(A[i, :i], x_new[:i]) 
            # Calcula a soma dos produtos dos elementos posteriores da linha i da matriz A com os elementos correspondentes do vetor solução anterior.
            s2 = np.dot(A[i, i + 1:], x[i + 1:]) 
            # Atualiza o elemento i do vetor solução.
            x_new[i] = (b[i] - s1 - s2) / A[i, i] 
        
        # Verifica a convergência comparando a norma infinito da diferença entre x novo e antigo com a tolerância.
        if np.linalg.norm(x_new - x, ord=np.inf) < tol: 
             # Retorna a solução e o número de iterações se a convergência for alcançada.
            return x_new, k + 1
        
        x = x_new

    return x, max_iterations # Se não convergir dentro do número máximo de iterações, retorna a solução aproximada e o número máximo de iterações.

def verificar_convergencia(A, b, tol=0.01, max_iter=100):
    determinante_ok, det_valor = verificar_determinante(A)
    if not determinante_ok:
        return "O sistema não tem uma solução única, pois o determinante é zero."

    jacobi_converge = verificar_diagonal_dominante_linhas(A)
    gauss_seidel_converge, _ = verificar_criterio_sassenfeld(A)

    if jacobi_converge and gauss_seidel_converge:
        print("O sistema converge para ambos os métodos (Jacobi e Gauss-Seidel). Calculando as soluções...")
        sol_jacobi, iter_jacobi = calcula_jacobi(A, b, tol, max_iter)
        sol_gauss_seidel, iter_gauss_seidel = calcula_gauss_seidel(A, b, tol, max_iter)
        return {
            "Jacobi": {"solucao": sol_jacobi, "iteracoes": iter_jacobi},
            "Gauss-Seidel": {"solucao": sol_gauss_seidel, "iteracoes": iter_gauss_seidel}
        }

    elif jacobi_converge:
        print("O sistema converge apenas para o método de Jacobi. Calculando as soluções...")
        sol_jacobi, iter_jacobi = calcula_jacobi(A, b, tol, max_iter)
        sol_gauss_seidel, iter_gauss_seidel = calcula_gauss_seidel(A, b, tol, max_iter)
        return {
            "Jacobi": {"solucao": sol_jacobi, "iteracoes": iter_jacobi},
            "Gauss-Seidel": {"solucao": sol_gauss_seidel, "iteracoes": iter_gauss_seidel},
            "Aviso": "O método de Gauss-Seidel foi aplicado, mas não há garantia de convergência."
        }

    elif gauss_seidel_converge:
        print("O sistema converge apenas para o método de Gauss-Seidel. Calculando as soluções...")
        sol_jacobi, iter_jacobi = calcula_jacobi(A, b, tol, max_iter)
        sol_gauss_seidel, iter_gauss_seidel = calcula_gauss_seidel(A, b, tol, max_iter)
        return {
            "Jacobi": {"solucao": sol_jacobi, "iteracoes": iter_jacobi},
            "Gauss-Seidel": {"solucao": sol_gauss_seidel, "iteracoes": iter_gauss_seidel},
            "Aviso": "O método de Jacobi foi aplicado, mas não há garantia de convergência."
        }

    else:
        print("O sistema não converge para nenhum dos métodos, mas as soluções serão calculadas.")
        sol_jacobi, iter_jacobi = calcula_jacobi(A, b, tol, max_iter)
        sol_gauss_seidel, iter_gauss_seidel = calcula_gauss_seidel(A, b, tol, max_iter)
        return {
            "Jacobi": {"solucao": sol_jacobi, "iteracoes": iter_jacobi},
            "Gauss-Seidel": {"solucao": sol_gauss_seidel, "iteracoes": iter_gauss_seidel},
            "Aviso": "Nenhum dos métodos tem garantia de convergência."
        }

def print_sistema(A, b):
    # Identificar a quantidade de variáveis
    n_variaveis = A.shape[1]
    variaveis = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    if n_variaveis > len(variaveis):
        raise ValueError("Número de variáveis maior do que o suportado.")
    
    # Iterar pelas linhas de A e pelos elementos de b para imprimir no formato de polinômio
    for i in range(A.shape[0]):
        partes = []
        
        for j in range(n_variaveis):
            coef = A[i, j]
            if coef != 0:
                parte = f"{coef}{variaveis[j]}"
                partes.append(parte)
        
        equacao_formatada = " + ".join(partes) + f" = {b[i]}"
        equacao_formatada = equacao_formatada.replace("+-", "- ")  # Ajustar sinais
        print(equacao_formatada)

def formatar_resultado_convergencia(resultado):
    if isinstance(resultado, str):
        print(resultado)
        return
    
    for metodo, info in resultado.items():
        if metodo == "Aviso":
            print(f"\nAviso: {info}")
        else:
            print(f"\nMétodo: {metodo}")
            print(f"Solução: {info['solucao']}")
            print(f"Número de Iterações: {info['iteracoes']}")

if __name__  == "__main__":

    # Exemplo 1 do slide
    A1 = np.array([[10, 2, 3],
                [1, 5, 1],
                [2, 3, 10]])

    b1 = np.array([7, -8, 6])

    # Exemplo 2 do slide
    A2 = np.array([[3, -0.1, -0.2],
                [0.1, 7, -0.3],
                [0.3, -0.2, 10]])

    b2 = np.array([7.85, -19.3, 71.4])

    # Exemplo de entrada sistema c 4 equações
    A3 = np.array([[10, -1, 2, 0],
                [-1, 11, -1, 3],
                [2, -1, 10, -1],
                [0, 3, -1, 8]])

    b3 = np.array([-6, -25, 11, -15])

    # Exemplo de entrada sistema c 5 equacoes
    A4 = np.array([
            [2, 1, -1, 0, 0],
            [4, -6, 0, 1, 0],
            [0, -3, 2, 0, 1],
            [1, -1, 1, 1, 0],
            [5, -2, 3, 0, 1]])

    b4 = np.array([-2, 2, -3, -1,-20])

    A5 = np.array([
            [10, 2, 3, 0, 0, 1],
            [1, 11, 1, 1, 0, 0],
            [2, 1, 10, 2, 1, 0],
            [0, 1, 2, 8, 1, 1],
            [0, 0, 1, 1, 9, 1],
            [1, 0, 0, 1, 1, 7]])

    b5 = np.array([7, 13, 17, 19, 21, 23])

    print("1º Exemplo - Exercícios feito em sala \n")

    print_sistema(A1,b1)

    result = verificar_convergencia(A1,b1,tol=0.0001)
    
    formatar_resultado_convergencia(result)

    print("2º Exemplo - Exercícios do slide\n")

    print_sistema(A2,b2)

    result = verificar_convergencia(A2,b2,tol=0.0001)
    
    formatar_resultado_convergencia(result)

    print("\n3º Exemplo - Sistema 4 variaveis\n")

    print_sistema(A3,b3)

    result = verificar_convergencia(A3,b3,tol=0.0001)
    
    formatar_resultado_convergencia(result)
    
    print("\n4º Exemplo - Sistema 5 variaveis\n")

    print_sistema(A4,b4)

    result = verificar_convergencia(A4,b4,tol=0.0001)
    
    formatar_resultado_convergencia(result)

    print("\n5º Exemplo - Sistema 6 variaveis\n")

    print_sistema(A5,b5)

    result = verificar_convergencia(A5,b5,tol=0.0001)
    
    formatar_resultado_convergencia(result)