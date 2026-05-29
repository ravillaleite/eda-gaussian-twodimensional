import numpy as np
import matplotlib.pyplot as plt

def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob

def d2b(n):

    strtemp = ''
    while n != 0:
        resto = str(int(n%2))
        strtemp = resto + '' + strtemp
        n = np.floor(n/2)

    return strtemp

#teste = d2b(8)
#print (teste, type(teste))
# ----------------------------------------------------------------------------
#Convert real numbers not integers to binary sequences.

#n = 10.7560

def f_d2b(n, bits_de_representacao):

    #bits_de_representacao = 3
    strn = str(n)
    #strn = strn.strip()
    strn = strn.replace(' ', '')

    if strn.find('.') == (-1):
        return d2b(n)
    else:
        k = strn.find('.')

    i_part = strn[0:k]
    f_part = strn[k::]

    number_i_part = int(i_part)
    number_f_part = float(f_part)

    bin_i_part = d2b(number_i_part)


    strtemp = ''
    temp = number_f_part
    #t = '1'
    #s = '0'

    aux = 0
    inf = 0
    sup = 1
    media = (sup - inf) / 2

    while aux < bits_de_representacao:

        if temp >= media:

            strtemp = strtemp + '1'
            inf = media
            media = ((sup - inf) / 2) + inf

        else:

            strtemp = strtemp + '0'
            sup = media
            media = ((sup - inf) / 2) + inf

        aux += 1

    if(i_part == '0'):
        return strtemp
    else:
        return (bin_i_part + '.' + strtemp)
    

V_mod_tilde = 1
excess_noise=0.02
tau = np.arange(0.02, 0.84, 0.02, dtype=np.float32) #transmitância (40 pontos)
#tau = np.linspace(0.02, 0.32, 11).reshape(-1,1)
D = -np.log10(tau)*(10/0.2) # tau = 10^(-alpha*d/10), alpha is the fiber attenuation in dB/km. d = -log10(tau)*10/alpha
V_mod = 4 * V_mod_tilde # it is the variance of the quadrature operator, having a factor of 4 over the variance of the modulating classical random variable

#SNRdB_list = np.linspace(-10, 0, 10)
#SNR_list = 10**(SNRdB_list/10)
SNR_list = (tau * (V_mod) / (1 + excess_noise)).astype(np.float32)
SNRdB_list = (10 * np.log10(SNR_list)).astype(np.float32)
#I_gaussian = np.log2(1 + SNR_lin)/2
print(SNRdB_list)

snrs = tau.size

mu0 = 0
sigma = 1
#sigmar = 1/SNR_list
#sigmar = (np.sqrt((1.02)/(tau*4))).astype(np.float32)

realizacoes = 1000 #64800
bits_de_representacao = 4
max_iterations = 1000

prec = np.float32

Entropia = np.zeros((len(SNR_list), bits_de_representacao), dtype=np.float32)
Pe = np.zeros((len(SNR_list), bits_de_representacao), dtype=np.float32)
Pe_media = np.zeros((len(SNR_list), bits_de_representacao), dtype=np.float32)
Pe_sum = np.zeros((len(SNR_list), bits_de_representacao), dtype=np.float64)

for iterations in range(0, max_iterations):

    print(f"Iteration global: {iterations}")

    M = np.random.normal(mu0, sigma, (100000, 1)).astype(np.float32)
    #Gaussian values of Alice
    x = M

    #cont_SNR = 0
    #Pe_SNR = np.zeros((len(SNRdB_list),bits_de_representacao), dtype=prec)

    kx = np.zeros(realizacoes, dtype = np.int32)
    probabilidade_acumulada_x = np.zeros(realizacoes, dtype=prec)

    Matriz_Alice = np.zeros((realizacoes, bits_de_representacao), dtype=np.int8)
    # compute the ECDF of the samples
    x1, px = ecdf(x)

    #Geração da matriz binária de Alice
    for i in range(0, realizacoes):

        indices = np.where(x1 == x[i])[0]  # Get the indexes where x1 == x[i]

        if indices.size > 0:  # Garante que há pelo menos um índice correspondente
            kx[i] = int(indices[0])  # Obtém o primeiro índice encontrado
            probabilidade_acumulada_x[i] = px[kx[i]]  # Atribui a probabilidade acumulada

        #foi gerado por Alice
        if probabilidade_acumulada_x[i] == 1:
            probabilidade_acumulada_x[i] = 0.99999
        elif probabilidade_acumulada_x[i] == 0:
            probabilidade_acumulada_x[i] = 0.00001

        expansao_base2_Alice_str = f_d2b(probabilidade_acumulada_x[i], bits_de_representacao)

        ii = 0
        for a in expansao_base2_Alice_str:
            #print('a: ', a)
            #Alice.append(a)
            Matriz_Alice[i,ii] = a
            ii += 1

    #print(np.shape(Matriz_Alice))

    # Gerando x e s(x) ---------------------------------------------------------------------------------------
    x = np.copy(Matriz_Alice)

    #d_hamming = np.zeros((len(SNR_list), realizacoes), dtype=prec)
    #dist = np.zeros(len(SNR_list), dtype=np.uint8)

    for glob in range(0, snrs):
        #sigmar = sigma / np.sqrt(SNR_list[glob])
        #Mr = np.random.normal(mu0, sigmar, (100000, 1)) #ruído

        sigmar = np.sqrt((1.02)/(tau[glob]*4))
        Mr = np.random.normal(mu0, sigmar-0.25*sigmar, (100000, 1)).astype(np.float32) #ruído

        #Valores de Bob
        y = M + Mr
        y1, py = ecdf(y)

        ky = np.zeros(realizacoes, dtype = np.int32)
        probabilidade_acumulada_y = np.zeros(realizacoes, dtype=prec)
        Matriz_Bob = np.zeros((realizacoes, bits_de_representacao), dtype = np.int8)
        #Geração da matriz binária de Bob
        for ii in range(0, realizacoes):

            indices_y = np.where(y1 == y[ii])[0]
            if indices_y.size > 0:  # Garante que há pelo menos um índice correspondente
                ky[ii] = int(indices_y[0])  # Obtém o primeiro índice encontrado
                probabilidade_acumulada_y[ii] = py[ky[ii]]  # Atribui a probabilidade acumulada

            #foi recebido por Bob
            if probabilidade_acumulada_y[ii] == 1:
                probabilidade_acumulada_y[ii] = 0.99999
            elif probabilidade_acumulada_y[ii] == 0:
                probabilidade_acumulada_y[ii] = 0.00001

            expansao_base2_Bob_str = f_d2b(probabilidade_acumulada_y[ii], bits_de_representacao)
            #Bob = []
            iii = 0
            for b in expansao_base2_Bob_str:
                #print("b", b)
                Matriz_Bob[ii,iii] = b
                iii = iii + 1

        # Gerando y e s(y) --------------------------------------------------------------------
        y = np.copy(Matriz_Bob)
        
        for canal in range(bits_de_representacao):
            d_hamming = np.bitwise_xor(x[:,canal], y[:,canal]).astype(np.int8)
            dist = np.sum(d_hamming)
            Pe = dist / realizacoes
            Pe_sum[glob,canal] += Pe

            Entropia[glob, canal] += -Pe*np.log2(Pe) - (1-Pe)*np.log2(1-Pe)

        # proteção numérica
        #eps = 1e-12
        #Pe_clip = np.clip(Pe, eps, 1 - eps)

        #Pe[glob] = dist[glob] / realizacoes
        #Entropia[glob] = -Pe[glob]*np.log2(Pe[glob]) - (1-Pe[glob])*np.log2(1-Pe[glob])
        #Pe_media[glob] = Pe_media[glob] + Pe[glob]



Pe_media = Pe_sum/max_iterations
Entropia = Entropia/max_iterations
print(f'Pe media: ', Pe_media)
Capacidade_media = 1-Entropia
#print(f"Capacidade média: {Capacidade_media}.")

fig, ax1 = plt.subplots()

ax1.plot(SNRdB_list, Pe_media)
plt.show()

np.savetxt('Ber_unmatched-02.txt', Pe_media)
np.savetxt('Capacidade_unmatched-02.txt', Capacidade_media)