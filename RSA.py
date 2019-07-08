import random as rd
import numpy as np
from math import sqrt, floor, log2

"""
Camille Claing 
Charly Jeffrey

"""

class RSA:
	
	# Huge prime number
	_PRIME = 74511137535541795231284529565483704199
	def __init__(self, n, k0, e):
		"""RSA Class implémentant l'encryption RSA
		
		Args:
			n (int): taille des blocks
			k0 (int): Taille du padding
		"""
		self.n = n
		self.k0 = k0
		self.e = e
	
	@staticmethod
	def getPrime(): return RSA._PRIME
	
	@staticmethod
	def bigPrime(l : int) -> int:
        """bigPrime Génère un nombre premier de 'l' bits
        
        Args:
            l (int): Taille en bit du nombre
        
        Returns:
            int: Nombre premier sur l bits 
        """		
		# Génère un nombre aléatoire de l-bits (impair)
		n = rd.getrandbits(l) | 1
		n |= (1 << l - 1)

		# Boucle pour s'assurer que n est premier
		while not RSA.millerRabin(n, 5):
			# Génère un nombre aléatoire de l-bit et fait un OR avec 1
			n = rd.getrandbits(l) | 1   # Obligatoirement impair
			n |= (1 << l - 1)
		return n
	
	@staticmethod
	def squareAndMultiply(b : int, n : int, m : int):
		"""squareAndMuliply Méthode pour faire l'exponentiation d'un nombre
		
		Args:
			b (int): Base
			n (int): Exposant
			m (int): Modulo
		
		Returns: b^n (mod m)
		"""
		if type(b) == str: b = int(b)
		if type(n) == str: n = int(n)
		return pow(b, n, m)

		# Array pour stocker la forme binaire de b
		bits = []
		# tant que l'exposant n'est pas nul
		while n != 0:
			# Ajoute le modulo 2 au array
			bits.append(n & 1)
			n >>= 1
		# bin_n == n en binaire renversé

		# Initialise la valeur du résultat
		res = 1
		# Boucle sur les elements de bin_b
		for p in reversed(bits):
			# Multiplie le resultat par lui-meme
			res = res*res
			# Vérifie si l'exposant est nul
			if p == 0:
				# Applique le modulo directement
				res = res % m
			# Sinon
			else:
				# Multiplie par la base
				res = (res * b) % m
		# Fin
		return res
	
	@staticmethod
	def EA(a : int, b : int) -> int:
        """
        EEA Méthode pour déterminer le PGCD de deux nombre entier
        
        Args:
            a (int): Premier nombre entier
            b (int): Second nombre entier
        
        Returns:
            int : Plus grand commun diviseur des nombre a et b
        """	

		# Cas de base
		if (b == 0): return a
		if (a == b): return b
		
		# Appel récursif selon le cas 
		if (a > b): return RSA.EA(b, a%b)
		else : return RSA.EA(a, b%a)
	
	@staticmethod
	def EEA(a : int, b : int) -> int:
		"""EEA Méthode pour déterminer le PGCD de deux nombres entiers
		
		Args:
			a (int): Premier nombre entier
			b (int): Second nombre entier
		
		Returns:
			int[]: Coefficients et plus grand commun diviseur des nombres a et b
		"""
		# S'assure que le nombre 'a' est plus grand
		if (a < b): RSA.EEA(b, a)

		# Initialisation des coefficients := [[x0, x1], [y0, y1]]
		coeff = [[1, 0], [0, 1]]

		# Initialise un tableau pour contenir les coefficients 'x' et 'y' ainsi que le PGCD
		result = [None, None, None]

		# Boucle principale
		while(True) :
			# Détermine q et r
			r = a % b   
			if (r == 0): return result
			q = a // b 

			# Obtient les résultats de la ie itération
			result[0] = coeff[0][0] - q * coeff[0][1]
			result[1] = coeff[1][0] - q * coeff[1][1]
			result[2] = r

			# Modifie les coefficients 
			coeff[0] = [coeff[0][1], result[0]]
			coeff[1] = [coeff[1][1], result[1]]

			# Modifie les valerus de 'a' et 'b' qui seront utilisées pour la prochaine itération
			a = b
			b = r
	
	@staticmethod 
	def Inverse( a : int, m : int):
		"""Inverse Méthode pour effectuer le modulo inverse
		
		Returns:
			int : x | ax = 1 (mod m)
		"""
		inv = RSA.EEA(a, m)[0]
		if inv < 0: inv += m
		return inv
	
	@staticmethod
	def hash_function(m : int, l : int):
		"""hash_function Retourne un hash de longueur 'l' de 'm'
		
		Args:
			m (int): Message à hasher
			l (int): Longueur du hash 
		"""
		if (type(m) == str):
			m = int(m, 2)
		# Convertie le message en nombre
		mhash = bin((m * RSA.getPrime())% pow(2, l-1))
		return mhash[2:].zfill(l)

	

	@staticmethod
	def millerRabin(n : int, s : int) -> bool:
		"""millerRabin Test si un nombre est premier
		
		Args:
			n (int): Nombre à tester
			s (int): Nombre de test
		
		Returns:
			bool: Vrai si n est premier
		"""
		# Vérifie si le nombre 'p' est pair; retourne vrai si 'p' vaut 2 et faux sinon
		if (n % 2 == 0) : return n == 2
		if n == 3: return True

		# Initialise r, u et _p
		u, _n = 0, n-1
		# Détermine la valeur de 'u'
		while _n & 1 == 0 :
			u += 1      # Augmente la valeur de 'u'
			_n //= 2    # Division entière de _n par 2

		# Boucle principale
		for i in range(s):
			# Détermine un a aléatoire [2, n-1[
			a = rd.randrange(2, n-1)
			# Détermine 'z' initiale
			z = pow(a, _n, n)

			if (z != 1 and z != n-1): 
				# Boucle pour déterminer si 'n' est composé
				j = 1
				while j < u and z != n-1:
					# Nouvelle valeur de z
					z = pow(z, _n, n)
					if z == 1: return False
					# Augmente le compteur
					j += 1
				if z != n-1: return False
		return True
	
	@staticmethod
	def getMessageInfo(msg : str, n : int, k0 : int):
		"""getMessageInfo Méthode pour obtenir les informations du message qui sera encrypté
		
			Args:
				msg (str): Message qui sera encrypté
				n (int): Taille d'un block encrypté
			
			Returns:
				str : Information sur la longueur et si OAEP est utilisé
		"""
		# OBtient la taille du message
		length = len(msg)
		# Détermine la taille du restant
		reminder_length = length%n
		
		# Initialise le message retourné
		msg_info = ""
		# Détermine si le message sera paddé
		if reminder_length == 0:
			msg_info = "0"
		elif reminder_length <= n- k0:
			msg_info = "1"
		else:
			msg_info = "0"
		
		# Ajoute la longuer du message
		msg_info += bin(length)[2:] + '1'
		return msg_info
	
	#staticmethod
	def OAEP(msg : str, n : int, k0 : int) -> str:
		"""OAEP Méthode implémentant «l'Optimal Asymmetric Encryption Padding»
		
		Args:
			msg (str): Message à «padder»
			l (int)  : Longueur totale du message «paddé»
		
		Returns:
			str: Message «paddé»
		"""
		#k0 = self.k0
		#n = self.n
		k1 = n - k0 - len(msg)
		# Genere un padding
		padding = "0" * k1
		# Ajoute le padding
		m = msg + padding
		
		# Génère un nombre aléatoire
		r = bin(rd.randrange(0,pow(2, k0-1)))[2:]
		r = r.zfill(k0)
		
		# Hash 'r' à n-k0 bits
		hash_r = RSA.hash_function(r, n-k0)
		
		# X == m XOR hash_r
		X = ''.join(str(int(a) ^ int(b)) for a,b in zip(m, hash_r))
		# Hash 'X'
		hash_X = RSA.hash_function(X, k0)

		# Y == r XOR hash_X
		Y = ''.join(str(int(a) ^ int(b)) for a,b in zip(r, hash_X))
		
		return X + Y
	
	@staticmethod
	def OAEP_inv(XY : str, n : int, k0 : int) -> str:
		"""OAEP_inv Méthode inverse de «OAEP» et permet de retrouver 'm'
		
		Args:
			XY (str): Message «paddé»
		
		Returns:
			str : Meesage original
		"""
		#k0 = self.k0
		#n = self.n
		# Sépare le message en deux parties: X || Y
		X = XY[:n-k0]
		Y = XY[n-k0:]
		
		# Hash 'X'
		hash_X = RSA.hash_function(X, k0)#[2:].zfill(k0)
		# Retrouve 'r'
		r = ''.join(str(int(a)^int(b)) for a, b in zip(Y, hash_X))
		# Hash 'r'
		hash_r = RSA.hash_function(r, n-k0)#[2:].zfill(l-k0)
		# Retrouve 'msg + padding'
		m = ''.join(str(int(a)^int(b)) for a, b in zip(X, hash_r))
		
		return m
	
	@staticmethod
	def genKeys(l = None):
		"""genKeys Méthode pour générer les clés 'PK' et 'SK'
		
		Args: 
			l (int): Longueur de 'n'
			
		Returns:
			tuple : Clés PK et SK
		"""
		# Détermine la valeur de 'l'
		#if l is None: l = self.n
			
		# Obtient la valeur de 'e'
		e = 217#self.
		
		# Obtient les bonnes valeurs de q et p
		while True:
			p, q = e+1, e+1
			while (p-1)%e == 0: p = RSA.bigPrime(l//2)
			while (q-1)%e == 0: q = RSA.bigPrime(l//2+1)
			phi_n = (p-1) * (q-1)
			if RSA.EEA(e, phi_n)[2] == 1: break
		
		# Détermine la valeur de 'n' et 'phi_n'
		n = p * q
		
		# Détermine la valeur de 'd'
		d = RSA.Inverse(e, phi_n)
		
		# Forme les clés
		PK = (n, e)
		SK = (p, q, d)
		
		return PK, SK

	#@staticmethod
	def exp_CRT(C : str, SK : list):
		"""exp_CRT Méthode qui implémente le Chinese Reminder Theorem
		
		Args:
			C (str): Cipher à décrypter
			SK (list): Clé privée
		"""
		# Convertie 'C' en valeur numérique
		C_num = int(C, 2)
		
		# Obtient les valeurs p, q et d de la clé SK
		p, q, d = SK[0], SK[1], SK[2]
		# Obtient la taille
		N = p * q
		n = len(bin(N)[2:]) - 1
		
		dp, dq = d%(p-1), d%(q-1)
		kp, kq = RSA.Inverse(q, p), RSA.Inverse(p, q)
		xp, xq = C_num%p, C_num%q
		yp, yq = RSA.squareAndMultiply(xp, dp, p), RSA.squareAndMultiply(xq, dq, q)
		
		# Obtient le message
		msg = bin(((q*kp)*yp + (p*kq)*yq)%(p*q))[2:].zfill(n)
		return msg
		 
	#self, @staticmethod
	def encrypt(msg : str, PK : list, k0 : int):
		"""encrypt Méthode pour encrypter un message avec une clé publique
		
		Args:
			msg (str): Message à encrypter
			PK (list): Clé publique
			
		Returns:
			cipher (str): Message encrypté
		"""
		# Obtient 'n' et 'e' de la clé publique
		N, e = PK[0], PK[1]
		# Obtient la taille
		n = len(bin(N)[2:])
		_n = n-1
		
		# Initialise le cipher
		cipher = ""
		
		# Détermine le nombre de block à encrypter
		n_block = len(msg) // _n
		
		# Détermine le restant
		reminder_length = len(msg)%_n
		
		# Obtient les informations du messages
		msg_info = RSA.getMessageInfo(msg, _n, k0)
		msg_info_block = RSA.OAEP(msg_info, _n, k0)
		cipher = bin(RSA.squareAndMultiply(msg_info_block, e, N))[2:].zfill(n)
		
		block = ""
		# Encrypte le message par block de n-bits
		for i in range(n_block):
			block = msg[i*_n : (i+1)*_n]
			cipher += bin(RSA.squareAndMultiply(block, e, N))[2:].zfill(n)
		
		# Détermine de quelle manière le dernier block sera encrypté
		if ((msg_info[0] == '0') and (reminder_length != 0)):
			# Ajoute des 0 puisque 'msg' est grand
			pad = '0' * (_n - reminder_length)
			block = msg[n_block*_n:] + pad
			cipher += bin(RSA.squareAndMultiply(block, e, N))[2:].zfill(n)
		elif ((msg_info[0] == '1') and (reminder_length != 0)):
			block = RSA.OAEP(msg[-reminder_length], _n, k0)
			cipher += bin(RSA.squareAndMultiply(block, e, N))[2:].zfill(n)

		return cipher
	
	#@staticmethod
	def decrypt(cipher : str, SK : list, k0 : int):
		"""decrypt Méthode pour décrypter un cipher selon une clé privée
		
		Args:
			cipher (str): Meesage à décrypter
			SK (list): Clé privée
		
		Returns:
			msg (str): Message décrypté
		"""
		# Taille des blocks encryptés
		n = len(bin(SK[0] * SK[1])[2:])
		# Détermine le nombre de block à décrypter
		n_block = len(cipher) // n
		# Initialise le message
		msg = ""
		
		# Obtient le premier block (contient les infos sur le message)
		# Décrypte le block
		msg_info = RSA.exp_CRT(cipher[0:n], SK).zfill(n-1)
		msg_info = RSA.OAEP_inv(msg_info, n-1, k0)
		
		# Détermine si un padding a été fait
		padded = (msg_info[0] == "1")
		
		
		# Obtient la taille du message
		length_block = msg_info[1:]
		i = len(length_block)-1
		while length_block[i] == "0":
			length_block = length_block[0:i]
			i -= 1
			
		length = int(length_block[0:i], 2)
		
		# Détermine le restant
		reminder_length = length%n
		
		# Boucle pour décrypter les blocks
		for i in range(1, n_block-1):
			block = cipher[i*n: (i+1)*n]
			msg += RSA.exp_CRT(block, SK)
		
		block = RSA.exp_CRT(cipher[-n:], SK).zfill(n-1)
		# Cas ou le message a été paddé avec OAEP
		if padded:
			block = RSA.OAEP_inv(block, n-1, k0)
		
		msg += block
		return msg[0:length]
