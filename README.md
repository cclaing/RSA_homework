# RSA_homework
Basic RSA implementation in python. In the context of a University homework.
It is in no way an official encryption program and recommend to no one the use
of it for standard encryption task as it is incomplete and of beginner level.

Authors : Camille Claing & Charly Jeffrey

The code is commented in french (Université de Montréal).


Methode implemented are :
  - Generation of big prime number by random drawing of large number fallowed by Miller Rabin test.
  - Square and Multiply for large exponantiations
  - EEA algorithm to find biggest common divider of two integers
  - Methode to find the inverse of Zn* groupe element
  - Elementary hash fonction by modulus of large prime multiplication
  
  - Standard OAEP padding & inverse fonction
  - Method to genarate pairs of Public key (n, e) & Secret Key (p, q, d)
  - Method of encryption by block of data by PK exponentiation/modulus (using OAEP if necessary)
  - Method of decryption by block of data by SK exponentiation/modulus (usinf OAEP_Inv if necessary)
  
  
