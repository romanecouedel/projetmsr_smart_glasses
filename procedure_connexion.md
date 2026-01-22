se connecter à la wifi 'RPI_HOTSPOT'

ssh projetmsr@10.42.0.1
mdp = projet123

coté rpi 
python3 com_pc.py

coté ordi
pour verifier sans code :
pour enlever les parfeux : sudo ufw status 
                si actif : sudo ufw allow 5000/udp
lancer le code 