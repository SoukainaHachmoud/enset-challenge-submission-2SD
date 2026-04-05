# Système de Surveillance Industrielle par IA

SecureWatch est une application de surveillance industrielle en temps réel basée sur la vision par ordinateur. Elle utilise YOLOv8 pour la détection de personnes et propose un tableau de bord interactif permettant de surveiller la sécurité, détecter les intrusions et analyser l’activité sur site.

## Présentation

Le système est conçu pour améliorer la sécurité dans les environnements industriels. Il combine des techniques avancées de détection en temps réel avec une interface moderne développée avec Streamlit. L’application permet de visualiser les flux, détecter la présence humaine et suivre les alertes ainsi que les indicateurs de performance.

## Fonctionnalités

- Détection en temps réel des têtes humaines à l’aide de YOLOv8  
- Tableau de bord interactif avec alertes et indicateurs clés  
- Analyse d’images importées et capture via webcam  
- Paramètres de détection ajustables (seuil de confiance, IOU, ratio de tête)  
- Suivi des incidents et état des caméras  
- Interface utilisateur moderne et responsive  

## Technologies utilisées

- Python  
- Streamlit  
- OpenCV  
- Ultralytics YOLOv8  
- NumPy  
- Pandas  
- Pillow  

## Structure du projet
modl.py Application principale Streamlit
README.md Documentation du projet
requirements.txt Dépendances du projet

## Installation
1. Cloner le dépôt :
git clone https://github.com/votre-utilisateur/votre-repository.git
cd votre-repository

2. Créer un environnement virtuel (optionnel mais recommandé) :
python -m venv venv
venv\Scripts\activate (Windows)
source venv/bin/activate (Linux/Mac)

3. Installer les dépendances :
pip install -r requirements.txt

## Utilisation

Lancer l’application avec Streamlit :
streamlit run modl.py

Puis ouvrir dans le navigateur :
http://localhost:8501

## Description du fonctionnement

L’application utilise YOLOv8 pour détecter les الأشخاص dans les images. Pour chaque personne détectée, une zone correspondant à la tête est estimée à partir d’un ratio configurable. Le système affiche ensuite des boîtes englobantes ainsi que des informations en temps réel telles que le nombre de détections et les niveaux de confiance.

L’interface est organisée en plusieurs sections :
- Accueil : présentation générale du système  
- Tableau de bord : surveillance en temps réel et alertes  
- Détection : analyse d’images et de captures webcam  
- Contact : informations sur le projet et l’équipe  

## Équipe

- Doha Zilaoui — Développement principal  
- Soukaina Hachmoud — Vision par ordinateur  
- Sara Fadil — Backend et système d’alertes  

## Perspectives d’amélioration

- Intégration de flux vidéo en temps réel (caméras RTSP)  
- Détection des équipements de protection individuelle  
- Détection de chutes via estimation de pose  
- Notifications en temps réel (SMS, email)  
- Déploiement sur des dispositifs embarqués  

## Licence

Ce projet est réalisé dans un cadre académique et de recherche.
