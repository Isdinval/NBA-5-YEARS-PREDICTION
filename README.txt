Le dossier "LIVRABLES" est composé de 3 dossiers : 
I. Collecte, nettoyage, exploration des données et entrainement d'un classifier
	I.1. Un dossier "image" est inclus contenant 1. analyse de correlation, 2. analyse des features, 3. reduction dimensionnelle et 4. distribution de la cible
	I.2. Un notebook Jupyter de collecte/nettoyage/exploration de données : EDA.ipynb
	I.3. Deux datasets nettoyés : un avec feature engineering (nba_logreg_clean_WITH_FE.csv) et l'autre sans (nba_logreg_clean_NO_FE.csv). 
	I.4. Un notebook Jupyter pour le training et le choix du meilleur classifier : STEP1_TRAINING_CLEAN_V2.ipynb
	
II. Creation d'une API via Flask en local
	II.1. Le script du serveur flask en local : app.py. 
	II.2. La documention pour réaliser des prédictions : tutoFlaskLocal.txt

III. Creation d'une API via Flask sur le cloud avec la plateforme PythonAnywhere
	III.1. Un dossier "INPUT", qui contient tous les élèments nécéssaires pour réaliser une prédiction : le seuil de décision, le nom des features, le scaler et enfin le modèle entrainé.
		- decision_threshold_WITH_FE.txt
		- feature_names_WITH_FE.joblib
		- nba_career_prediction_model_WITH_FE.joblib
		- scaler_WITH_FE.joblib
	III.2. Les différents codes nécéssaires au fonctionnement de l'API sur le cloud : 
		- flask_app_WITH_FE.py
		- index_WITH_FE.html
		- result.html
		- isdinval_pythonanywhere_com_wsgi.py.py
	III.3. Le lien vers la web app : https://isdinval.pythonanywhere.com/

Enfin, une note méthodologique expliquant mes choix est présente à la racine du dossier "LIVRABLES" : Note Méthodologique NBA.docx