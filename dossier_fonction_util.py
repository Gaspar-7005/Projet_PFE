#fonction pour importer les données brutes 
import pandas as pd 
import requests
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from sklearn import metrics

def DownloadRawData():
    url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
   
    # Nom du fichier pour telecharger les données brutes
    zip_filename = "credit_card_data.zip"

    #telechargement du fichier zip
    response = requests.get(url)
    with open(zip_filename,'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall()
#fonction pour importer les données brutes 
def ReadRawData():
    data_file = "default of credit card clients.xls"
    #charger les données dans un fichier pandas
    data = pd.read_excel(data_file, header=1) # header=1 IGNORE LA PREMIERE LGNE D'ENTETE
    return data

#formatage des données 
def formattageRawData():
    #lire les données 
    raw_data = ReadRawData()
    raw_data.columns = raw_data.columns.str.lower().str.replace(" ", "_")
    months = ["sep", "aug", "jul", "jun",  "may", "apr"]
    variables = ["payment_status","bill_statement","previous_payment"]
    new_column_names = [x + "_" + y for x in variables for y in months]
    rename_dict = {x: y for x,y in zip(raw_data.loc[:,"pay_0":"pay_amt6"].columns,new_column_names)}
    raw_data.rename(columns=rename_dict,inplace=True)

    # mapper les nombres aux chaines de caractere
    gender_dic = {
        1:"Male",
        2:"Female"
    }
    education_dict = {
        0:"Others",
        1:"Graduate school",
        2:"University",
        3:"Hight shool",
        4:"others",
        5:"others",
        6:"others"
    }
    marital_status_dict = {
        0:"Others",
        1:"Married",
        2:"Single",
        3:"Others"
    }
    payment_status = {
        -2:"Unknown",
        -1:"payed duly",
        0:"Unknown",
        1:"payment delayed 1 month",
        2:"payment delayed 2 months",
        3:"payment delayed 3 months",
        4:"payment delayed 4 months",
        5:"payment delayed 5 months",
        6:"payment delayed 6 months",
        7:"payment delayed 7 months",
        8:"payment delayed 8 months",
        9:"payment delayed >=9 months"
    }
    #Application du script de mappage pour les données 
    raw_data["sex"] = raw_data["sex"].map(gender_dic)
    raw_data["education"] = raw_data["education"].map(education_dict)
    raw_data["marriage"] = raw_data["marriage"].map(marital_status_dict)

    #Convertir les colonnes 'sex', 'education', 'default_payment', et 'marriage' en varriable categoriel
    categorical_columns = ['sex','marriage', 'education', 'default_payment_next_month']
    raw_data[categorical_columns] = raw_data[categorical_columns].astype('category')

    #Convertir les colonnes payment_status en varialbles ordinales
    payment_order = list(payment_status.keys())
    payment_categories = pd.CategoricalDtype(categories=payment_order, ordered=True)
    payment_columns = ['payment_status_sep', 'payment_status_aug', 'payment_status_jul',
                       'payment_status_jun', 'payment_status_may', 'payment_status_apr']
    raw_data[payment_columns] = raw_data[payment_columns].astype(payment_categories)

    #Sauvegarder au format csv
    raw_data.to_csv("credit_card_default.csv", index=False)

    #retourner les données formatées
    return raw_data


def plot_distribution(df):
    #selectionner uniquement les colonnes numerique
    numeric_columns = df.select_dtypes(include='number').drop(columns=['id'])

    #Afficher un histogramme pour chaque colonne numerique
    plt.figure(figsize=(15,20))
    num_cols = len(numeric_columns.columns)

    # Créer une grille de sous graphiques au nombres de colonne numerique
    for i, column in enumerate(numeric_columns.columns):
        plt.subplot(5, 3, i + 1) #Adapté a 15 colonnes numeriques
        plt.hist(df[column], bins=20, color='blue', alpha=0.7)
        plt.title(f'Histogramme de {column}')
        plt.xlabel(column)
        plt.ylabel('frequence')

    plt.tight_layout()
    plt.show()
    plt.close()

    #Affficher une boite a moustache pour chaque 
    plt.figure(figure=(15, 20)) #Adapter  la taille de la figure en fonction du nombre de colonnes  
    num_cols = len(numeric_columns.columns)

    # Creer un grille de sous graphique pour les boites a moustaches 
    for i, column in enumerate(numeric_columns.columns):   
        plt.subplot(5, 3, i + 1) # adapté a 15 colonnes numeriques
        sns.set(style="whitegrid")
        sns.boxplot(x=df[column], palette="Set2")
        plt.title(f'Boite a Moustachede {column}')
        plt.xlabel(column)
    plt.tight_layout()
    plt.show()
    plt.close()

    # colonnes a analyser 
    columns_to_analyse = [
        'sex', 'education', 'marriage', 'default_payment_next_month',
        'payment_status_sep', 'payment_status_aug', 'payment_status_jul',
        'payment_status_jun', 'payment_status_may', 'payment_status_apr'
    ]
    # diagramme a barres
    for column in columns_to_analyse:
        print(f"Analyse univariable de la colonnes'{column}':\n")

        # compter les occcurences de chaque categorie
        value_counts = df[column].value_counts(normalize=True)
        print(f"frequence des categories :\n{value_counts}\n")
        # Afficher un graphique a barre pour visualiser la distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=column, palette='Set1')
        plt.title(f'Distribution de {column}')
        plt.xlabel({column})
        plt.ylabel('frequence')
        plt.show()
        plt.close()

        #statistique descriptive
        print(f"statistiques descriptive pour {column}:\n") 
        print(df[column].describe())

        print(f"\n" + "="*50 + "\n")


# Afficher les distribution en discretisant suivant une variable categorielle
def plot_discretize_distributions(df, cat_var):
    #créer des boites a moustaches pour chaque colonnes numerique en les segmentant par sexe
    plt.figure(figsize=(15,20))

    # selectionner uniquement les colonnes numeriques
    numeric_columns = df.select_dtypes(include='number').drop(columns=['id'])

    #Créer une grille de sous graphique pour les boites a moustaches 
    for i, column in enumerate(numeric_columns.columns):
        plt.subplot(5, 3, i + 1) # adapté a 15 colonnes numeriques
        sns.set(style="whitegrid")
        sns.boxplot(data=df, x=column, y=cat_var, palette="Set2")
        plt.title(column  + 'par' + cat_var)
        plt.xlabel(column)
        plt.ylabel(cat_var)

    plt.tight_layout()
    plt.show()
    plt.close()

    #Colonnes a analyser par sexe
    columns_to_analyze = [
        'sex', 'education', 'marriage', 'default_payment_next_month',
        'payment_status_sep', 'payment_status_aug', 'payment_status_jul',
        'payment_status_jun', 'payment_status_may', 'payment_status_apr'
    ]
    columns_to_analyze.remove(cat_var)

    # Créer des graphiques a barres pour chaque colonne
    for column in columns_to_analyze:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=column, hue=cat_var, palette='Set1')

        # personnalisation du graphique
        plt.title(column + 'par' + cat_var)
        plt.xlabel(column)
        plt.ylabel('Frequence')
        plt.xticks(rotation=45) # faire pivoter les etiquette de l'axe des x pour plus de lisibilité

        #Affciher les graphiques 
        plt.legend(title=cat_var)
        plt.show()
        plt.close()



#Matrice de correlation 
def plot_correlation_matrice(corr_mat):
    sns.set(style="white")
    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask)]= True
    fig, ax = plt.subplots(figsize=(12,10))
    cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)
    sns.heatmap(
        corr_mat, mask=mask, cmap=cmap, annot=True,
        fmt=".2f", vmin=-1, vmax=1, center=0, square=True,
        linewidths=.5, cbar_kws={"shrink": .5},ax=ax
    )
    ax.set_title("matrice de correlation", fontsize=16)
    sns.set(style="darkgrid")

def pct_default_by_category(df, cat_var):
    #pourcentage de default de paiement par categorie 
    ax = df.groupby(cat_var)["default_payment_next_month"] \
    .value_counts(normalize=True)  \
    .unstack()  \
    .plot(kind="barh", stacked="True")
    ax.set_title("poucentage de defaults de paiement", fontsize=16)
    ax.legend(title="Defaut_de_paiement",bbox_to_anchor=(1,1))
    plt.show()

def performance_evaluation_report(model, X_test, y_test, show_plot=False, labels=None, show_pr_curve=False):
    """
    function for creating a performance report of a classification model.
    parameters
    -------------
    model : scikit-learn estimator 
       A fited estimator for classification problems.
       x_test : pd.dataFrame : 
           DataFrame with features matching y_test 
        y_test : array /pd.Series
            Target_of a classification problem.
        show_plot : bool
            Flag_whether to also show PR-curve, for to take affect,
            show_plot must be True 
        Return 
        _ _ _
        stats : ps.series 
            A series xith the most important evaluation metrics  

    """

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    cm = metrics.confusion_matrix(y_test, y_pred)
    tn, fp, tp, fn = cm.ravel()

    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_prob)
    pr_auc = metrics.auc(recall, precision)
    if show_plot:

        if labels is None:
            labels = ["Negative","Positive"]
            N_SUBPLOTS = 3 if show_pr_curve else 2
            PLOT_WIDTH = 20 if show_pr_curve else 12
            PLOT_HIGHT = 5 if show_pr_curve else 6

            fig, ax  = plt.subplots(
                1, N_SUBPLOTS, figsize=(PLOT_WIDTH, PLOT_HIGHT))
            fig.suptitle("Evaluation de la performance du modele", fonction=16)
            #plot 1 : confusion matrix ----

            # preparing more descriptive labels for confusion matrix
            cm_counts = [f"{val:0.0f}" for val in cm.flatten()]
            cm_percentages = [f"{val:.2%}" for val in cm.flatten()/np.sum(cm)]
            cm_labels = [f"{v1}\n{v2}" for v1, v2 in zip(cm_counts, cm_percentages)]
            cm_labels = np.asarray(cm_labels).reshape(2,2)


            sns.heatmap(cm, annot=cm_labels, fmt="", linewidths=.5, cmap="Greens",
                       square=True, cbar=False, ax = ax[0],
                       annot_kws={"ha":"center","va":"center"})
            ax[0].set(xlabel="predicted label",
                      ylabel="Actual label", title="confusion Metrix")
            ax[0].xaxis.set_tick_labels(labels)
            ax[0].yaxis.set_tick_labels(labels)


            # Plot  2 
            metrics.RoeCorveDisplay.from_estimator(model, X_test, y_test, ax=ax[1], name="")
            ax[1].set_title("ROC Curve")
            ax[1].plot(fp/(fp+tn), tp/(tp+fn), "ro",
                       markersize=8, label="Decision point")
            ax[1].plot([0, 1], [0, 1], "r--")

            #partie alternative au cas ou 


            #
            if show_pr_curve:
                metrics.PrecisionRecallDispaly.from_estimator(model, X_test, y_test, ax=ax[2], name="")
                ax[2].set_title("precision du model curve")

                #alterative a faire plus tard

    stats = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred),
        "recall": metrics.recall_score(y_test, y_pred),
        "specificity": (tn/(tn + fp)),
        "cohens_kappa": metrics.cohen_kappa_score(y_test, y_pred),
        "f1_score": metrics.f1_score(y_test, y_pred),
        "matthews_corr_coeff": metrics.matthews_corrcoef(y_test, y_pred),
        "roc_auc":roc_auc,
        "pr_auc":pr_auc,
        "average_precision": metrics.average_precision_score(y_test, y_pred_prob)

        }    
    return stats



            

            


    






