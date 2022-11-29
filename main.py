import random
import pandas as pd
import numpy as np
from time import localtime, strftime, time
import joblib
import itertools

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FuncFormatter, EngFormatter
from imblearn.over_sampling import SMOTE, RandomOverSampler
import seaborn as sns

from scipy.stats import chi2_contingency
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, \
    roc_auc_score, roc_curve, f1_score

train_mode = True
fprs, tprs = [], []

pd.set_option('display.max_columns', None)
pd.set_option('max_columns', None)


def print_time_line_sep(msg):
    """
    :param msg: massage to print
    print len sep with current date time.
    """
    dt_fmt = "%d/%m/%Y %H:%M:%S"
    line_msg = " " * 41 + msg
    line_sep = "-" * 35 + " " + strftime(dt_fmt, localtime()) + " " + "-" * 35 + '\n'
    print(line_msg)
    print(line_sep)


def read_data():
    data = 'data.csv'
    data_frame = pd.read_csv(data)
    return data_frame


def missing_values():
    """ Missing values """
    print_time_line_sep("Missing values")
    print(df.isnull().sum())
    print(df.isnull().mean().round(4))  # It returns percentage of missing values in each column in the dataframe.

    # assert that there are no missing values in the dataframe
    # assert pd.notnull(df).all().all(), "missing values exits!"


def fix_value_spaces_and_names(df):
    df = df.replace(to_replace=[" <=50K.", " <=50K"], value="<=50K")
    df = df.replace(to_replace=[" >50K", " >50K."], value=">50K")
    df = df.replace(to_replace=[" ?", "?"], value=np.nan)
    income_map = {'<=50K': 0, '>50K': 1}
    df['income'] = df['income'].map(income_map)
    print(df.columns)
    cols = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    df[cols] = df[cols].apply(lambda x: x.str.strip())
    # df[cols].apply(lambda x: print(pd.unique(df['workclass']).tolist()))
    # print(df.isin([' ? ']).any())
    # column_names = df.columns.tolist()
    # print(type(df['income'].iloc[0:1]))
    # for i in range(len(column_names)):
    #     print("%%%%", i, type(df[column_names[i]].iloc[0:1].dtypes))
    #     df[df.columns] = df.apply(lambda x: x.str.strip())
    #     # if type(df[column_names[i]].iloc[0:1]) is object:
    #     #     print(i, ": ", df[column_names[i]])
    #     #     df[column_names[i]] = df[column_names[i]].strip()
    #     # else:
    #     #     print(i, df[column_names[i]].iloc[0:1])

    return df


def describe_df(df):
    print_time_line_sep("describe data")
    print(df.describe(include=['object']).T)
    print(df.describe().T)


def corr_matrices(df):
    corr = df.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Spearman", fontsize=22, fontweight="bold")
    plt.subplots_adjust(bottom=.26)
    plt.show()
    # corr = df.corr(method="pearson")
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.title("Pearson")
    # plt.show()


def check_columns():
    column_names = df.columns.tolist()
    for i in range(len(column_names)):
        print(column_names[i])
        print(df[column_names[i]].isnull().sum())
        print(df[column_names[i]].nunique())
        print(df[column_names[i]].value_counts())
        if type(df[column_names[i]]) is str:
            print(df[column_names[i]].str.startswith(' '), '\n')


def income_general_distribution():
    print_time_line_sep("Descriptive Statistics")
    # # visualize frequency distribution of income variable
    f = plt.subplots(1, 1, figsize=(10, 8))
    # ax[0] = df['income'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
    df['income'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['tomato', 'navajowhite'],
                                     startangle=90, shadow=False, legend=True, fontsize=19, labels=["<=50K$", ">50K$"])
    plt.title('Income distribution', fontsize=22, fontweight="bold")
    plt.legend(fontsize='x-large')
    plt.ylabel('', fontsize=20)
    plt.show()


def age_distribution():
    g = sns.displot(df, x='age', hue='income', bins=20, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    g.ax.set_title('Age distribution', fontsize=14, fontweight="bold")
    plt.show()


def workclass_distribution():
    df_1 = df.replace(to_replace=[np.nan], value=None)
    workclass_names = df_1['workclass'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df_1, x='workclass', hue='income', legend='full', multiple="stack", shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(8), xlabels_new)
    plt.tight_layout()
    plt.title('Work class distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def education_distribution():
    workclass_names = df['education'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df, x='education', hue='income', legend='full', multiple="stack", shrink=.9)
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(16), xlabels_new, rotation=60)
    plt.tight_layout()
    plt.title('Education distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def education_num_distribution():
    education = df['education'].unique().tolist()
    education_num = df['education.num'].unique().tolist()
    education_sorted = [x for _, x in sorted(zip(education_num, education))]
    xlabels_new = [label.replace('-', '-\n') for label in education_sorted]
    xlabels_new = [""] + xlabels_new
    plt.xticks(np.arange(0, 17, 1), xlabels_new, rotation=90)
    ax = sns.histplot(data=df, x='education.num', hue='income', legend='full', multiple="stack")
    plt.tight_layout()
    plt.title('Education num distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def marital_status_distribution():
    workclass_names = df['marital.status'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df, x='marital.status', shrink=.9, hue='income', legend='full', multiple="stack")
    # data = df, x = 'education.num'
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(7), xlabels_new, rotation=0)
    plt.tight_layout()
    plt.title('Marital status distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def relationship_distribution():
    workclass_names = df['relationship'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df, x='relationship', shrink=.9, hue='income', legend='full', multiple="stack")
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(6), xlabels_new, rotation=0)
    plt.tight_layout()
    plt.title('Relationship distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def occupation_distribution():
    df_1 = df.replace(to_replace=[np.nan], value=None)
    workclass_names = df_1['occupation'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df_1, x='occupation', hue='income', legend='full', multiple="stack")
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(14), xlabels_new, rotation=90)
    plt.tight_layout()
    plt.title('Occupation distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def race_distribution():
    workclass_names = df['race'].unique().tolist()
    print(workclass_names)
    ax = sns.histplot(data=df, x='race', shrink=.9, hue='income', legend='full', multiple="stack")
    xlabels = workclass_names
    xlabels_new = [label.replace('-', '-\n') for label in xlabels]
    plt.xticks(range(5), xlabels_new, rotation=0)
    plt.tight_layout()
    plt.title('Race distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def sex_distribution():
    ax = sns.histplot(data=df, x='income', shrink=.9, hue='sex', legend='full', multiple="stack")
    plt.tight_layout()
    plt.title('Gender distribution', fontsize=22, fontweight="bold")
    plt.subplots_adjust(top=0.90)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(12)
    plt.show()


def hours_per_week_distribution():
    g = sns.displot(df, x='hours.per.week', hue='income', bins=20, legend='full', multiple="stack")
    # g = sns.displot(df, x=df['hours.per.week'], bins=20)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Hours per week distribution', fontsize=14, fontweight="bold")
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def fnlwgt_distribution():
    g = sns.displot(df, x='fnlwgt', hue='income', bins=20, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('fnlwgt distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    plt.xticks(np.arange(0, 1_200_000, 250_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def capital_gain_distribution():
    g = sns.displot(df, x='capital.gain', hue='income', bins=10, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Capital gain distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    plt.xticks(np.arange(0, 100_000, 20_000))
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def capital_loss_distribution():
    g = sns.displot(df, x='capital.loss', hue='income', bins=10, legend='full', multiple="stack")
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title('Capital loss distribution', fontsize=14, fontweight="bold")
    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    g.ax.xaxis.set_major_formatter(formatter1)
    g.ax.xaxis.label.set_size(20)
    g.ax.yaxis.label.set_size(12)
    plt.show()


def native_country_distribution():
    def autopct(pct):  # only show the label when it's > 10%
        return ('%.2f' % pct) if pct > 10 else ''

    my_labels = df['native.country'].unique().tolist()
    ax = df['native.country'].value_counts().plot(kind='pie', figsize=(28, 12), autopct=autopct)
    ax.axes.get_yaxis().set_visible(False)
    plt.show()


def plot_descriptive_statistics():
    age_distribution()
    # workclass_distribution()
    fnlwgt_distribution()
    # education_distribution()
    # education_num_distribution()
    # occupation_distribution()
    # relationship_distribution()
    # marital_status_distribution()
    race_distribution()
    sex_distribution()
    # capital_gain_distribution()
    # capital_loss_distribution()
    hours_per_week_distribution()
    # native_country_distribution()
    income_general_distribution()


def distribution_workclass_income(df):
    class_order = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov',
                   'Without-pay', 'Never-worked']
    hue_order = ['<=50K', '>50K']
    f, ax = plt.subplots(figsize=(12, 8))
    ax = sns.countplot(x="workclass", hue="income", data=df, palette="rocket", order=class_order, hue_order=hue_order)
    ax.set_title("Frequency distribution of workclass vs income", fontsize=25, fontweight="bold")
    ax.legend(loc='upper right')
    counts = df.workclass.value_counts()
    i = 0
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(12)
    plt.legend(fontsize=20, loc='upper right')
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / counts[i])
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center')
        i += 1
        if i == 8:
            i = 0
    plt.show()


def box_plots(df):
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.suptitle('Box-plots', fontweight="bold")
    df.boxplot(column=['age'], ax=axs[0, 0], widths=(0.5), color='darkred')
    df.boxplot(column=['fnlwgt'], ax=axs[0, 1], widths=(0.5), color='darkred')
    df.boxplot(column=['education.num'], ax=axs[0, 2], widths=(0.5), color='darkred')
    df.boxplot(column=['capital.gain'], ax=axs[1, 0], widths=(0.5), color='darkred')
    df.boxplot(column=['capital.loss'], ax=axs[1, 1], widths=(0.5), color='darkred')
    df.boxplot(column=['hours.per.week'], ax=axs[1, 2], widths=(0.5), color='darkred')
    plt.show()


def box_plot3(df):
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="income", y="age", hue="sex", data=df, palette="rocket")
    ax.set_title("Box plot: Income vs age and sex", fontsize=22, fontweight="bold")
    ax.legend(loc='upper right')
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(15)
    plt.legend(fontsize=17, loc='upper right')
    plt.show()


def income_plot(df):
    pair = sns.pairplot(df, hue="income", palette="rocket")
    plt.show()


def income_plot2(df):
    ## Income by age and hours per week
    ax = sns.scatterplot(x="age", y="hours.per.week", hue="income",
                         data=df, palette='rocket')
    ax.set_title("Income vs age and hours.per.week", fontsize=15, fontweight="bold")
    ax.legend(loc='upper right')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(12)
    plt.legend(fontsize=11, loc='upper right')
    plt.show()


def income_plot3(df):
    ## Income By age and race
    ax = sns.boxplot(x="income", y="age", hue="race",
                     data=df, palette="rocket")
    ax.set_title("Income vs age and race", fontsize=15, fontweight="bold")
    ax.legend(loc='upper right')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(12)
    plt.yticks(np.arange(10, 130, 10))
    plt.legend(fontsize=8, loc='upper right')
    plt.show()


def handle_mis_val(df):
    print("Number of Missing values of 3 variables:")
    print(df.loc[df['native.country'].isnull()].count())
    print(df.loc[df['occupation'].isnull() & (df['workclass'].isnull())].count())
    print("Number of Missing values of workclass and native.country:")
    print(df.loc[df['occupation'].isnull() & (df['workclass'].isnull())].count())
    """ Dropping records:"""
    # df = df.drop(df[(df['occupation'].isnull() ) & (df['workclass'].isnull()) & df['native.country'].isnull()].index)
    # print('The shape of the dataset : ', df.shape)
    df = df.drop(df[(df['occupation'].isnull()) & (df['workclass'].isnull())].index)

    df['occupation'].fillna(df['occupation'].mode()[0], inplace=True)
    print('The shape of the dataset : ', df.shape)
    print("Number of Missing values of 1ariables:")
    print(df.loc[df['occupation'].isnull() | df['workclass'].isnull() | df['native.country'].isnull()].count())
    df['native.country'].fillna(df['native.country'].mode()[0], inplace=True)
    print(df.loc[df['native.country'].isnull()].count())
    return df


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, size=18)
    plt.yticks(tick_marks, classes, size=18)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel('True label', size=18)
        plt.xlabel('Predicted label', size=18)
    plt.show()


def results(grid_search_model, X_train_encoded, X_test_encoded, y_test, estimator, y_train):
    test_predictions = grid_search_model.predict(X_test_encoded)
    cm = confusion_matrix(y_test, test_predictions)
    plot_confusion_matrix(cm, classes=['0 - <=50K', '1 - >50K'],
                          title='income Confusion Matrix')
    if estimator == 'svm':
        test_probs = grid_search_model.decision_function(X_test_encoded)
        train_probs = grid_search_model.decision_function(X_train_encoded)
    else:
        test_probs = grid_search_model.predict_proba(X_test_encoded)[:, 1]
        train_probs = grid_search_model.predict_proba(X_train_encoded)[:, 1]

    train_predictions = grid_search_model.predict(X_train_encoded)
    evaluate_model(test_predictions, test_probs, train_predictions, train_probs, estimator, y_test, y_train,
                   grid_search_model)


def plot_feature_importance(X_train_en, grid_search_object, orig_name):
    feature_importances = list(zip(X_train_en, grid_search_object.best_estimator_.feature_importances_))
    feature_importances_ranked = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # [print('Feature: {:35} Importance: {}'.format(*pair)) for pair in feature_importances_ranked]

    df_names = pd.DataFrame(feature_importances, columns=['Name', 'score'])  # 63
    column_names = orig_name  # 12
    df_temp = pd.DataFrame(columns=column_names)  # 12
    df_temp.loc[len(df)] = 0
    for i in range(df_names.shape[0]):
        name = df_names.iloc[[i]].values[0][0]
        for idx, j in enumerate(column_names):
            if j in name:
                df_temp.iloc[0, df_temp.columns.get_loc(j)] += df_names.iloc[[i]].values[0][1]
                # print("value to add: ", df_names.iloc[[i]].values[0][1])
                break
    # todo: make it nicer
    new_temp_df = df_temp.sort_values(df_temp.last_valid_index(), axis=1, ascending=[False])
    # new_temp_df["sum"] = new_temp_df.sum(axis=1) # check it sum's to 1
    # print(new_temp_df)
    # change order of columns, or plot horizontally
    data = new_temp_df.iloc[0].to_dict()
    names = list(data.keys())
    values = list(data.values())
    y_pos = np.arange(len(names))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, names, rotation=90)
    plt.ylabel('Importance')
    plt.title("Feature importance", fontsize=15, fontweight="bold")
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.45)
    plt.show()


def rf_pipe(orig_names_list):
    print_time_line_sep('Random Forest')
    checking = True
    # train_mode = True
    random_grid = {}
    n_estimators = [int(x) for x in np.linspace(start=50, stop=450, num=3)]
    max_depth = [10, 30, 50, None]  # Maximum number of levels in tree
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9]  # Minimum number of samples required to split a node
    bootstrap = [True]  # Method of selecting samples for training each tree
    if checking:
        random_grid = {'n_estimators': [100],
                       'min_samples_split': [2]}
    else:
        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'bootstrap': bootstrap,
                       }
    # Create base model to tune
    rf = RandomForestClassifier(oob_score=True)

    # Create random search model and fit the data
    rf_random = GridSearchCV(estimator=rf, param_grid=random_grid, n_jobs=-1, cv=5, verbose=2, scoring='f1')
    # rf.fit(X_train_encoded, y_train)
    if train_mode:
        rf_random.fit(X_train, y_train)
        joblib.dump(rf_random, 'RF_final2.pkl')  # save your model or results
    else:
        rf_random = joblib.load("RF_final2.pkl")  # load your model for further usage
    results(rf_random, X_train, X_test, y_test, 'RF', y_train)
    # print("feature importance")
    plot_feature_importance(X_train_en=X_train, grid_search_object=rf_random, orig_name=orig_names_list)
    # plot_hyper_parameter(grid_search=rf_random, hyper_param_to_plot='n_estimators')
    # plot_hyper_parameter(grid_search=rf_random, hyper_param_to_plot='max_depth')
    # plot_hyper_parameter(grid_search=rf_random, hyper_param_to_plot='min_samples_split')


def XGBoost_pipe():
    print_time_line_sep('XGBoost')
    checking = False
    train_mode = False
    random_grid = {}
    max_depth = [1, 2, 3]  # Maximum number of levels in tree
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 1]
    if checking:
        random_grid = {'n_estimators': [100],
                       'learning_rate': [0.1]}
    else:
        random_grid = {'max_depth': max_depth,
                       'learning_rate': learning_rate}
    # Create base model to tune
    XG = GradientBoostingClassifier()
    # Create random search model and fit the data
    XG_grid_search = GridSearchCV(
        estimator=XG,
        param_grid=random_grid,
        n_jobs=-1, cv=5,
        verbose=2,
        scoring='f1')
    if train_mode:
        XG_grid_search.fit(X_train, y_train)
        joblib.dump(XG_grid_search, 'XG_final2.pkl')  # save your model or results
    else:
        XG_grid_search = joblib.load("XG_final2.pkl")  # load your model for further usage

    results(XG_grid_search, X_train, X_test, y_test, 'XG', y_train)
    plot_hyper_parameter(grid_search=XG_grid_search, hyper_param_to_plot='max_depth')
    plot_hyper_parameter(grid_search=XG_grid_search, hyper_param_to_plot='learning_rate')


def NN_pipe():
    print_time_line_sep('NN')
    checking = False
    train_mode = False
    random_grid = {}
    if checking:
        param_grid = {'max_iter': [200],
                      'activation': ['relu']}
    else:
        param_grid = {'solver': ['adam'],
                      'max_iter': [200, 500, 1000],
                      'alpha': [0.00001], 'hidden_layer_sizes': [5, 25, 45, 65, 85, 105]}
    # Create base model to tune
    NN = MLPClassifier()

    # Create random search model and fit the data
    NN_grid_search = GridSearchCV(estimator=NN, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2, scoring='f1')
    if train_mode:
        NN_grid_search.fit(X_train, y_train)
        joblib.dump(NN_grid_search, 'NN_final2.pkl')  # save your model or results
    else:
        NN_grid_search = joblib.load("NN_final2.pkl")  # load your model for further usage
    results(NN_grid_search, X_train, X_test, y_test, 'NN', y_train)
    plot_hyper_parameter(grid_search=NN_grid_search, hyper_param_to_plot='max_iter')
    plot_hyper_parameter(grid_search=NN_grid_search, hyper_param_to_plot='hidden_layer_sizes')


def svm_pipe(X_train, X_test):
    print_time_line_sep('SVM')
    checking = False
    train_mode = False
    random_grid = {}
    alpha = [0.0001, 0.001, 0.01, 0.1]
    max_iter = [1_000, 4_000, 7_000, 10_000]
    if checking:
        random_grid = {'alpha': [0.0001],
                       'max_iter': [1_000]}
    else:
        random_grid = {'alpha': alpha,
                       'max_iter': max_iter
                       }
    # Create base model to tune
    SVM = SGDClassifier()
    # Create random search model and fit the data
    SVM_grid_search = GridSearchCV(
        estimator=SVM,
        param_grid=random_grid,
        n_jobs=2, cv=5,
        verbose=2,
        scoring='f1')

    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    if train_mode:
        SVM_grid_search.fit(X_train, y_train)
        joblib.dump(SVM_grid_search, 'SVM_final2.pkl')  # save your model or results
    else:
        SVM_grid_search = joblib.load("SVM_final2.pkl")  # load your model for further usage
    results(SVM_grid_search, X_train, X_test, y_test, "svm", y_train)
    plot_hyper_parameter(grid_search=SVM_grid_search, hyper_param_to_plot='alpha')
    plot_hyper_parameter(grid_search=SVM_grid_search, hyper_param_to_plot='max_iter')


def stats_to_lists(stats_to_list):
    a = [list(col) for col in zip(*[d.values() for d in stats_to_list])]
    return a[0], a[1], a[2]


def plot_stats(stats_dict):
    """
    plot bar chart of roc, F1, recall, precision
    """
    recalls, precisions, f1_scores = stats_to_lists(stats_dict)
    labels = ['RF', 'XGBoost', 'NN', 'SVM']
    x = np.arange(len(labels))  # the label locations
    width = 0.29  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, recalls, width, label='Recall')
    rects2 = ax.bar(x, precisions, width, label='Precision')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1_score')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('perfomnces')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=3, facecolor="white")

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = round(rect.get_height(), 2)
            ax.text(rect.get_x(), height,
                    height, va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.show()


def evaluate_model(y_pred, probs, train_predictions, train_probs, estimator, y_test, y_train, grid_obj):
    results, train_results = {}, {}
    results['recall'] = recall_score(y_test, y_pred)
    results['precision'] = precision_score(y_test, y_pred)
    results['f1_score'] = f1_score(y_test, y_pred)
    results['accuracy'] = accuracy_score(y_test, y_pred)
    results['roc'] = roc_auc_score(y_test, probs)
    train_results['recall'] = recall_score(y_train, train_predictions)
    train_results['precision'] = precision_score(y_train, train_predictions)
    train_results['f1_score'] = f1_score(y_train, train_predictions)
    train_results['accuracy'] = accuracy_score(y_train, train_predictions)
    train_results['roc'] = roc_auc_score(y_train, train_probs)
    stats.append(results)
    print(grid_obj.best_params_)
    for metric in ['recall', 'precision', 'f1_score', 'accuracy', 'roc']:
        print(f'{metric.capitalize()} '
              # f'Baseline: {round(baseline[metric], 2)} '
              f'Test: {round(results[metric], 2)} '
              f'Train: {round(train_results[metric], 2)} ')
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)
    rocs.append([base_fpr, base_tpr, model_fpr, model_tpr])
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


def plot_hyper_parameter(grid_search, hyper_param_to_plot):
    """plot the error as a function of a specific hyper parameter while setting the others constant
    the other hyper parameters are taken from the best configuration found in the grid search.

    Parameters
    ----------
    grid_search: GridSearchCV.
        fitted GridSearchCV object
    hyper_param_to_plot: str.
        name of the hyper parameter to plot. one of grid_search.best_params_ keys.

    """
    best_params_to_keep_constant = {
        f"param_{param}": value
        for param, value in grid_search.best_params_.items()
        if param != hyper_param_to_plot
    }

    hyper_parameter_values, err_score = get_scores_for_plot(
        best_params_to_keep_constant,
        grid_search.cv_results_,
        f"param_{hyper_param_to_plot}",
    )
    plt.plot(hyper_parameter_values, err_score)
    plt.xlabel(hyper_param_to_plot)
    plt.ylabel("Validation Error")
    ymin = min(err_score)
    xpos = err_score.index(ymin)
    xmin = hyper_parameter_values[xpos]
    try:
        xmin = round(xmin, 3)
    except TypeError:
        pass  # some hyper parameters are tuples
    plt.title(
        f"{hyper_param_to_plot} validation error \n "
        f"Local minimum at {xmin} with \n "
        f"validation error {round(ymin, 3)}",
        fontsize=14, fontweight="bold"
    )
    plt.subplots_adjust(bottom=.24)
    plt.subplots_adjust(left=.2)
    plt.subplots_adjust(right=.8)
    plt.subplots_adjust(top=.84)
    plt.show()


def get_scores_for_plot(
        hyper_params_to_keep_constant, cv_results, hyper_parameter_to_plot_name
):
    """return the scores of a hyper parameter in a grid-search by setting other parameters constant
    Parameters
    ----------
    hyper_params_to_keep_constant: dict.
        dict containing the other hyper parameter names and values to look for in the grid search results
    cv_results: dict.
        dict holding the cv results from the grid search
    hyper_parameter_to_plot_name: str.
        name of the hyper parameter we are tuning

    Returns
    -------
    hyper_parameter_values: array.
        array of the hyper parameter values tested in the grid search
    err_score: array.
        array holding the corresponding error scores for each hyper parameter value

    See Also
    -------
    plot_hyper_parameter
    """
    n_candidates = cv_results["mean_fit_time"].size
    hyper_parameter_values = []
    err_score = []
    for candidate in range(n_candidates):
        if all(  # filters only the relevant results
                [
                    hyper_params_to_keep_constant[key] == cv_results[key][candidate]
                    for key in hyper_params_to_keep_constant.keys()
                ]
        ):
            hyper_parameter_values.append(
                cv_results[hyper_parameter_to_plot_name][candidate]
            )
            err_score.append(1 - cv_results["mean_test_score"][candidate])
    return hyper_parameter_values, err_score


def plot_roc_curve(rocs):
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16
    # Plot both curves
    plt.plot(rocs[0][0], rocs[0][1], 'b', label='baseline')
    plt.plot(rocs[0][2], rocs[0][3], 'r', label='RF')
    plt.plot(rocs[1][2], rocs[1][3], 'g', label='XG')
    plt.plot(rocs[2][2], rocs[2][3], 'y', label='NN')
    plt.plot(rocs[3][2], rocs[3][3], 'r', label='SVM')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.show()


def knn(df):
    # insert after pd.get_dummis
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print("_______________________________________")
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df_columns = df.columns.tolist()
    df = pd.DataFrame(scaler.inverse_transform(df))
    return df


def age_bins(df):
    df['age_bins'] = None
    df.loc[df.age <= 28, "age_bins"] = 1
    df.loc[(df.age > 28) & (df.age <= 38), "age_bins"] = 2
    df.loc[(df.age > 38) & (df.age <= 46), "age_bins"] = 3
    df.loc[(df.age > 46) & (df.age <= 56), "age_bins"] = 4
    df.loc[df.age > 56, "age_bins"] = 5
    print(df[['age', 'age_bins']])
    return df


def relationship_bins(df):
    df['relationship_bins'] = None
    df.loc[df.relationship == "Husband", "relationship_bins"] = "married"
    df.loc[df.relationship == "Wife", "relationship_bins"] = "married"
    df.loc[(df.relationship_bins != "married"), "relationship_bins"] = "else"
    return df



def hours_per_week_bins(df):
    df['hours.per.week_bins'] = None
    df.loc[df['hours.per.week'] <= 35, "hours.per.week_bins"] = 1
    df.loc[(df['hours.per.week'] > 35) & (df['hours.per.week'] <= 50), "hours.per.week_bins"] = 2
    df.loc[(df['hours.per.week'] > 50) & (df['hours.per.week'] <= 65), "hours.per.week_bins"] = 3
    df.loc[df['hours.per.week'] > 65, "hours.per.week_bins"] = 4
    print(df[['hours.per.week', 'hours.per.week_bins']])
    return df


def education_bins(df):
    df['education.num_bins'] = df['education.num']
    df.loc[df['education.num'] <= 8, "education.num_bins"] = 1  # bin for all school categories
    df.loc[df['education.num'] == 9, "education.num_bins"] = 2
    df.loc[df['education.num'] == 10, "education.num_bins"] = 3
    df.loc[df['education.num'] == 11, "education.num_bins"] = 4
    df.loc[df['education.num'] == 12, "education.num_bins"] = 5
    df.loc[df['education.num'] == 13, "education.num_bins"] = 6
    df.loc[df['education.num'] == 14, "education.num_bins"] = 7
    df.loc[df['education.num'] == 15, "education.num_bins"] = 8
    df.loc[df['education.num'] == 16, "education.num_bins"] = 9
    return df

def remove_variables(df):
    df = df.drop('hours.per.week', axis=1)
    df = df.drop('education.num', axis=1)
    df = df.drop('age', axis=1)
    df = df.drop('fnlwgt', axis=1)
    df = df.drop('relationship', axis=1)
    return df


def bins(df):
    # ----- bin for age
    df = age_bins(df)
    # ----- bin for fnlwgt
    # print(pd.qcut(df['fnlwgt'], q=5))  # for the bins range
    df['fnlwgt_bins'] = pd.qcut(df['fnlwgt'], q=5, labels=[1, 2, 3, 4, 5])  # equal distribution bins
    # ----- bin for education.num
    df = education_bins(df)
    # ----- bin for hours.per.week
    df = hours_per_week_bins(df)
    df = relationship_bins(df)
    df = remove_variables(df)

    df = df.rename({'age_bins': 'age', 'fnlwgt_bins': 'fnlwgt',
                    'education.num_bins': 'education.num', 'hours.per.week_bins': 'hours.per.week',
                    'relationship_bins': 'relationship'}, axis=1)
    return df


def remove_add_variables(df):
    df = df.drop('native.country', axis=1)  # missing values
    df = df.drop('education', axis=1)  # same as education.num
    df['capital_balance'] = df['capital.gain'] - df['capital.loss']
    df = df.drop('capital.gain', axis=1)
    df = df.drop('capital.loss', axis=1)
    # to avoid Multicollinearity problem with relationship. relationship had stronger connection to income
    df = df.drop('marital.status', axis=1)

    return df


if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    df = read_data()
    df = fix_value_spaces_and_names(df)
    """ -----------------------------  """
    df = handle_mis_val(df)

    print(cramers_v(df['income'], df['relationship']))
    print(cramers_v(df['marital.status'], df['income']))
    print(cramers_v(df['marital.status'], df['relationship']))
    # -------- descriptive statistics: --------
    # plot_descriptive_statistics()
    df = remove_add_variables(df)
    df = bins(df)
    # x = df.groupby(['hours.per.week', 'income']).size().reset_index(name='count')
    # print(x)
    # -------- prepare data to models: --------
    plot_descriptive_statistics()
    y = df.pop('income')
    orig_names = df.columns.tolist()  # column names before dummies
    print("_______________________________________")
    df = pd.get_dummies(df, columns=['workclass', 'occupation', 'relationship', 'race',
                                     'sex'])
    df.columns = df.columns.tolist()
    # df[['age', 'fnlwgt', 'hours.per.week']] = df[['age', 'fnlwgt', 'hours.per.week']].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # ------- oversample -----------
    oversample = RandomOverSampler()
    X_train_overSample, y_train_overSample = oversample.fit_resample(X_train, y_train)
    X_train, y_train = X_train_overSample, y_train_overSample
    # --------
    stats, rocs = [], []  # stats = { 'recall', 'precision','f1_score', 'accuracy', 'roc'},  rocs = [TP...]
    # TODO
    # scaler = MinMaxScaler()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    rf_pipe(orig_names)  # (x train,x test, y_train,y_test) both X with dummies.
    # XGBoost_pipe()
    # NN_pipe()
    # svm_pipe(X_train, X_test)
    # plot_stats(stats)
    # plot_roc_curve(rocs)  # ALL the rocs