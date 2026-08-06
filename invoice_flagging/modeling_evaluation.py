from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


def train_logistic_regression(X_train, y_train):

    model = LogisticRegression()

    model.fit(X_train, y_train)

    return model


def train_decision_tree(X_train, y_train):

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    return model


def tune_random_forest(X_train, y_train):

    param_dist = {

        "n_estimators":[100,200,300],

        "max_depth":[5,10,20,None],

        "min_samples_split":[2,5,10],

        "min_samples_leaf":[1,2,4],

        "max_features":["sqrt","log2"],

        "class_weight":[None,"balanced"]

    }

    random_search = RandomizedSearchCV(

        estimator=RandomForestClassifier(random_state=42),

        param_distributions=param_dist,

        n_iter=20,

        scoring="f1",

        cv=5,

        random_state=42,

        verbose=2,

        n_jobs=-1

    )

    random_search.fit(X_train,y_train)

    print(random_search.best_params_)

    print(random_search.best_score_)

    return random_search.best_estimator_


def evaluate_model(model,X_test,y_test,model_name):

    y_pred = model.predict(X_test)

    print("="*60)

    print(model_name)

    print("="*60)

    print(classification_report(y_test,y_pred))