from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

DATA_FILE = Path(__file__).with_name('train_1.csv')

def semeval_score(hazard_true, product_true,
                  hazard_pred, product_pred):

    hazard_true = np.asarray(hazard_true)
    product_true = np.asarray(product_true)
    hazard_pred = np.asarray(hazard_pred)
    product_pred = np.asarray(product_pred)

    hazard_f1 = f1_score(
        hazard_true, hazard_pred, average="macro"
    )

    correct_hazards = hazard_pred == hazard_true

    if correct_hazards.any():
        product_f1 = f1_score(
            product_true[correct_hazards],
            product_pred[correct_hazards],
            average="macro",
            zero_division=0
        )
    else:
        product_f1 = 0.0

    print(f"SemEval Hazard F1: {hazard_f1:.4f}")
    print(f"SemEval Conditional Product F1: {product_f1:.4f}")

    return (hazard_f1 + product_f1) / 2



def main():
    train = pd.read_csv(DATA_FILE)
    test = pd.read_csv(
        Path(__file__).with_name('incidents_test.csv')
    )

    print("Train:", len(train))
    print("Test:", len(test))

    print("Test columns:", list(test.columns))

    train = train.dropna(
        subset=['title', 'hazard-category', 'product-category']
    ).copy()

    test = test.dropna(subset=['title']).copy()

    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(train['title'])
    X_test = vectorizer.transform(test['title'])

    hazard_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    )

    product_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=10.0
    )

    hazard_model.fit(X_train, train['hazard-category'])
    product_model.fit(X_train, train['product-category'])

    hazard_pred = hazard_model.predict(X_test)
    product_pred = product_model.predict(X_test)

    labels = ['hazard-category', 'product-category']

    if all(col in test.columns for col in labels) and \
       test[labels].notna().all().all():

        for name, true, pred in [
            ('Hazard', test['hazard-category'], hazard_pred),
            ('Product', test['product-category'], product_pred),
        ]:
            print(
                f'{name}: '
                f'accuracy={accuracy_score(true, pred):.4f}, '
                f'macro F1={f1_score(true, pred, average="macro", zero_division=0):.4f}'
            )

        score = semeval_score(
            test['hazard-category'],
            test['product-category'],
            hazard_pred,
            product_pred
        )

        print(f'SemEval ST1 Test Score: {score:.4f}')

    else:
        print("Test labels are missing.")
        print("Predictions were generated, but scoring is unavailable.")

    results = pd.DataFrame({
        'id': test['id'] if 'id' in test.columns else test.index,
        'hazard-category': hazard_pred,
        'product-category': product_pred
    })

    output_file = Path(__file__).with_name(
        'test_predictions.csv'
    )
    results.to_csv(output_file, index=False)

    print("Predictions saved to:", output_file)


if __name__ == '__main__':
    main()

