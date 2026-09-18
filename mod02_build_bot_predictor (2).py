# Bot predictor: gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier

seed = 314


def train_model(X, y, seed=seed):
    """Train a gradient boosting classifier on the supplied data."""
    model = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=200,
        max_depth=1,
        subsample=1.0,
        min_samples_leaf=5,
        random_state=seed,
    )
    model.fit(X, y)
    return model
