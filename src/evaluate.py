from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2_score": r2_score(y_test, y_pred)
    }
    return metrics