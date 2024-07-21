import tensorflow as tf
import pandas as pd

class Predictor:
    def predict_single(self, model, X):
        f_mean, f_var = model.predict_f(X, full_cov=False)
        y_mean, y_var = model.predict_y(X)
        return f_mean, f_var, y_mean, y_var

    def predict_combined(self, alpha, beta, daily_model, weekly_model, monthly_model, X_daily, X_weekly, X_monthly):
        f_mean_daily, f_var_daily, y_mean_daily, y_var_daily = self.predict_single(daily_model, X_daily)
        f_mean_weekly, f_var_weekly, y_mean_weekly, y_var_weekly = self.predict_single(weekly_model, X_weekly)
        f_mean_monthly, f_var_monthly, y_mean_monthly, y_var_monthly = self.predict_single(monthly_model, X_monthly)

        f_mean_weekly_upsampled = self.upsample_predictions(X_daily, X_weekly, f_mean_weekly, period='w')
        f_mean_monthly_upsampled = self.upsample_predictions(X_daily, X_monthly, f_mean_monthly, period='m')

        f_var_weekly_upsampled = self.upsample_predictions(X_daily, X_weekly, f_var_weekly, period='w')
        f_var_monthly_upsampled = self.upsample_predictions(X_daily, X_monthly, f_var_monthly, period='m')

        y_mean_weekly_upsampled = self.upsample_predictions(X_daily, X_weekly, y_mean_weekly, period='w')
        y_mean_monthly_upsampled = self.upsample_predictions(X_daily, X_monthly, y_mean_monthly, period='m')

        y_var_weekly_upsampled = self.upsample_predictions(X_daily, X_weekly, y_var_weekly, period='w')
        y_var_monthly_upsampled = self.upsample_predictions(X_daily, X_monthly, y_var_monthly, period='m')

        f_combined_mean = alpha * f_mean_daily + beta * f_mean_weekly_upsampled + (1 - alpha - beta) * f_mean_monthly_upsampled
        f_combined_variance = alpha * f_var_daily + beta * f_var_weekly_upsampled + (1 - alpha - beta) * f_var_monthly_upsampled

        y_combined_mean = alpha * y_mean_daily + beta * y_mean_weekly_upsampled + (1 - alpha - beta) * y_mean_monthly_upsampled
        y_combined_variance = alpha * y_var_daily + beta * y_var_weekly_upsampled + (1 - alpha - beta) * y_var_monthly_upsampled

        return f_combined_mean, f_combined_variance, y_combined_mean, y_combined_variance

    def upsample_predictions(self, X_daily_tf, X_tf, predictions, period='d'):
        X_daily_np = X_daily_tf.numpy().reshape(-1)
        X_np = X_tf.numpy().reshape(-1)
        predictions_np = predictions.numpy().reshape(-1)

        data = {'date': X_np, 'prediction': predictions_np}
        df = pd.Series(data['prediction'], index=data['date'])

        if period in ['w', 'm']:
            df = df.reindex(X_daily_np).interpolate(method='linear')
        else:
            return predictions

        upsampled_predictions = df.values.reshape(-1, 1)
        upsampled_predictions_tf = tf.convert_to_tensor(upsampled_predictions, dtype=tf.float64)

        return upsampled_predictions_tf
