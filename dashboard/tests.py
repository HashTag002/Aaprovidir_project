import numpy as np
from django.test import SimpleTestCase
from sklearn.linear_model import LinearRegression

from dashboard import dash_app


class ForecastModelInputTests(SimpleTestCase):
    def setUp(self):
        self.df = dash_app.df_full[
            (dash_app.df_full["Produit_ID"] == "Maïs")
            & (dash_app.df_full["Region_Vente"] == "Centre")
        ].sort_values("Date").copy()
        self.last_row = self.df.iloc[-1].copy()
        self.feature_cols = dash_app.numeric_forecast_features(self.last_row)
        self.model = LinearRegression().fit(
            np.asarray(self.df[self.feature_cols]),
            self.df[dash_app.TARGET_COL],
        )

    def test_unnamed_model_input_excludes_app_helper_columns(self):
        all_numeric = [
            col
            for col in self.last_row.index
            if col != dash_app.TARGET_COL and dash_app.pd.api.types.is_number(self.last_row[col])
        ]

        x_pred = dash_app.build_model_input(self.last_row, self.model)

        self.assertEqual(len(self.feature_cols), 21)
        self.assertEqual(len(all_numeric), 22)
        self.assertNotIn("Score_Choc", self.feature_cols)
        self.assertEqual(x_pred.shape, (1, self.model.n_features_in_))

    def test_model_forecast_uses_unnamed_21_feature_model(self):
        original_load_model = dash_app.load_model
        try:
            dash_app.load_model = lambda model_name: self.model
            forecast_df, message = dash_app.model_forecast(
                self.df,
                3,
                "regression_model.joblib",
            )
        finally:
            dash_app.load_model = original_load_model

        self.assertEqual(len(forecast_df), 3)
        self.assertIn("Prévision générée", message)
        self.assertNotIn("Repli indicatif", message)
