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

    def test_cluster_profile_respects_user_cluster_count(self):
        profile, _ = dash_app.cluster_profile(dash_app.df_full, 3)

        self.assertIn("Cluster", profile.columns)
        self.assertLessEqual(profile["Cluster"].nunique(), 3)

    def test_model_comparison_page_uses_loaded_model(self):
        original_load_model = dash_app.load_model
        original_model_files = dash_app.model_files

        class FakeModelPath:
            name = "regression_model.joblib"

        try:
            dash_app.load_model = lambda model_name: self.model
            dash_app.model_files = lambda: [FakeModelPath()]
            content = dash_app.layout_model_test_results("Maïs", "Centre", 20)
        finally:
            dash_app.load_model = original_load_model
            dash_app.model_files = original_model_files

        self.assertIsNotNone(content)

    def test_model_files_excludes_lstm_preprocessor_artifacts(self):
        original_models_dir = dash_app.MODELS_DIR
        try:
            from tempfile import TemporaryDirectory
            from pathlib import Path

            with TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                (tmp_path / "regression_model.joblib").write_text("model")
                (tmp_path / "lstm_preprocessor.joblib").write_text("preprocessor")
                dash_app.MODELS_DIR = tmp_path
                names = [path.name for path in dash_app.model_files()]
        finally:
            dash_app.MODELS_DIR = original_models_dir

        self.assertIn("regression_model.joblib", names)
        self.assertNotIn("lstm_preprocessor.joblib", names)
